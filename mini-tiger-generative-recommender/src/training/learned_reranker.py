"""为 Gen-Rerank 构造候选特征，并训练无泄漏的学习式精排器。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import ranking_metrics
from src.training.evaluate import (
    _device,
    build_prefix_trie,
    constrained_beam_candidates,
)
from src.training.hybrid import _masked_zscore, _synchronize


FEATURE_NAMES = (
    "generation_zscore",
    "sasrec_zscore",
    "generation_rank_discount",
    "sasrec_rank_discount",
    "item_popularity",
    "history_length",
    "last_item_embedding_similarity",
    "last_item_semantic_prefix_ratio",
)


@dataclass
class CandidateCache:
    """可重复使用的候选及特征矩阵，避免每次训练都重新 Beam Search。"""

    candidate_items: np.ndarray
    features: np.ndarray
    mask: np.ndarray
    targets: np.ndarray
    target_positions: np.ndarray
    metadata: dict

    def validate(self) -> None:
        rows, candidates = self.candidate_items.shape
        if self.features.shape[:2] != (rows, candidates):
            raise ValueError("feature and candidate shapes differ")
        if self.mask.shape != (rows, candidates):
            raise ValueError("mask and candidate shapes differ")
        if self.targets.shape != (rows,) or self.target_positions.shape != (rows,):
            raise ValueError("target arrays must contain one value per row")
        if self.features.shape[2] != len(FEATURE_NAMES):
            raise ValueError("unexpected candidate feature dimension")
        if not np.isfinite(self.features[self.mask]).all():
            raise ValueError("candidate features contain NaN or infinity")


def item_popularity(train_sequences, num_items: int) -> np.ndarray:
    """只用训练时间窗统计物品热度，防止验证/测试信息泄漏。"""
    counts = np.zeros(num_items, dtype=np.float32)
    for sequence in train_sequences:
        np.add.at(counts, np.asarray(sequence, dtype=np.int64), 1.0)
    maximum = float(np.log1p(counts).max())
    return np.log1p(counts) / max(maximum, 1.0)


def _rank_discount(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """把分数转成 1/log2(rank+1)，排名第一为 1。"""
    masked = scores.masked_fill(~mask, -torch.inf)
    order = masked.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    positions = torch.arange(
        1, scores.shape[1] + 1, device=scores.device
    ).expand_as(order)
    ranks.scatter_(1, order, positions)
    discount = 1.0 / torch.log2(ranks.float() + 1.0)
    return discount.masked_fill(~mask, 0.0)


def _semantic_prefix_ratio(
    candidate_items: torch.Tensor,
    history_items: torch.Tensor,
    item_codes: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """计算候选与最近一次行为从第一层开始共享多少比例的 Semantic ID。"""
    lengths = history_items.ge(0).sum(dim=1)
    # 数据集使用左 padding，所以最近一次真实行为始终在最后一列。
    last_items = history_items[:, -1]
    last_codes = item_codes[last_items.clamp_min(0)]
    candidate_codes = item_codes[candidate_items]
    equal = candidate_codes.eq(last_codes[:, None, :])
    prefix = equal.long().cumprod(dim=2).sum(dim=2).float()
    ratios = prefix / item_codes.shape[1]
    return ratios.masked_fill(~mask | lengths[:, None].eq(0), 0.0)


@torch.no_grad()
def build_candidate_cache(
    semantic_model,
    sasrec_model,
    semantic_dataset,
    sasrec_dataset,
    item_codes,
    popularity,
    *,
    candidate_k: int = 200,
    beam_size: int = 500,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    exclude_seen: bool = True,
    split_name: str = "unknown",
) -> CandidateCache:
    """生成候选并提取八个轻量特征。

    这里不使用目标物品构造任何输入特征；目标只用于保存监督标签，因此同一函数
    可以安全地处理精排训练、校准和最终测试数据。
    """
    if len(semantic_dataset) != len(sasrec_dataset):
        raise ValueError("semantic and SASRec datasets must contain the same samples")
    if beam_size < candidate_k:
        raise ValueError("beam_size must be at least candidate_k")
    runtime_device = _device(device)
    semantic_model = semantic_model.to(runtime_device).eval()
    sasrec_model = sasrec_model.to(runtime_device).eval()
    children, code_to_item = build_prefix_trie(item_codes)
    semantic_loader = DataLoader(semantic_dataset, batch_size=batch_size)
    sasrec_loader = DataLoader(sasrec_dataset, batch_size=batch_size)
    code_tensor = torch.as_tensor(item_codes, dtype=torch.long, device=runtime_device)
    popularity_tensor = torch.as_tensor(
        popularity, dtype=torch.float32, device=runtime_device
    )
    embedding = sasrec_model.item_embedding.weight[1:]
    normalized_embedding = nn.functional.normalize(embedding, dim=1)

    all_items, all_features, all_masks = [], [], []
    all_targets, all_target_positions = [], []
    generation_seconds = 0.0
    feature_seconds = 0.0

    for semantic_batch, sasrec_batch in zip(semantic_loader, sasrec_loader):
        histories, _, semantic_targets, history_items = semantic_batch
        sas_histories, sas_targets, sas_history_items = sasrec_batch
        if not torch.equal(semantic_targets, sas_targets):
            raise ValueError("semantic and SASRec target order differs")
        if not torch.equal(history_items, sas_history_items):
            raise ValueError("semantic and SASRec history order differs")

        _synchronize(runtime_device)
        started = time.perf_counter()
        semantic_states = semantic_model.encode_history(histories.to(runtime_device))
        generated = []
        for row in range(len(histories)):
            seen = (
                set(history_items[row][history_items[row].ge(0)].tolist())
                if exclude_seen
                else set()
            )
            generated.append(
                constrained_beam_candidates(
                    semantic_model,
                    semantic_states[row : row + 1],
                    children,
                    code_to_item,
                    beam_size=beam_size,
                    candidate_k=candidate_k,
                    seen_items=seen,
                )
            )
        _synchronize(runtime_device)
        generation_seconds += time.perf_counter() - started

        _synchronize(runtime_device)
        started = time.perf_counter()
        current = len(generated)
        items = torch.zeros(
            (current, candidate_k), dtype=torch.long, device=runtime_device
        )
        generation_scores = torch.zeros(
            (current, candidate_k), dtype=torch.float32, device=runtime_device
        )
        mask = torch.zeros(
            (current, candidate_k), dtype=torch.bool, device=runtime_device
        )
        for row, candidates in enumerate(generated):
            length = len(candidates)
            if length:
                items[row, :length] = torch.tensor(
                    [item for item, _ in candidates], device=runtime_device
                )
                generation_scores[row, :length] = torch.tensor(
                    [score for _, score in candidates], device=runtime_device
                )
                mask[row, :length] = True

        sas_histories_device = sas_histories.to(runtime_device)
        sas_states = sasrec_model.encode_history(sas_histories_device)
        sas_scores = sasrec_model.score_candidates(sas_states, items).float()
        generation_z = _masked_zscore(generation_scores, mask)
        sasrec_z = _masked_zscore(sas_scores, mask)
        generation_rank = _rank_discount(generation_scores, mask)
        sasrec_rank = _rank_discount(sas_scores, mask)
        popularity_feature = popularity_tensor[items].masked_fill(~mask, 0.0)
        lengths = sas_histories_device.ne(0).sum(dim=1).float()
        history_length = (
            lengths / sas_histories_device.shape[1]
        )[:, None].expand_as(sas_scores).masked_fill(~mask, 0.0)

        last_tokens = sas_histories_device[:, -1]
        last_vectors = normalized_embedding[(last_tokens - 1).clamp_min(0)]
        candidate_vectors = normalized_embedding[items]
        last_similarity = torch.einsum(
            "bd,bkd->bk", last_vectors, candidate_vectors
        ).masked_fill(~mask | lengths[:, None].eq(0), 0.0)
        prefix_ratio = _semantic_prefix_ratio(
            items,
            sas_history_items.to(runtime_device),
            code_tensor,
            mask,
        )
        features = torch.stack(
            (
                generation_z,
                sasrec_z,
                generation_rank,
                sasrec_rank,
                popularity_feature,
                history_length,
                last_similarity,
                prefix_ratio,
            ),
            dim=2,
        )
        targets_device = semantic_targets.to(runtime_device)
        matches = items.eq(targets_device[:, None]) & mask
        positions = torch.where(
            matches.any(dim=1),
            matches.float().argmax(dim=1).long(),
            torch.full((current,), -1, dtype=torch.long, device=runtime_device),
        )
        all_items.append(items.cpu().numpy().astype(np.int32))
        all_features.append(features.cpu().numpy().astype(np.float32))
        all_masks.append(mask.cpu().numpy())
        all_targets.append(semantic_targets.numpy().astype(np.int32))
        all_target_positions.append(positions.cpu().numpy().astype(np.int32))
        _synchronize(runtime_device)
        feature_seconds += time.perf_counter() - started

    cache = CandidateCache(
        candidate_items=np.concatenate(all_items),
        features=np.concatenate(all_features),
        mask=np.concatenate(all_masks),
        targets=np.concatenate(all_targets),
        target_positions=np.concatenate(all_target_positions),
        metadata={
            "format_version": 1,
            "split": split_name,
            "candidate_k": candidate_k,
            "beam_size": beam_size,
            "users": len(semantic_dataset),
            "feature_names": list(FEATURE_NAMES),
            "target_used_as_feature": False,
            "generation_seconds": generation_seconds,
            "feature_seconds": feature_seconds,
        },
    )
    cache.validate()
    return cache


def save_candidate_cache(cache: CandidateCache, path: str | Path) -> None:
    """原子保存压缩候选缓存。"""
    cache.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            candidate_items=cache.candidate_items,
            features=cache.features,
            mask=cache.mask,
            targets=cache.targets,
            target_positions=cache.target_positions,
            metadata=np.asarray(json.dumps(cache.metadata)),
        )
    os.replace(temporary, destination)


def load_candidate_cache(path: str | Path) -> CandidateCache:
    """加载并验证候选缓存。"""
    with np.load(path, allow_pickle=False) as stored:
        cache = CandidateCache(
            candidate_items=stored["candidate_items"],
            features=stored["features"],
            mask=stored["mask"],
            targets=stored["targets"],
            target_positions=stored["target_positions"],
            metadata=json.loads(str(stored["metadata"])),
        )
    cache.validate()
    return cache


class _CacheRows(Dataset):
    def __init__(self, cache: CandidateCache, indices):
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        return (
            torch.from_numpy(self.cache.features[row]),
            torch.from_numpy(self.cache.mask[row]),
            torch.tensor(self.cache.target_positions[row], dtype=torch.long),
        )


@torch.no_grad()
def evaluate_candidate_ranker(
    model,
    cache: CandidateCache,
    *,
    indices=None,
    ks: tuple[int, ...] = (5, 10, 20),
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> dict:
    """评估学习式精排，并同时报告候选覆盖上限。"""
    runtime_device = _device(device)
    model = model.to(runtime_device).eval()
    selected = (
        np.arange(len(cache.targets), dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    rankings = []
    targets = cache.targets[selected].tolist()
    for start in range(0, len(selected), batch_size):
        rows = selected[start : start + batch_size]
        features = torch.from_numpy(cache.features[rows]).to(runtime_device)
        mask = torch.from_numpy(cache.mask[rows]).to(runtime_device)
        scores = model(features, mask)
        topk = min(max(ks), scores.shape[1])
        orders = scores.topk(topk, dim=1).indices.cpu().numpy()
        for row, order in zip(rows, orders):
            rankings.append(cache.candidate_items[row, order].tolist())
    metrics = ranking_metrics(rankings, targets, list(ks))
    positions = cache.target_positions[selected]
    metrics["candidate_recall"] = float(np.mean(positions >= 0))
    metrics["candidate_hits"] = int(np.sum(positions >= 0))
    metrics["users"] = int(len(selected))
    return metrics


def train_candidate_ranker(
    model,
    cache: CandidateCache,
    train_indices,
    calibration_indices,
    *,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    monitor: str = "ndcg@20",
    guard_metric: str | None = None,
    ks: tuple[int, ...] = (5, 10, 20),
    device: str | torch.device = "cpu",
) -> tuple[object, list[dict]]:
    """在候选命中的样本上做 listwise 交叉熵，并用独立校准集早停。"""
    runtime_device = _device(device)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    train_indices = train_indices[cache.target_positions[train_indices] >= 0]
    if not len(train_indices):
        raise ValueError("no positive target exists in the training candidates")
    loader = DataLoader(
        _CacheRows(cache, train_indices),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=runtime_device.type == "cuda",
    )
    model = model.to(runtime_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # epoch 0 是固定融合基线。即使后续训练没有改善，也会回滚到这个状态。
    initial_metrics = evaluate_candidate_ranker(
        model,
        cache,
        indices=calibration_indices,
        ks=ks,
        batch_size=batch_size,
        device=runtime_device,
    )
    guard_metric = guard_metric or f"recall@{max(ks)}"
    best_score = float(initial_metrics[monitor])
    guard_floor = float(initial_metrics[guard_metric])
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    stale = 0
    history = [
        {
            "epoch": 0,
            "loss": None,
            "calibration": initial_metrics,
            "note": "fixed-fusion initialization",
        }
    ]
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        samples = 0
        for features, mask, positions in loader:
            features = features.to(runtime_device)
            mask = mask.to(runtime_device)
            positions = positions.to(runtime_device)
            scores = model(features, mask)
            loss = nn.functional.cross_entropy(scores, positions)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(features)
            samples += len(features)
        metrics = evaluate_candidate_ranker(
            model,
            cache,
            indices=calibration_indices,
            ks=ks,
            batch_size=batch_size,
            device=runtime_device,
        )
        record = {
            "epoch": epoch + 1,
            "loss": total_loss / samples,
            "calibration": metrics,
        }
        history.append(record)
        score = float(metrics[monitor])
        if score > best_score and float(metrics[guard_metric]) >= guard_floor:
            best_score = score
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        print(
            f"epoch={epoch + 1} loss={record['loss']:.6f} "
            f"{monitor}={score:.6f}"
        )
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def diagnose_candidate_cache(cache: CandidateCache) -> dict:
    """报告候选覆盖和目标生成位置分布，定位召回与精排瓶颈。"""
    positions = cache.target_positions
    hits = positions >= 0
    result = {
        "users": int(len(positions)),
        "candidate_hits": int(hits.sum()),
        "candidate_recall": float(hits.mean()),
    }
    if hits.any():
        one_based = positions[hits] + 1
        hit_rows = np.flatnonzero(hits)
        target_sas_discounts = cache.features[
            hit_rows, positions[hits], FEATURE_NAMES.index("sasrec_rank_discount")
        ]
        target_sas_ranks = np.rint(
            np.power(2.0, 1.0 / target_sas_discounts) - 1.0
        )
        result.update(
            {
                "target_generation_rank_median": float(np.median(one_based)),
                "target_generation_rank_p75": float(np.percentile(one_based, 75)),
                "target_generation_rank_p90": float(np.percentile(one_based, 90)),
                "target_sasrec_rank_median": float(np.median(target_sas_ranks)),
                "target_sasrec_rank_p75": float(
                    np.percentile(target_sas_ranks, 75)
                ),
                "target_sasrec_rank_p90": float(
                    np.percentile(target_sas_ranks, 90)
                ),
            }
        )
    for boundary in (20, 50, 100, 200, 500):
        if boundary <= cache.candidate_items.shape[1]:
            result[f"candidate_recall@{boundary}"] = float(
                np.mean((positions >= 0) & (positions < boundary))
            )
    return result
