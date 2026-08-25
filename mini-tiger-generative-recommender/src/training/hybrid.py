"""工业二阶段推荐：Semantic ID 生成候选，SASRec 仅重排候选。"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import ranking_metrics
from src.training.evaluate import (
    _device,
    build_prefix_trie,
    constrained_beam_candidates,
)


def _zscore(values: torch.Tensor) -> torch.Tensor:
    """按用户标准化不同模型的分数，使线性融合具有可解释权重。"""
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    if float(std) < 1e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def _masked_zscore(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """逐行标准化候选分数，并忽略 padding 候选。"""
    counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
    safe = values.masked_fill(~mask, 0.0)
    means = safe.sum(dim=1, keepdim=True) / counts
    centered = (values - means).masked_fill(~mask, 0.0)
    variances = (centered.square().sum(dim=1, keepdim=True) / counts).clamp_min(
        1e-8
    )
    return (centered / variances.sqrt()).masked_fill(~mask, 0.0)


def _synchronize(device: torch.device) -> None:
    """计时时只在 CUDA 上显式同步。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_generated_candidates_with_sasrec(
    semantic_model,
    sasrec_model,
    semantic_dataset,
    sasrec_dataset,
    item_codes,
    *,
    candidate_k: int = 200,
    beam_size: int = 500,
    candidate_ks: tuple[int, ...] = (50, 100, 200),
    final_ks: tuple[int, ...] = (5, 10, 20),
    fusion_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    exclude_seen: bool = True,
) -> dict:
    """评估生成候选的召回上限，以及 SASRec/融合分数的最终 Top-K。

    ``alpha=0`` 表示候选内纯 SASRec 精排，``alpha=1`` 表示保持生成概率排序。
    中间值使用逐用户 z-score 后的线性融合。
    """
    if len(semantic_dataset) != len(sasrec_dataset):
        raise ValueError("semantic and SASRec datasets must contain the same samples")
    if max(candidate_ks) > candidate_k:
        raise ValueError("candidate_ks cannot exceed candidate_k")
    if beam_size < candidate_k:
        raise ValueError("beam_size must be at least candidate_k")
    if any(not 0.0 <= alpha <= 1.0 for alpha in fusion_alphas):
        raise ValueError("fusion alphas must be between zero and one")

    runtime_device = _device(device)
    semantic_model = semantic_model.to(runtime_device).eval()
    sasrec_model = sasrec_model.to(runtime_device).eval()
    children, code_to_item = build_prefix_trie(item_codes)
    semantic_loader = DataLoader(semantic_dataset, batch_size=batch_size)
    sasrec_loader = DataLoader(sasrec_dataset, batch_size=batch_size)

    candidate_rankings: list[list[int]] = []
    full_sasrec_rankings: list[list[int]] = []
    reranked: dict[float, list[list[int]]] = {
        float(alpha): [] for alpha in fusion_alphas
    }
    targets: list[int] = []
    generated_counts = []
    generation_seconds = 0.0
    sasrec_encoding_seconds = 0.0
    rerank_seconds = 0.0
    full_catalog_seconds = 0.0

    for semantic_batch, sasrec_batch in zip(semantic_loader, sasrec_loader):
        histories, _, semantic_targets, history_items = semantic_batch
        sas_histories, sas_targets, sas_history_items = sasrec_batch
        if not torch.equal(semantic_targets, sas_targets):
            raise ValueError("semantic and SASRec target order differs")
        if not torch.equal(history_items, sas_history_items):
            raise ValueError("semantic and SASRec history order differs")

        _synchronize(runtime_device)
        started = time.perf_counter()
        semantic_states = semantic_model.encode_history(
            histories.to(runtime_device)
        )
        batch_candidates: list[list[tuple[int, float]]] = []
        for row in range(len(histories)):
            seen = (
                set(history_items[row][history_items[row].ge(0)].tolist())
                if exclude_seen
                else set()
            )
            batch_candidates.append(
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
        sas_states = sasrec_model.encode_history(sas_histories.to(runtime_device))
        _synchronize(runtime_device)
        sasrec_encoding_seconds += time.perf_counter() - started

        _synchronize(runtime_device)
        started = time.perf_counter()
        max_candidates = max((len(row) for row in batch_candidates), default=0)
        item_matrix = torch.zeros(
            (len(batch_candidates), max_candidates),
            dtype=torch.long,
            device=runtime_device,
        )
        semantic_matrix = torch.zeros(
            (len(batch_candidates), max_candidates),
            dtype=torch.float32,
            device=runtime_device,
        )
        candidate_mask = torch.zeros(
            (len(batch_candidates), max_candidates),
            dtype=torch.bool,
            device=runtime_device,
        )
        for row, candidates in enumerate(batch_candidates):
            items = [item for item, _ in candidates]
            candidate_rankings.append(items)
            generated_counts.append(len(items))
            if items:
                length = len(items)
                item_matrix[row, :length] = torch.tensor(
                    items, dtype=torch.long, device=runtime_device
                )
                semantic_matrix[row, :length] = torch.tensor(
                    [score for _, score in candidates],
                    dtype=torch.float32,
                    device=runtime_device,
                )
                candidate_mask[row, :length] = True

        if max_candidates:
            sas_scores = sasrec_model.score_candidates(
                sas_states, item_matrix
            ).float()
            semantic_normalized = _masked_zscore(
                semantic_matrix, candidate_mask
            )
            sas_normalized = _masked_zscore(sas_scores, candidate_mask)
            topk = min(max(final_ks), max_candidates)
            for alpha, rankings in reranked.items():
                fused = (
                    alpha * semantic_normalized
                    + (1.0 - alpha) * sas_normalized
                ).masked_fill(~candidate_mask, -torch.inf)
                orders = fused.topk(topk, dim=1).indices.cpu().tolist()
                for row, order in enumerate(orders):
                    length = len(batch_candidates[row])
                    rankings.append(
                        [
                            batch_candidates[row][index][0]
                            for index in order
                            if index < length
                        ]
                    )
        else:
            for rankings in reranked.values():
                rankings.extend([[] for _ in batch_candidates])
        _synchronize(runtime_device)
        rerank_seconds += time.perf_counter() - started

        # 同一批用户做全目录 SASRec，作为候选精排的效果与成本上界。
        _synchronize(runtime_device)
        started = time.perf_counter()
        full_scores = sasrec_model.score_all_items(sas_states).float().cpu()
        if exclude_seen:
            for row, seen_items in enumerate(history_items):
                seen_items = seen_items[seen_items.ge(0)]
                full_scores[row, seen_items] = -torch.inf
        full_topk = min(max(final_ks), sasrec_model.num_items)
        full_sasrec_rankings.extend(
            full_scores.topk(full_topk, dim=1).indices.tolist()
        )
        _synchronize(runtime_device)
        full_catalog_seconds += time.perf_counter() - started
        targets.extend(semantic_targets.tolist())

    candidate_metrics = ranking_metrics(
        candidate_rankings, targets, list(candidate_ks)
    )
    return {
        "candidate_k": candidate_k,
        "beam_size": beam_size,
        "average_generated_candidates": float(np.mean(generated_counts)),
        "candidate_metrics": candidate_metrics,
        "rerank_metrics": {
            f"alpha={alpha:g}": ranking_metrics(
                rankings, targets, list(final_ks)
            )
            for alpha, rankings in reranked.items()
        },
        "full_catalog_sasrec_metrics": ranking_metrics(
            full_sasrec_rankings, targets, list(final_ks)
        ),
        "timing_seconds": {
            "generation": generation_seconds,
            "sasrec_history_encoding": sasrec_encoding_seconds,
            "candidate_scoring_and_fusion": rerank_seconds,
            "full_catalog_sasrec_scoring": full_catalog_seconds,
            "hybrid_total": (
                generation_seconds + sasrec_encoding_seconds + rerank_seconds
            ),
            "full_catalog_sasrec_total": (
                sasrec_encoding_seconds + full_catalog_seconds
            ),
            "users": len(targets),
        },
    }
