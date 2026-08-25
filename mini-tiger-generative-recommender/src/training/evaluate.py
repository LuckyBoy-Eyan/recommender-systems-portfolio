"""KuaiRec 规模下的目录约束评估。

提供两种模式：

1. ``exact``：缓存一次 Transformer 历史编码，再分块枚举全目录。结果精确，
   适合验证集、较小目录以及 Beam Search 对照。
2. ``beam``：用合法 Semantic ID 前缀 Trie 约束 Beam Search。不会生成目录外
   编码，计算量不再与全商品数线性相乘，适合 KuaiRec 的万级目录。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import ranking_metrics


def _device(value: str | torch.device) -> torch.device:
    """把配置中的设备字符串转为可用 torch.device。"""
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


@torch.no_grad()
def evaluate_model_exact(
    model,
    dataset,
    item_codes,
    ks: list[int],
    *,
    batch_size: int = 32,
    catalog_chunk_size: int = 512,
    device: str | torch.device = "cpu",
    exclude_seen: bool = False,
) -> dict:
    """通过分块全目录打分计算精确 Top-K 指标。

    与旧实现相比，Transformer 对每条历史只计算一次；不同候选只重复运行很轻的
    GRU 解码器。候选再按 catalog_chunk_size 分块，避免一次展开
    ``batch_size × num_items`` 导致内存峰值。
    """
    runtime_device = _device(device)
    model = model.to(runtime_device)
    model.eval()
    rankings, targets = [], []
    valid_codes = torch.as_tensor(item_codes, dtype=torch.long)
    num_items = len(valid_codes)
    topk = min(max(ks), num_items)

    for history, _, target_items, history_items in DataLoader(
        dataset, batch_size=batch_size
    ):
        history = history.to(runtime_device)
        state = model.encode_history(history)
        current_batch = len(history)
        # 分数矩阵保存在 CPU；只有当前候选块进入加速设备。
        scores = torch.empty((current_batch, num_items), dtype=torch.float32)

        for start in range(0, num_items, catalog_chunk_size):
            end = min(start + catalog_chunk_size, num_items)
            candidate_codes = valid_codes[start:end].to(runtime_device)
            chunk_size = len(candidate_codes)
            expanded_state = state.repeat_interleave(chunk_size, dim=0)
            expanded_codes = candidate_codes.repeat(current_batch, 1)
            logits = model.decode(expanded_state, expanded_codes)
            candidate_scores = torch.zeros(
                current_batch * chunk_size, device=runtime_device
            )
            for level, level_logits in enumerate(logits):
                candidate_scores += level_logits.log_softmax(-1).gather(
                    1, expanded_codes[:, level : level + 1]
                ).squeeze(1)
            scores[:, start:end] = candidate_scores.view(
                current_batch, chunk_size
            ).cpu()

        if exclude_seen:
            for row, seen_items in enumerate(history_items):
                seen_items = seen_items[seen_items.ge(0)]
                scores[row, seen_items] = -torch.inf

        rankings.extend(scores.topk(topk, dim=1).indices.tolist())
        targets.extend(target_items.tolist())
    return ranking_metrics(rankings, targets, ks)


def build_prefix_trie(item_codes) -> tuple[dict[tuple[int, ...], tuple[int, ...]], dict]:
    """把合法商品编码构造成 ``前缀 -> 下一步合法 token`` 的紧凑 Trie。

    返回:
        children: 例如前缀 ``(2, 5)`` 映射到合法 tail ``(0, 1, 2)``。
        code_to_item: 完整编码 tuple 到连续 item index 的映射。
    """
    codes = np.asarray(item_codes, dtype=np.int64)
    child_sets: dict[tuple[int, ...], set[int]] = defaultdict(set)
    code_to_item = {}
    for item, row in enumerate(codes.tolist()):
        code = tuple(int(token) for token in row)
        code_to_item[code] = item
        for level, token in enumerate(code):
            child_sets[code[:level]].add(token)
    children = {
        prefix: tuple(sorted(tokens)) for prefix, tokens in child_sets.items()
    }
    return children, code_to_item


@torch.no_grad()
def constrained_beam_search(
    model,
    state: torch.Tensor,
    children: dict[tuple[int, ...], tuple[int, ...]],
    code_to_item: dict[tuple[int, ...], int],
    *,
    beam_size: int,
    topk: int,
    seen_items: set[int] | None = None,
) -> list[int]:
    """为单个用户执行合法前缀约束的自回归 Beam Search。

    beam 中每个元素保存“已生成前缀、累计 log 概率、当前 GRU 状态”。每一级
    只扩展 Trie 允许的 token，因此最终叶子一定对应真实视频。
    """
    return [
        item
        for item, _ in constrained_beam_candidates(
            model,
            state,
            children,
            code_to_item,
            beam_size=beam_size,
            candidate_k=topk,
            seen_items=seen_items,
        )
    ]


@torch.no_grad()
def constrained_beam_candidates(
    model,
    state: torch.Tensor,
    children: dict[tuple[int, ...], tuple[int, ...]],
    code_to_item: dict[tuple[int, ...], int],
    *,
    beam_size: int,
    candidate_k: int,
    seen_items: set[int] | None = None,
) -> list[tuple[int, float]]:
    """生成带累计 log 概率的合法 Item 候选，供二阶段排序器使用。"""
    if state.shape[0] != 1:
        raise ValueError("constrained_beam_candidates expects one encoded user state")
    if beam_size < candidate_k:
        raise ValueError("beam_size must be at least candidate_k")
    beams = [((), 0.0, state)]

    for level, head in enumerate(model.heads):
        parent_states = torch.cat([beam[2] for beam in beams], dim=0)
        if level == 0:
            previous = model.start_token.expand(len(beams), -1)
        else:
            previous_tokens = torch.tensor(
                [beam[0][-1] for beam in beams],
                dtype=torch.long,
                device=state.device,
            )
            previous = model.target_embeddings[level - 1](previous_tokens)
        next_states = model.decoder_cell(previous, parent_states)
        log_probs = head(next_states).log_softmax(-1)

        expanded = []
        for parent_index, (prefix, score, _) in enumerate(beams):
            allowed = children.get(prefix, ())
            for token in allowed:
                expanded.append(
                    (
                        (*prefix, token),
                        score + float(log_probs[parent_index, token]),
                        next_states[parent_index : parent_index + 1],
                    )
                )
        expanded.sort(key=lambda candidate: candidate[1], reverse=True)
        beams = expanded[:beam_size]
        if not beams:
            break

    seen_items = seen_items or set()
    candidates = []
    for code, score, _ in beams:
        item = code_to_item.get(code)
        if item is not None and item not in seen_items:
            candidates.append((item, score))
            if len(candidates) == candidate_k:
                break
    return candidates


@torch.no_grad()
def evaluate_model_beam(
    model,
    dataset,
    item_codes,
    ks: list[int],
    *,
    beam_size: int = 100,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    exclude_seen: bool = False,
) -> dict:
    """使用合法前缀 Trie Beam Search 计算 Top-K 指标。"""
    runtime_device = _device(device)
    model = model.to(runtime_device)
    model.eval()
    children, code_to_item = build_prefix_trie(item_codes)
    requested_topk = max(ks)
    if beam_size < requested_topk:
        raise ValueError("beam_size must be at least max(ks)")

    rankings, targets = [], []
    for history, _, target_items, history_items in DataLoader(
        dataset, batch_size=batch_size
    ):
        states = model.encode_history(history.to(runtime_device))
        for row in range(len(history)):
            seen = (
                set(history_items[row][history_items[row].ge(0)].tolist())
                if exclude_seen
                else set()
            )
            rankings.append(
                constrained_beam_search(
                    model,
                    states[row : row + 1],
                    children,
                    code_to_item,
                    beam_size=beam_size,
                    topk=requested_topk,
                    seen_items=seen,
                )
            )
        targets.extend(target_items.tolist())
    return ranking_metrics(rankings, targets, ks)


def evaluate_model(
    model,
    dataset,
    item_codes,
    ks: list[int],
    *,
    mode: str = "exact",
    **kwargs,
) -> dict:
    """按配置分派到精确目录评估或 Trie Beam Search。"""
    if mode == "exact":
        return evaluate_model_exact(model, dataset, item_codes, ks, **kwargs)
    if mode == "beam":
        return evaluate_model_beam(model, dataset, item_codes, ks, **kwargs)
    raise ValueError(f"Unknown evaluation mode: {mode}")


def evaluate_popularity(
    dataset,
    train_sequences: list[list[int]],
    ks: list[int],
    *,
    exclude_seen: bool = False,
) -> dict:
    """评估训练集热门榜，并可为每条历史排除已经交互过的物品。"""
    counts = {}
    for sequence in train_sequences:
        for item in sequence:
            counts[item] = counts.get(item, 0) + 1
    global_ranking = [
        item for item, _ in sorted(counts.items(), key=lambda pair: -pair[1])
    ]

    rankings, targets = [], []
    for index in range(len(dataset)):
        _, _, target, history_items = dataset[index]
        if exclude_seen:
            seen = set(history_items[history_items.ge(0)].tolist())
            rankings.append([item for item in global_ranking if item not in seen])
        else:
            rankings.append(global_ranking)
        targets.append(int(target))
    return ranking_metrics(rankings, targets, ks)


@torch.no_grad()
def evaluate_sasrec(
    model,
    dataset,
    ks: list[int],
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    exclude_seen: bool = False,
) -> dict:
    """对 SASRec 做全目录精确 Top-K 评估。"""
    runtime_device = _device(device)
    model = model.to(runtime_device)
    model.eval()
    rankings, targets = [], []
    topk = min(max(ks), model.num_items)
    for history_tokens, target_items, history_items in DataLoader(
        dataset, batch_size=batch_size
    ):
        scores = model(history_tokens.to(runtime_device)).float().cpu()
        if exclude_seen:
            for row, seen_items in enumerate(history_items):
                seen_items = seen_items[seen_items.ge(0)]
                scores[row, seen_items] = -torch.inf
        rankings.extend(scores.topk(topk, dim=1).indices.tolist())
        targets.extend(target_items.tolist())
    return ranking_metrics(rankings, targets, ks)
