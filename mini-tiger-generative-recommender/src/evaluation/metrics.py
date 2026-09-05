"""单一真实目标场景下的 Top-K 推荐排序指标。

training/evaluate.py 的模型评估和热门基线都会调用 ranking_metrics，因此所有
实验使用同一套指标实现。
"""

from __future__ import annotations

import math

import torch


def ranking_metrics(rankings: list[list[int]], targets: list[int], ks: list[int]) -> dict:
    """
    根据推荐排序列表和真实目标物品，计算 HitRate、NDCG、MRR。

    rankings[i] 是第 i 个样本的推荐物品列表，越靠前分数越高。
    targets[i] 是第 i 个样本真实发生的下一个物品。

    参数:
        rankings: 每条测试样本的完整或截断推荐 item index 列表。
        targets: 每条测试样本唯一的真实下一个 item index。
        ks: 指标截断位置列表，例如 [5, 10, 20]。

    返回:
        包含 hitrate@k、ndcg@k 和 mrr@k 的扁平字典。

    调用:
        training.evaluate.evaluate_model 和 evaluate_popularity。

    指标解释:
        HitRate: 真实物品是否进入前 k。
        NDCG: 命中位置越靠前越高，折扣为 1/log2(rank+1)。
        MRR: 命中位置的倒数，即 1/rank。
    """
    metrics = {}
    for k in ks:
        # hits 统计 top-k 是否命中；ndcg 和 reciprocal 关注命中位置是否靠前。
        hits, ndcg, reciprocal = 0, 0.0, 0.0
        for ranking, target in zip(rankings, targets):
            # 只看前 k 个推荐物品。
            top = ranking[:k]
            if target in top:
                # rank 从 1 开始，rank 越小表示真实物品排得越靠前。
                rank = top.index(target) + 1
                hits += 1
                # NDCG 给靠前命中更高奖励，位置越靠后折扣越大。
                ndcg += 1.0 / math.log2(rank + 1)
                # MRR 只关心第一个正确答案的位置，越靠前得分越高。
                reciprocal += 1.0 / rank
        # 避免空测试集导致除以 0；正常情况下 count 就是测试样本数。
        count = max(len(targets), 1)
        metrics[f"hitrate@{k}"] = hits / count
        metrics[f"ndcg@{k}"] = ndcg / count
        metrics[f"mrr@{k}"] = reciprocal / count
    return metrics


def single_positive_auc(
    scores: torch.Tensor, targets: torch.Tensor
) -> tuple[float, int, list[float]]:
    """计算单目标、全目录排序的 AUC 充分统计量。

    ``-inf`` 表示被排除的候选。比较只发生在同一用户内部；目标与负物品同分时
    记 0.5。返回正确排序对的加权数量、有效正负对数量及逐用户 AUC。
    """
    scores = scores.float().cpu()
    targets = targets.long().cpu()
    target_scores = scores.gather(1, targets[:, None]).squeeze(1)
    eligible = torch.isfinite(scores)
    negatives = eligible.clone()
    negatives.scatter_(1, targets[:, None], False)
    better = (target_scores[:, None] > scores) & negatives
    tied = (target_scores[:, None] == scores) & negatives
    concordant = better.sum(dim=1).double() + 0.5 * tied.sum(dim=1).double()
    counts = negatives.sum(dim=1)
    valid = counts > 0
    user_auc = (concordant[valid] / counts[valid]).tolist()
    return float(concordant[valid].sum()), int(counts[valid].sum()), user_auc
