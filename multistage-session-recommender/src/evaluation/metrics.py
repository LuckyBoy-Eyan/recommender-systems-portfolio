"""计算候选覆盖率、单路召回率和多目标加权 Recall@K。"""

from __future__ import annotations

import pandas as pd


# OTTO 风格多目标指标权重；权重之和为 1。
WEIGHTS = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}


def evaluate_rankings(scored: pd.DataFrame, labels: pd.DataFrame, k: int) -> dict:
    """评估排序结果的分行为 Recall@K 和 Weighted Recall@K。

    参数：
        scored:
            已按每个 Session 分数降序排列的候选表，包含 ``session、aid、score``。
            函数直接取每个 Session 的前 ``k`` 行，因此调用方必须预先正确排序。
        labels:
            每个 Session 一行的真实目标，包含 ``session、target_aid、target_type``。
        k:
            每个 Session 参与评估的推荐列表长度。

    返回：
        指标字典，包含 ``recall_clicks@k``、``recall_carts@k``、
        ``recall_orders@k`` 和 ``weighted_recall@k``。某类行为在标签中不存在时，
        该类 Recall 记为 0。
    """
    # 使用集合进行目标商品成员判断；每个 Session 当前只有一个目标。
    topk = scored.groupby("session").head(k).groupby("session")["aid"].apply(set)
    merged = labels.copy()
    merged["hit"] = merged.apply(
        lambda row: int(row["target_aid"] in topk.get(row["session"], set())), axis=1
    )
    recalls = merged.groupby("target_type")["hit"].mean().to_dict()
    weighted = sum(WEIGHTS[action] * recalls.get(action, 0.0) for action in WEIGHTS)
    metrics = {f"recall_{action}@{k}": recalls.get(action, 0.0) for action in WEIGHTS}
    metrics[f"weighted_recall@{k}"] = weighted
    return metrics


def source_recall(recalled: pd.DataFrame, labels: pd.DataFrame, k: int) -> dict:
    """分别计算每一路召回源独立的 Recall@K。

    参数：
        recalled:
            多路召回长表，包含 ``session、aid、source、source_rank``。
        labels:
            每个 Session 的真实目标表。
        k:
            每一路、每个 Session 只取内部排名前 ``k`` 个候选。

    返回：
        形如 ``{"recent_recall@20": 0.66, ...}`` 的字典。这里的分母是全部
        标签 Session，不区分目标行为类型。
    """
    output = {}
    for source, group in recalled.groupby("source"):
        ranked = group.sort_values(["session", "source_rank"]).groupby("session").head(k)
        candidates = ranked.groupby("session")["aid"].apply(set)
        hits = labels.apply(
            lambda row: int(row["target_aid"] in candidates.get(row["session"], set())), axis=1
        )
        output[f"{source}_recall@{k}"] = float(hits.mean())
    return output


def candidate_recall(recalled: pd.DataFrame, labels: pd.DataFrame) -> float:
    """计算四路合并候选集对真实目标的整体覆盖率。

    参数：
        recalled:
            多路召回长表。同一商品可被多路重复召回，函数会按 Session 转为商品集合。
        labels:
            每个 Session 一行的真实目标表。

    返回：
        被合并候选集命中的标签 Session 比例，范围为 ``[0, 1]``。

    口径说明：
        该指标使用每路完整候选集，没有执行 Top-K 截断，也没有按照行为类型加权，
        因此不能直接与 Weighted Recall@20 比较大小。
    """
    candidates = recalled.groupby("session")["aid"].apply(set)
    hits = labels.apply(
        lambda row: int(row["target_aid"] in candidates.get(row["session"], set())), axis=1
    )
    return float(hits.mean())


def candidate_recall_by_action(
    recalled: pd.DataFrame, labels: pd.DataFrame
) -> dict[str, float]:
    """计算分行为候选覆盖率及对应加权上限。

    整体 Candidate Recall 的分母混合了三种行为，而最终排序指标使用
    ``0.1/0.3/0.6`` 权重。两者不能直接比较；该函数提供与 Weighted Recall
    相同任务权重下的候选上限。
    """
    output = {}
    for action, weight in WEIGHTS.items():
        action_labels = labels[labels["target_type"] == action]
        value = (
            candidate_recall(recalled, action_labels)
            if not action_labels.empty
            else 0.0
        )
        output[f"candidate_recall_{action}"] = value
    output["weighted_candidate_recall"] = sum(
        WEIGHTS[action] * output[f"candidate_recall_{action}"]
        for action in WEIGHTS
    )
    return output
