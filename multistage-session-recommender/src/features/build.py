"""把多路召回结果转换为排序模型可使用的特征表。"""

from __future__ import annotations

import pandas as pd


# RRF 平滑常数。各路候选数为 50，取 60 可避免头部单个名次差异过度放大。
RRF_K = 60


def build_candidate_features(
    history: pd.DataFrame, recalled: pd.DataFrame, reference_events: pd.DataFrame | None = None
) -> pd.DataFrame:
    """为每个 ``(session, aid)`` 候选构建召回、商品和会话特征。

    参数：
        history:
            当前待预测 Session 的本地可见历史。用于计算 Session 长度、不同商品数、
            候选商品在 Session 内出现次数等实时特征。
        recalled:
            多路召回长表，必须包含 ``session、aid、source、source_rank、
            source_score``。同一候选可因多路命中而出现多行。
        reference_events:
            用于计算全局商品统计的历史事件快照。为 ``None`` 时退化为使用
            ``history``；Point-in-Time 流程会显式传入严格早于快照时间的事件。

    返回：
        每个 ``(session, aid)`` 一行的数值特征表。缺失的召回源和历史统计填 0。

    主要特征：
        各召回源排名/分数、召回源数量、商品事件与转化统计、Session 长度、
        Session-Item 重复次数、最近出现时间差、最佳召回排名和 RRF 融合分数。
        各路原始分数仍分别保留给Shared-Bottom学习，但不会再跨来源直接求和。
    """
    reference_events = history if reference_events is None else reference_events
    # 将多行召回结果透视为一个候选一行、不同召回源各占一组列的宽表。
    source_features = recalled.pivot_table(
        index=["session", "aid"],
        columns="source",
        values=["source_rank", "source_score"],
        aggfunc={"source_rank": "min", "source_score": "max"},
    )
    source_features.columns = [f"{left}_{right}" for left, right in source_features.columns]
    source_features = source_features.reset_index()
    source_count = recalled.groupby(["session", "aid"])["source"].nunique().rename("source_count")
    source_features = source_features.merge(source_count, on=["session", "aid"], how="left")

    if reference_events.empty:
        # 最早快照可能没有任何全局历史，显式创建带类型的空表以保持特征结构稳定。
        item_stats = pd.DataFrame({
            "aid": pd.Series(dtype="int64"),
            "item_events": pd.Series(dtype="float64"),
            "item_sessions": pd.Series(dtype="float64"),
            "item_last_ts": pd.Series(dtype="float64"),
            "item_clicks": pd.Series(dtype="float64"),
            "item_carts": pd.Series(dtype="float64"),
            "item_orders": pd.Series(dtype="float64"),
        })
    else:
        # 商品事件量、覆盖 Session 数和最近事件时间。
        item_stats = reference_events.groupby("aid").agg(
            item_events=("aid", "size"),
            item_sessions=("session", "nunique"),
            item_last_ts=("ts", "max"),
        ).reset_index()
        # 分行为类型统计商品事件次数，列名变为 item_clicks/item_carts/item_orders。
        action_counts = reference_events.pivot_table(
            index="aid", columns="type", values="session", aggfunc="size", fill_value=0
        ).add_prefix("item_").reset_index()
        item_stats = item_stats.merge(action_counts, on="aid", how="left")
    for action in ("clicks", "carts", "orders"):
        # 某个快照可能完全没有某类行为，需要补齐固定特征列。
        column = f"item_{action}"
        if column not in item_stats:
            item_stats[column] = 0
    # 分母至少为 1，避免空统计产生除零错误。
    item_stats["cart_rate"] = item_stats["item_carts"] / item_stats["item_events"].clip(lower=1)
    item_stats["order_rate"] = item_stats["item_orders"] / item_stats["item_events"].clip(lower=1)
    # Session 级实时上下文特征。
    session_stats = history.groupby("session").agg(
        session_length=("aid", "size"),
        session_unique_items=("aid", "nunique"),
        session_last_ts=("ts", "max"),
    ).reset_index()
    # Session 与候选商品的交叉特征；未在历史中出现的候选合并后会填 0。
    pair_stats = history.groupby(["session", "aid"]).agg(
        in_session_count=("aid", "size"),
        pair_last_ts=("ts", "max"),
    ).reset_index()
    features = source_features.merge(item_stats, on="aid", how="left")
    features = features.merge(session_stats, on="session", how="left")
    features = features.merge(pair_stats, on=["session", "aid"], how="left")
    # 名称沿用 seconds_since_seen；其实际单位与输入 ts 一致，RetailRocket 中为毫秒。
    features["seconds_since_seen"] = features["session_last_ts"] - features["pair_last_ts"]
    rank_columns = [column for column in features if column.startswith("source_rank_")]
    # 排名越小越好；0 表示缺失，计算最优排名时先替换为足够大的哨兵值。
    features["best_source_rank"] = features[rank_columns].replace(0, 10_000).min(axis=1)
    valid_ranks = features[rank_columns].where(features[rank_columns] > 0)
    features["rrf_score"] = (1.0 / (RRF_K + valid_ranks)).fillna(0.0).sum(axis=1)
    return features.fillna(0)
