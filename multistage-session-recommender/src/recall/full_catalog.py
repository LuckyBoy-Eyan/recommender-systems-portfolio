"""全量目录冻结索引召回：热度、有向转移、ItemCF 和类目。"""

from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np
import pandas as pd

from src.recall.sources import build_itemcf


ACTION_WEIGHTS = {"clicks": 1.0, "carts": 3.0, "orders": 6.0}
DAY_MS = 86_400_000
PRODUCTION_ROUTE_LIMITS = {
    "recent": 30,
    "hybrid_popular": 100,
    "category": 150,
    "itemcf": 200,
    "transition": 150,
}


def _ranked_items(scores: pd.Series, limit: int = 200) -> list[int]:
    return (
        scores.sort_values(ascending=False, kind="mergesort")
        .head(limit).index.astype(int).tolist()
    )


def _merge_with_budget(
    parts: list[tuple[list[int], int]], fallback: list[int], budget: int
) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for items, quota in parts:
        added = 0
        for aid in items:
            aid = int(aid)
            if aid in seen:
                continue
            output.append(aid)
            seen.add(aid)
            added += 1
            if added >= quota or len(output) >= budget:
                break
    for aid in fallback:
        if len(output) >= budget:
            break
        aid = int(aid)
        if aid not in seen:
            output.append(aid)
            seen.add(aid)
    return output


def build_hybrid_popularity(events: pd.DataFrame, cutoff_ts: int) -> dict[str, list[int]]:
    """构建行为分层、趋势及少量新品混合的因果热门池。"""
    reference = events[events["ts"] < cutoff_ts].copy().reset_index(drop=True)
    reference["action_weight"] = reference["type"].map(ACTION_WEIGHTS).astype(float)
    fallback = _ranked_items(reference.groupby("aid")["action_weight"].sum())
    action_lists = {
        action: _ranked_items(reference.loc[reference["type"].eq(action), "aid"].value_counts())
        for action in ACTION_WEIGHTS
    }

    age_days = (cutoff_ts - reference["ts"].to_numpy()) / DAY_MS
    reference["trend_score"] = reference["action_weight"] * (
        np.exp(-age_days) + 0.3 * np.exp(-age_days / 7.0)
    )
    trend = _ranked_items(reference.groupby("aid")["trend_score"].sum())

    first_seen = reference.groupby("aid")["ts"].min()
    recent_items = set(first_seen[first_seen >= cutoff_ts - 30 * DAY_MS].index.astype(int))
    new_reference = reference[reference["aid"].isin(recent_items)].copy()
    item_age = (cutoff_ts - new_reference["aid"].map(first_seen)) / DAY_MS
    event_age = (cutoff_ts - new_reference["ts"]) / DAY_MS
    new_reference["new_score"] = (
        new_reference["action_weight"] * np.exp(-event_age / 7.0)
        / np.sqrt(item_age + 1.0)
    )
    new_items = _ranked_items(new_reference.groupby("aid")["new_score"].sum())

    quotas = {
        "click": (36, 18, 6),
        "cart": (12, 24, 24),
        "order": (6, 12, 42),
    }
    result = {}
    for stage, (clicks, carts, orders) in quotas.items():
        result[stage] = _merge_with_budget(
            [
                (action_lists["clicks"], clicks),
                (action_lists["carts"], carts),
                (action_lists["orders"], orders),
                (trend, 20),
                (new_items, 20),
            ],
            fallback,
            100,
        )
    return result


def build_directional_transitions(
    events: pd.DataFrame, max_neighbors: int = 200
) -> dict[int, list[tuple[int, float]]]:
    """统计未来三步、有方向、行为加权且长短期混合衰减的商品转移。"""
    transitions: dict[int, Counter] = defaultdict(Counter)
    ordered = events.sort_values(["session", "ts", "event_order"], kind="mergesort")
    for _, group in ordered.groupby("session", sort=False):
        rows = list(group[["aid", "ts", "type"]].itertuples(index=False, name=None))
        for left_index, (left, left_ts, _) in enumerate(rows):
            for distance, (right, right_ts, right_type) in enumerate(
                rows[left_index + 1:left_index + 4], start=1
            ):
                if right_ts <= left_ts or int(left) == int(right):
                    continue
                gap_minutes = max((int(right_ts) - int(left_ts)) / 60_000, 0.0)
                weight = ACTION_WEIGHTS[str(right_type)] * (
                    math.exp(-gap_minutes / 30.0)
                    + 0.2 * math.exp(-gap_minutes / 1440.0)
                ) / distance
                transitions[int(left)][int(right)] += weight
    return {
        left: counter.most_common(max_neighbors)
        for left, counter in transitions.items()
    }


def build_category_popularity(
    events_enriched: pd.DataFrame, cutoff_ts: int, max_per_category: int = 200
) -> dict[int, list[tuple[int, float]]]:
    """按照事件发生时已知类目，构建行为加权、时间衰减的类目热度。"""
    valid = events_enriched[events_enriched["categoryid"].ne(-1)].copy()
    age_days = (cutoff_ts - valid["ts"].to_numpy()) / DAY_MS
    valid["trend_score"] = valid["type"].map(ACTION_WEIGHTS).astype(float) * (
        np.exp(-age_days / 7.0) + 0.2 * np.exp(-age_days / 30.0)
    )
    counts = (
        valid.groupby(["categoryid", "aid"], as_index=False)["trend_score"]
        .sum()
        .rename(columns={"trend_score": "score"})
        .sort_values(["categoryid", "score", "aid"], ascending=[True, False, True])
    )
    output = {}
    for category, group in counts.groupby("categoryid", sort=False):
        output[int(category)] = [
            (int(row.aid), float(row.score))
            for row in group.head(max_per_category).itertuples(index=False)
        ]
    return output


def latest_category_map(
    category_changes: pd.DataFrame, cutoff_ts: int
) -> dict[int, int]:
    """取得冻结索引时点每个商品最近已知类目。"""
    visible = category_changes[category_changes["timestamp"] < cutoff_ts]
    latest = visible.sort_values(["itemid", "timestamp"]).groupby("itemid").tail(1)
    return dict(zip(latest["itemid"].astype(int), latest["categoryid"].astype(int)))


def build_frozen_indexes(
    events: pd.DataFrame,
    events_enriched: pd.DataFrame,
    category_changes: pd.DataFrame,
    cutoff_ts: int,
    *,
    neighbor_k: int = 200,
) -> dict:
    """只使用验证开始前事件，构建一次冻结召回索引。"""
    reference = events[events["ts"] < cutoff_ts]
    enriched_reference = events_enriched[events_enriched["ts"] < cutoff_ts]
    popularity = {}
    for name, days in (("1d", 1), ("7d", 7), ("30d", 30), ("all", None)):
        window = (
            reference
            if days is None
            else reference[reference["ts"] >= cutoff_ts - days * 86_400_000]
        )
        popularity[name] = window["aid"].value_counts().index.astype(int).tolist()
    return {
        "index_version": 5,
        "cutoff_ts": int(cutoff_ts),
        "reference_events": int(len(reference)),
        "catalog": set(reference["aid"].astype(int)),
        "popularity": popularity,
        "hybrid_popularity": build_hybrid_popularity(reference, cutoff_ts),
        "itemcf": build_itemcf(reference, max_neighbors=neighbor_k),
        "transition": build_directional_transitions(reference, max_neighbors=neighbor_k),
        "category_popularity": build_category_popularity(
            enriched_reference, cutoff_ts, max_per_category=neighbor_k
        ),
        "item_category": latest_category_map(category_changes, cutoff_ts),
    }


def _aggregate_neighbors(
    recent: list[int],
    neighbors: dict[int, list[tuple[int, float]]],
    limit: int,
    *,
    seed_limit: int = 5,
    weight_mode: str = "reciprocal",
) -> list[tuple[int, float]]:
    scores = Counter()
    for recency, aid in enumerate(recent[:seed_limit]):
        weight = (
            1.0 / (recency + 1)
            if weight_mode == "reciprocal"
            else math.exp(-0.1 * recency)
        )
        for neighbor, score in neighbors.get(int(aid), []):
            scores[int(neighbor)] += float(score) * weight
    return scores.most_common(limit)


def recall_from_frozen_indexes(
    samples: pd.DataFrame, indexes: dict, route_limits: dict[str, int] | None = None
) -> pd.DataFrame:
    """对序列样本执行多路召回，并保留来源内分数和排名。"""
    limits = dict(PRODUCTION_ROUTE_LIMITS if route_limits is None else route_limits)
    rows = []
    for sample in samples.itertuples(index=False):
        recent = list(dict.fromkeys(list(sample.history_aids)[::-1]))
        sources: dict[str, list[tuple[int, float]]] = {
            "recent": [
                (int(aid), 1.0 / rank)
                for rank, aid in enumerate(recent[:limits["recent"]], start=1)
            ]
        }
        history_types = set(map(int, sample.history_type_ids))
        stage = "order" if 2 in history_types else ("cart" if 1 in history_types else "click")
        sources["hybrid_popular"] = [
            (int(aid), 1.0 / rank)
            for rank, aid in enumerate(indexes["hybrid_popularity"][stage][:limits["hybrid_popular"]], start=1)
        ]
        sources["itemcf"] = _aggregate_neighbors(
            recent,
            indexes["itemcf"],
            limits["itemcf"],
            seed_limit=10,
            weight_mode="exponential",
        )
        sources["transition"] = _aggregate_neighbors(
            recent, indexes["transition"], limits["transition"]
        )
        category_scores = Counter()
        for recency, aid in enumerate(recent[:5]):
            category = indexes["item_category"].get(int(aid))
            if category is None:
                continue
            for rank, (candidate, score) in enumerate(
                indexes["category_popularity"].get(category, []), start=1
            ):
                category_scores[int(candidate)] += float(score) / (
                    (recency + 1) * rank
                )
        sources["category"] = category_scores.most_common(limits["category"])
        causal_catalog = indexes["catalog"] | set(int(aid) for aid in recent)
        for source, candidates in sources.items():
            for rank, (aid, score) in enumerate(candidates, start=1):
                if int(aid) not in causal_catalog:
                    raise AssertionError("召回结果超出冻结因果目录")
                rows.append(
                    (int(sample.session), int(aid), source, rank, float(score))
                )
    return pd.DataFrame(
        rows,
        columns=["session", "aid", "source", "source_rank", "source_score"],
    )
