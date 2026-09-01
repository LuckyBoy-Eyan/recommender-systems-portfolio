"""Compare causal category recall variants on one temporal OOF fold."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


DAY_MS = 86_400_000
ACTION_WEIGHTS = {"clicks": 1.0, "carts": 3.0, "orders": 6.0}
BASE_SOURCES = ("recent", "itemcf", "transition", "popular_30d", "item2vec", "two_tower")


def ranked_pools(frame: pd.DataFrame, key: str, score: str, limit: int = 200) -> dict[int, list[tuple[int, float]]]:
    grouped = frame.groupby([key, "aid"], as_index=False)[score].sum().sort_values(
        [key, score, "aid"], ascending=[True, False, True], kind="mergesort"
    )
    return {
        int(value): [(int(row.aid), float(getattr(row, score))) for row in group.head(limit).itertuples(index=False)]
        for value, group in grouped.groupby(key, sort=False)
    }


def aggregate(history: list[int], item_key: dict[int, int], pools: dict[int, list[tuple[int, float]]]) -> list[int]:
    scores = Counter()
    recent = list(dict.fromkeys(map(int, history[::-1])))[:5]
    for recency, aid in enumerate(recent, 1):
        key = item_key.get(aid)
        if key is None:
            continue
        for rank, (candidate, score) in enumerate(pools.get(key, []), 1):
            scores[candidate] += score / (recency * rank)
    return [aid for aid, _ in scores.most_common(50)]


def merge(parts: list[tuple[list[int], int]], budget: int = 50) -> set[int]:
    output, seen = [], set()
    for items, quota in parts:
        added = 0
        for aid in items:
            if aid in seen:
                continue
            output.append(aid); seen.add(aid); added += 1
            if added >= quota or len(output) >= budget:
                break
    return set(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--fold-candidates", required=True)
    parser.add_argument("--fold-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.processed)
    fold = json.loads(Path(args.fold_report).read_text())
    cutoff, end = int(fold["cutoff_ts"]), int(fold["end_ts"])

    samples = pd.read_parquet(root / "train_samples.parquet").sort_values(
        ["target_ts", "session"], kind="mergesort"
    ).reset_index(drop=True)
    samples["sample_id"] = samples.index.astype("int64")
    samples = samples[(samples["target_ts"] >= cutoff) & (samples["target_ts"] < end)].copy()
    samples["novel"] = [
        int(target) not in set(map(int, history))
        for target, history in zip(samples["target_aid"], samples["history_aids"])
    ]

    columns = ["sample_id", "aid", "label", "source_rank_category"]
    columns += [f"source_rank_{source}" for source in BASE_SOURCES]
    positives = pd.read_parquet(args.fold_candidates, columns=columns, filters=[[('label', '=', 1)]]).drop_duplicates("sample_id")
    samples = samples.merge(positives.drop(columns="label"), on="sample_id", how="left", validate="one_to_one")
    if not samples["aid"].eq(samples["target_aid"]).all():
        raise AssertionError("OOF positive rows do not match targets")
    base = samples[[f"source_rank_{source}" for source in BASE_SOURCES]].notna().any(axis=1).to_numpy()

    events = pd.read_parquet(root / "events_enriched.parquet")
    reference = events[events["ts"] < cutoff].copy()
    reference = reference[reference["categoryid"].ne(-1)].reset_index(drop=True)
    paths = pd.read_parquet(root / "category_paths.parquet")
    path_map = {int(r.categoryid): tuple(map(int, r.category_path)) for r in paths.itertuples(index=False) if bool(r.in_tree)}
    reference["parent_categoryid"] = reference["categoryid"].map(
        lambda value: path_map.get(int(value), (int(value),))[-2] if len(path_map.get(int(value), ())) >= 2 else -1
    )
    reference["root_key"] = reference["root_categoryid"].astype(int)
    reference["count_score"] = 1.0
    age = (cutoff - reference["ts"].to_numpy()) / DAY_MS
    reference["trend_score"] = reference["type"].map(ACTION_WEIGHTS).astype(float) * (
        np.exp(-age / 7.0) + 0.2 * np.exp(-age / 30.0)
    )
    count_exact = ranked_pools(reference, "categoryid", "count_score")
    trend_exact = ranked_pools(reference, "categoryid", "trend_score")
    count_parent = ranked_pools(reference[reference["parent_categoryid"].ne(-1)], "parent_categoryid", "count_score")
    trend_parent = ranked_pools(reference[reference["parent_categoryid"].ne(-1)], "parent_categoryid", "trend_score")
    trend_root = ranked_pools(reference, "root_key", "trend_score")

    changes = pd.read_parquet(root / "item_category_changes.parquet")
    latest = changes[changes["timestamp"] < cutoff].sort_values(
        ["itemid", "timestamp"]
    ).groupby("itemid").tail(1)
    item_category = dict(zip(latest["itemid"].astype(int), latest["categoryid"].astype(int)))
    item_parent = {
        aid: (path_map.get(category, ())[-2] if len(path_map.get(category, ())) >= 2 else -1)
        for aid, category in item_category.items()
    }
    root_map = dict(zip(paths["categoryid"].astype(int), paths["root_categoryid"].astype(int)))
    item_root = {aid: root_map.get(category, -1) for aid, category in item_category.items()}

    candidate_sets = {name: [] for name in (
        "current_exact", "trend_exact", "hierarchy_count", "hierarchy_trend",
        "trend_exact45_parent5", "trend_exact40_parent10",
    )}
    for history in samples["history_aids"]:
        exact_count = aggregate(history, item_category, count_exact)
        exact_trend = aggregate(history, item_category, trend_exact)
        parent_count = aggregate(history, item_parent, count_parent)
        parent_trend = aggregate(history, item_parent, trend_parent)
        root_trend = aggregate(history, item_root, trend_root)
        candidate_sets["current_exact"].append(set(exact_count))
        candidate_sets["trend_exact"].append(set(exact_trend))
        candidate_sets["hierarchy_count"].append(merge([(exact_count, 35), (parent_count, 15)]))
        candidate_sets["hierarchy_trend"].append(merge([(exact_trend, 30), (parent_trend, 15), (root_trend, 5)]))
        candidate_sets["trend_exact45_parent5"].append(merge([(exact_trend, 45), (parent_trend, 5)]))
        candidate_sets["trend_exact40_parent10"].append(merge([(exact_trend, 40), (parent_trend, 10)]))

    weights = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}
    base_macro = sum(weights[a] * base[samples["target_type"].eq(a).to_numpy()].mean() for a in weights)
    reports = []
    targets = samples["target_aid"].astype(int)
    for name, pools in candidate_sets.items():
        hit = np.fromiter((target in pool for target, pool in zip(targets, pools)), bool, len(samples))
        union = base | hit
        union_macro = sum(weights[a] * union[samples["target_type"].eq(a).to_numpy()].mean() for a in weights)
        reports.append({
            "variant": name,
            "average_candidates": float(np.mean([len(pool) for pool in pools])),
            "recall_at_50": float(hit.mean()),
            "novel_recall_at_50": float(hit[samples["novel"].to_numpy()].mean()),
            "exclusive_hits_vs_base_six": int((hit & ~base).sum()),
            "union_lift_vs_base_six": float((hit & ~base).mean()),
            "union_macro_weighted_lift": float(union_macro - base_macro),
        })
    stored = samples["source_rank_category"].notna().to_numpy()
    sanity = int(np.sum(stored != np.fromiter((target in pool for target, pool in zip(targets, candidate_sets["current_exact"])), bool, len(samples))))
    result = {"protocol": {"fold": int(fold["fold_id"]), "samples": len(samples), "formal_validation_evaluated": False, "test_evaluated": False}, "current_reconstruction_mismatches": sanity, "variants": reports}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pd.DataFrame(reports).to_csv(output / "variants.csv", index=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
