"""Compare causal popularity variants on the last temporal OOF fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DAY_MS = 86_400_000
ACTION_WEIGHTS = {"clicks": 1.0, "carts": 3.0, "orders": 6.0}
BASE_SOURCES = ("recent", "category", "itemcf", "transition", "item2vec", "two_tower")


def top_items(scores: pd.Series, limit: int = 200) -> list[int]:
    return scores.sort_values(ascending=False, kind="mergesort").head(limit).index.astype(int).tolist()


def merge_with_budget(parts: list[tuple[list[int], int]], fallback: list[int], budget: int) -> tuple[int, ...]:
    output = []
    seen = set()
    for items, quota in parts:
        added = 0
        for aid in items:
            if aid in seen:
                continue
            output.append(aid); seen.add(aid); added += 1
            if added >= quota or len(output) >= budget:
                break
    for aid in fallback:
        if len(output) >= budget:
            break
        if aid not in seen:
            output.append(aid); seen.add(aid)
    return tuple(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--fold-candidates", default="outputs/rolling_oof_candidates_compact/fold_3_candidates.parquet")
    parser.add_argument("--fold-report", default="outputs/rolling_oof_candidates_compact/fold_3_report.json")
    parser.add_argument("--output", default="outputs/popular_variant_comparison")
    args = parser.parse_args()

    fold_report = json.loads(Path(args.fold_report).read_text())
    cutoff, end = int(fold_report["cutoff_ts"]), int(fold_report["end_ts"])
    samples = pd.read_parquet(Path(args.processed) / "train_samples.parquet").sort_values(
        ["target_ts", "session"], kind="mergesort"
    ).reset_index(drop=True)
    samples["sample_id"] = samples.index.astype("int64")
    samples = samples[(samples["target_ts"] >= cutoff) & (samples["target_ts"] < end)].copy()
    samples["novel"] = [
        int(target) not in set(map(int, history))
        for target, history in zip(samples["target_aid"], samples["history_aids"])
    ]
    samples["stage"] = [
        "order" if 2 in history else ("cart" if 1 in history else "click")
        for history in samples["history_type_ids"]
    ]

    positive_columns = ["sample_id", "aid", "label", "source_rank_popular_30d"]
    positive_columns += [f"source_rank_{source}" for source in BASE_SOURCES]
    positives = pd.read_parquet(
        args.fold_candidates, columns=positive_columns, filters=[[('label', '=', 1)]]
    ).drop_duplicates("sample_id")
    samples = samples.merge(
        positives.drop(columns="label"), on="sample_id", how="left", validate="one_to_one",
        suffixes=("", "_candidate"),
    )
    if not samples["aid"].eq(samples["target_aid"]).all():
        raise AssertionError("Fold正样本与训练样本目标不一致")
    base_columns = [f"source_rank_{source}" for source in BASE_SOURCES]
    samples["base_hit"] = samples[base_columns].notna().any(axis=1)

    events = pd.read_parquet(
        Path(args.processed) / "events_all.parquet", columns=["aid", "ts", "type"]
    )
    reference = events[events["ts"] < cutoff].copy().reset_index(drop=True)
    reference["action_weight"] = reference["type"].map(ACTION_WEIGHTS).astype(float)
    weighted_global = top_items(reference.groupby("aid")["action_weight"].sum())
    action_lists = {
        action: top_items(reference.loc[reference["type"].eq(action), "aid"].value_counts())
        for action in ACTION_WEIGHTS
    }
    current_30d = top_items(
        reference.loc[reference["ts"] >= cutoff - 30 * DAY_MS, "aid"].value_counts(), 50
    )

    age_days = np.maximum((cutoff - reference["ts"].to_numpy()) / DAY_MS, 0.0)
    reference["trend_score"] = reference["action_weight"] * (
        np.exp(-age_days / 1.0) + 0.3 * np.exp(-age_days / 7.0)
    )
    trend = top_items(reference.groupby("aid")["trend_score"].sum(), 200)
    first_seen = reference.groupby("aid")["ts"].min()
    recent_items = set(first_seen[first_seen >= cutoff - 30 * DAY_MS].index.astype(int))
    new_reference = reference[reference["aid"].isin(recent_items)].copy()
    item_age = (cutoff - new_reference["aid"].map(first_seen)) / DAY_MS
    new_reference["new_score"] = new_reference["action_weight"] * np.exp(-age_days[new_reference.index] / 7.0) / np.sqrt(item_age + 1.0)
    new_items = top_items(new_reference.groupby("aid")["new_score"].sum(), 200)

    action_quota = {
        "click": (30, 15, 5), "cart": (10, 20, 20), "order": (5, 10, 35),
    }
    hybrid_quota = {
        "click": (18, 9, 3), "cart": (6, 12, 12), "order": (3, 6, 21),
    }
    action_candidates = {}
    hybrid_candidates = {}
    for stage in action_quota:
        cq, aq, oq = action_quota[stage]
        action_candidates[stage] = merge_with_budget(
            [(action_lists["clicks"], cq), (action_lists["carts"], aq), (action_lists["orders"], oq)],
            weighted_global, 50,
        )
        cq, aq, oq = hybrid_quota[stage]
        hybrid_candidates[stage] = merge_with_budget(
            [
                (action_lists["clicks"], cq), (action_lists["carts"], aq),
                (action_lists["orders"], oq), (trend, 10), (new_items, 10),
            ], weighted_global, 50,
        )

    variants: dict[str, list[set[int]]] = {
        "popular_30d": [set(current_30d)] * len(samples),
        "action_popular": [set(action_candidates[stage]) for stage in samples["stage"]],
        "trend_popular": [set(trend[:50])] * len(samples),
        "new_item_popular": [set(new_items[:50])] * len(samples),
        "hybrid_popular": [set(hybrid_candidates[stage]) for stage in samples["stage"]],
    }
    variants["trend_plus_hybrid"] = [
        trend_pool | hybrid_pool
        for trend_pool, hybrid_pool in zip(variants["trend_popular"], variants["hybrid_popular"])
    ]
    variants["popular_30d_plus_hybrid"] = [
        old_pool | hybrid_pool
        for old_pool, hybrid_pool in zip(variants["popular_30d"], variants["hybrid_popular"])
    ]
    reports = []
    metric_weights = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}
    base = samples["base_hit"].to_numpy(bool)
    base_by_action = {
        action: float(base[samples["target_type"].eq(action).to_numpy()].mean())
        for action in metric_weights
    }
    base_macro = sum(metric_weights[action] * base_by_action[action] for action in metric_weights)
    for name, candidate_sets in variants.items():
        hit = np.fromiter(
            (int(target) in pool for target, pool in zip(samples["target_aid"], candidate_sets)),
            dtype=bool, count=len(samples),
        )
        exclusive = hit & ~base
        route_by_action = {
            action: float(hit[samples["target_type"].eq(action).to_numpy()].mean())
            for action in metric_weights
        }
        union = base | hit
        union_by_action = {
            action: float(union[samples["target_type"].eq(action).to_numpy()].mean())
            for action in metric_weights
        }
        report = {
            "variant": name,
            "average_candidates": float(np.mean([len(pool) for pool in candidate_sets])),
            "recall_at_50": float(hit.mean()),
            "novel_recall_at_50": float(hit[samples["novel"].to_numpy()].mean()),
            **{f"{action}_recall_at_50": route_by_action[action] for action in metric_weights},
            "macro_weighted_recall_at_50": float(sum(
                metric_weights[action] * route_by_action[action] for action in metric_weights
            )),
            "exclusive_hits_vs_base_six": int(exclusive.sum()),
            "union_recall_with_base_six": float(union.mean()),
            "union_lift_vs_base_six": float(exclusive.mean()),
            "union_macro_weighted_recall": float(sum(
                metric_weights[action] * union_by_action[action] for action in metric_weights
            )),
            "union_macro_weighted_lift": float(
                sum(metric_weights[action] * union_by_action[action] for action in metric_weights)
                - base_macro
            ),
            "union_event_weighted_recall": float(np.average(
                union, weights=samples["target_type"].map(metric_weights)
            )),
        }
        reports.append(report)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": {
            "fold": int(fold_report["fold_id"]), "cutoff_ts": cutoff, "end_ts": end,
            "samples": len(samples), "candidate_budget": 50,
            "test_evaluated": False, "formal_validation_evaluated": False,
        },
        "base_six_union_recall": float(samples["base_hit"].mean()),
        "base_six_recall_by_action": base_by_action,
        "base_six_macro_weighted_recall": base_macro,
        "variants": reports,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(reports).to_csv(output / "variants.csv", index=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
