"""Summarize the seven production recall routes on the frozen validation split."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCES = ("recent", "category", "itemcf", "transition", "hybrid_popular", "item2vec", "two_tower")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/ranker_datasets_full_union/validation_candidates.parquet")
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--output", default="outputs/recall_route_summary")
    parser.add_argument("--catalog-items", type=int, default=198951)
    args = parser.parse_args()

    columns = ["sample_id", "aid", "target_type", "label"]
    columns += [f"source_rank_{source}" for source in SOURCES]
    columns += [f"source_score_{source}" for source in SOURCES]
    candidates = pd.read_parquet(args.candidates, columns=columns)
    validation = pd.read_parquet(
        Path(args.processed) / "validation_samples.parquet",
        columns=["session", "target_ts", "target_aid", "history_aids"],
    ).sort_values(["target_ts", "session"], kind="mergesort").reset_index(drop=True)
    validation["sample_id"] = validation.index.astype("int64")
    validation["novel"] = [
        int(target) not in set(map(int, history))
        for target, history in zip(validation["target_aid"], validation["history_aids"])
    ]
    total = len(validation)
    novel_ids = set(validation.loc[validation["novel"], "sample_id"].astype(int))
    positives = candidates[candidates["label"].eq(1)].copy()
    presence = pd.DataFrame({
        source: positives[f"source_rank_{source}"].notna() for source in SOURCES
    }, index=positives.index)
    route_count = presence.sum(axis=1)

    reports = []
    hit_sets = {}
    for source in SOURCES:
        rank_column = f"source_rank_{source}"
        score_column = f"source_score_{source}"
        source_rows = candidates[candidates[rank_column].notna()][
            ["sample_id", "aid", "target_type", "label", rank_column, score_column]
        ].copy()
        source_positive = source_rows[source_rows["label"].eq(1)]
        source_positive_50 = source_positive[source_positive[rank_column].le(50)]
        hit_ids = set(source_positive["sample_id"].astype(int))
        hit_ids_50 = set(source_positive_50["sample_id"].astype(int))
        hit_sets[source] = hit_ids
        exclusive = int((presence[source] & route_count.eq(1)).sum())
        target_ranks = source_positive[rank_column].astype(float)

        positive_score = source_positive.set_index("sample_id")[score_column]
        negatives = source_rows[source_rows["label"].eq(0)][["sample_id", score_column]].copy()
        negatives["positive_score"] = negatives["sample_id"].map(positive_score)
        eligible = negatives["positive_score"].notna()
        comparisons = (
            (negatives.loc[eligible, "positive_score"] > negatives.loc[eligible, score_column]).astype(float)
            + 0.5 * (negatives.loc[eligible, "positive_score"] == negatives.loc[eligible, score_column]).astype(float)
        )
        negatives.loc[eligible, "pair_auc"] = comparisons
        session_auc = negatives.loc[eligible].groupby("sample_id")["pair_auc"].mean()
        reports.append({
            "source": source,
            "average_candidates": float(len(source_rows) / total),
            "catalog_coverage": float(source_rows["aid"].nunique() / args.catalog_items),
            "recall_at_20": float(source_positive[rank_column].le(20).sum() / total),
            "recall_at_50": float(len(hit_ids_50) / total),
            "recall_at_100": float(len(hit_ids) / total),
            "novel_recall_at_50": float(len(hit_ids_50 & novel_ids) / len(novel_ids)),
            "novel_recall_at_100": float(len(hit_ids & novel_ids) / len(novel_ids)),
            "hits": int(len(hit_ids)),
            "exclusive_hits": exclusive,
            "exclusive_share_of_source_hits": float(exclusive / len(hit_ids)) if hit_ids else 0.0,
            "union_recall_drop_without_source": float(exclusive / total),
            "conditional_target_rank_median": float(target_ranks.median()) if len(target_ranks) else None,
            "mrr_at_50": float((1.0 / target_ranks).sum() / total),
            "ndcg_at_50": float((1.0 / np.log2(target_ranks + 1.0)).sum() / total),
            "conditional_candidate_auc": float(comparisons.mean()) if len(comparisons) else None,
            "conditional_session_gauc": float(session_auc.mean()) if len(session_auc) else None,
            "auc_eligible_sessions": int(len(session_auc)),
        })

    label_actions = candidates[["sample_id", "target_type"]].drop_duplicates("sample_id")
    action_counts = label_actions["target_type"].value_counts()
    for report in reports:
        source = report["source"]
        source_positive = positives[presence[source] & positives[f"source_rank_{source}"].le(50)]
        for action in ("clicks", "carts", "orders"):
            report[f"{action}_recall_at_50"] = float(
                source_positive["target_type"].eq(action).sum() / action_counts[action]
            )
    union_hits = set().union(*hit_sets.values())
    union_frame = label_actions.copy()
    union_frame["hit"] = union_frame["sample_id"].isin(union_hits)
    union_by_action = union_frame.groupby("target_type")["hit"].mean().to_dict()
    action_weights = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}
    pairwise = {}
    for left, right in combinations(SOURCES, 2):
        union = hit_sets[left] | hit_sets[right]
        pairwise[f"{left}__{right}"] = float(
            len(hit_sets[left] & hit_sets[right]) / len(union) if union else 0.0
        )
    summary = {
        "protocol": {
            "validation_samples": total,
            "route_topk": {
                "recent": 30, "category": 150, "itemcf": 200,
                "transition": 150, "hybrid_popular": 100,
                "item2vec": 250, "two_tower": 300,
            },
            "pre_rank_truncation": False,
            "positive_injection": False,
            "test_evaluated": False,
        },
        "union": {
            "hits": len(union_hits),
            "recall": len(union_hits) / total,
            "novel_recall": len(union_hits & novel_ids) / len(novel_ids),
            "average_candidates_after_deduplication": len(candidates) / total,
            "recall_by_action": {key: float(union_by_action[key]) for key in action_weights},
            "macro_weighted_recall": float(sum(
                action_weights[key] * union_by_action[key] for key in action_weights
            )),
            "event_weighted_recall": float(np.average(
                union_frame["hit"],
                weights=union_frame["target_type"].map(action_weights),
            )),
        },
        "routes": reports,
        "pairwise_target_hit_jaccard": pairwise,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(reports).to_csv(output / "routes.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
