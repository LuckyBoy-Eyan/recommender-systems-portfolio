"""Evaluate the pre-ranker RRF order on the untouched validation candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


UTILITY = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="outputs/ranker_datasets/features/validation.parquet")
    parser.add_argument("--output", default="outputs/windows_rrf_baseline_validation/metrics.json")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    frame = pd.read_parquet(
        args.features,
        columns=["sample_id", "aid", "target_type", "label", "rrf_score"],
    ).sort_values(["sample_id", "rrf_score", "aid"], ascending=[True, False, True])
    samples = frame[["sample_id", "target_type"]].drop_duplicates("sample_id")
    frame["rank"] = frame.groupby("sample_id", sort=False).cumcount() + 1
    positives = frame.loc[frame["label"].eq(1), ["sample_id", "rank", "rrf_score"]]
    positive_rank = dict(zip(positives["sample_id"], positives["rank"]))
    ranks = samples["sample_id"].map(positive_rank).fillna(np.inf).to_numpy()
    hits = ranks <= args.topk

    result = {
        "ranking": "rrf_score",
        "topk": args.topk,
        "recall_at_20": float(hits.mean()),
        "mrr_at_20": float(np.where(hits, 1.0 / ranks, 0.0).mean()),
        "ndcg_at_20": float(np.where(hits, 1.0 / np.log2(ranks + 1.0), 0.0).mean()),
    }
    for action in UTILITY:
        mask = samples["target_type"].eq(action).to_numpy()
        result[f"{action}_recall_at_20"] = float(hits[mask].mean())
    result["weighted_recall_at_20"] = float(sum(
        weight * result[f"{action}_recall_at_20"]
        for action, weight in UTILITY.items()
    ))

    positive_score = dict(zip(positives["sample_id"], positives["rrf_score"]))
    negatives = frame.loc[frame["label"].eq(0), ["sample_id", "rrf_score"]].copy()
    negatives["positive_score"] = negatives["sample_id"].map(positive_score)
    eligible = negatives["positive_score"].notna()
    comparisons = (
        (negatives.loc[eligible, "positive_score"] > negatives.loc[eligible, "rrf_score"]).astype(float)
        + 0.5 * (negatives.loc[eligible, "positive_score"] == negatives.loc[eligible, "rrf_score"]).astype(float)
    )
    negatives.loc[eligible, "pair_auc"] = comparisons
    session_auc = negatives.loc[eligible].groupby("sample_id")["pair_auc"].mean()
    result["candidate_auc"] = float(comparisons.mean())
    result["session_gauc"] = float(session_auc.mean())
    result["auc_eligible_sessions"] = int(len(session_auc))
    result["test_evaluated"] = False

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
