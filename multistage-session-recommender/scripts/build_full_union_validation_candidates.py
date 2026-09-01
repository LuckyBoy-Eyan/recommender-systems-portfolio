"""Build the complete, label-blind validation recall union without pre-ranking truncation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SOURCES = ("recent", "category", "itemcf", "transition", "hybrid_popular", "item2vec", "two_tower")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument(
        "--baseline", default="outputs/full_catalog_recall_validation_item2vec/recall_candidates.parquet"
    )
    parser.add_argument(
        "--two-tower",
        default="outputs/windows_early_stopping_import/windows_two_tower_validation/two_tower_candidates.parquet",
    )
    parser.add_argument("--output", default="outputs/ranker_datasets_full_union/validation_candidates.parquet")
    args = parser.parse_args()

    validation = pd.read_parquet(Path(args.processed) / "validation_samples.parquet").sort_values(
        ["target_ts", "session"], kind="mergesort"
    ).reset_index(drop=True)
    validation["sample_id"] = validation.index.astype("int64")
    labels = validation.set_index("session")

    baseline = pd.read_parquet(
        args.baseline, columns=["session", "aid", "source", "source_rank", "source_score"]
    )
    baseline = baseline[baseline["source"].isin(SOURCES)]
    tower = pd.read_parquet(
        args.two_tower, columns=["session", "aid", "source", "source_rank", "source_score"]
    )
    raw = pd.concat([baseline, tower], ignore_index=True)
    raw["source"] = raw["source"].astype("category")
    raw = raw.groupby(["session", "aid", "source"], observed=True, as_index=False).agg(
        source_rank=("source_rank", "min"), source_score=("source_score", "max")
    )
    raw["rrf_part"] = 1.0 / (60.0 + raw["source_rank"].astype(float))

    summary = raw.groupby(["session", "aid"], as_index=False).agg(
        rrf_score=("rrf_part", "sum"),
        source_count=("source", "nunique"),
        best_source_rank=("source_rank", "min"),
    )
    ranks = raw.pivot(index=["session", "aid"], columns="source", values="source_rank")
    ranks.columns = [f"source_rank_{column}" for column in ranks.columns]
    scores = raw.pivot(index=["session", "aid"], columns="source", values="source_score")
    scores.columns = [f"source_score_{column}" for column in scores.columns]
    candidates = summary.merge(ranks.reset_index(), on=["session", "aid"], how="left").merge(
        scores.reset_index(), on=["session", "aid"], how="left"
    )
    for source in SOURCES:
        for prefix in ("source_rank_", "source_score_"):
            column = f"{prefix}{source}"
            if column not in candidates:
                candidates[column] = pd.NA

    candidates["sample_id"] = candidates["session"].map(labels["sample_id"]).astype("int64")
    candidates["target_type"] = candidates["session"].map(labels["target_type"])
    candidates["target_ts"] = candidates["session"].map(labels["target_ts"]).astype("int64")
    target = candidates["session"].map(labels["target_aid"]).astype("int64")
    candidates["label"] = candidates["aid"].astype("int64").eq(target).astype("int8")
    for action in ("clicks", "carts", "orders"):
        candidates[f"label_{action}"] = (
            candidates["label"].eq(1) & candidates["target_type"].eq(action)
        ).astype("int8")
    candidates = candidates.sort_values(
        ["sample_id", "rrf_score", "source_count", "aid"],
        ascending=[True, False, False, True], kind="mergesort",
    ).reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(output, index=False, compression="zstd")
    counts = candidates.groupby("sample_id").size()
    report = {
        "samples": int(len(validation)),
        "rows": int(len(candidates)),
        "candidate_recall": float(candidates["label"].sum() / len(validation)),
        "positive_sessions": int(candidates.loc[candidates["label"].eq(1), "sample_id"].nunique()),
        "candidates_per_sample": {
            "mean": float(counts.mean()), "p50": float(counts.quantile(0.5)),
            "p90": float(counts.quantile(0.9)), "p99": float(counts.quantile(0.99)),
            "max": int(counts.max()),
        },
        "sources": list(SOURCES),
        "pre_rank_truncation": False,
        "positive_injection": False,
        "test_evaluated": False,
    }
    (output.parent / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
