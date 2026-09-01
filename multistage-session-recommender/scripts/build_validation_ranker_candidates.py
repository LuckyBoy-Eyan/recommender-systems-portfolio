"""Compact frozen validation candidates to the same schema as rolling OOF train data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_rolling_oof_candidates import compact_candidates


ALLOWED_BASELINE_SOURCES = {
    "recent", "category", "itemcf", "transition", "hybrid_popular", "item2vec"
}


def main() -> None:
    parser = argparse.ArgumentParser(description="构建与 OOF 同口径的验证精排候选")
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument(
        "--baseline",
        default="outputs/full_catalog_recall_validation_item2vec/recall_candidates.parquet",
    )
    parser.add_argument(
        "--two-tower",
        default="outputs/windows_early_stopping_import/windows_two_tower_validation/two_tower_candidates.parquet",
    )
    parser.add_argument("--output", default="outputs/ranker_datasets/validation_candidates.parquet")
    parser.add_argument(
        "--max-negatives", type=int, default=0,
        help="0表示验证/推理保留全部召回候选；正数仅用于显式消融",
    )
    args = parser.parse_args()

    validation = pd.read_parquet(ROOT / args.processed / "validation_samples.parquet")
    validation = validation.sort_values(["target_ts", "session"], kind="mergesort").reset_index(drop=True)
    validation["sample_id"] = np.arange(len(validation), dtype=np.int64)
    session_to_sample = dict(zip(validation["session"].astype(int), validation["sample_id"].astype(int)))
    original_session = dict(zip(validation["sample_id"].astype(int), validation["session"].astype(int)))

    baseline = pd.read_parquet(ROOT / args.baseline)
    baseline = baseline[baseline["source"].isin(ALLOWED_BASELINE_SOURCES)]
    tower = pd.read_parquet(ROOT / args.two_tower)
    raw = pd.concat([baseline, tower], ignore_index=True)
    raw["session"] = raw["session"].map(session_to_sample)
    if raw["session"].isna().any():
        raise ValueError("验证候选包含未知 Session")
    raw["session"] = raw["session"].astype(int)
    samples = validation.rename(columns={"session": "original_session", "sample_id": "session"})
    compact, recalled_hits = compact_candidates(
        raw, samples, max_negatives=args.max_negatives, topk=50,
        inject_missing_positives=False,
        prioritize_positives=False,
    )
    compact = compact.rename(columns={"session": "sample_id"})
    compact["session"] = compact["sample_id"].map(original_session).astype(int)
    compact["target_ts"] = compact["sample_id"].map(
        dict(zip(validation["sample_id"].astype(int), validation["target_ts"].astype(int)))
    )
    compact["label_clicks"] = compact["label"]
    compact["label_carts"] = compact["label"] * compact["target_type"].isin(["carts", "orders"]).astype(int)
    compact["label_orders"] = compact["label"] * compact["target_type"].eq("orders").astype(int)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    compact.to_parquet(output, index=False)
    print(
        {
            "samples": int(compact["sample_id"].nunique()),
            "rows": int(len(compact)),
            "recall_before_positive_injection": recalled_hits / len(validation),
            "max_rows_per_sample": int(compact.groupby("sample_id").size().max()),
        }
    )


if __name__ == "__main__":
    main()
