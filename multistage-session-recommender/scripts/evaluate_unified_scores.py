"""Evaluate one unified Top-K list from any candidate score column."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.unified_toplist import evaluate_unified_toplist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--score-column", default="final_score")
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--samples-file", default="validation_samples.parquet")
    args = parser.parse_args()

    columns = ["sample_id", "aid", "target_type", "label", args.score_column]
    scored = pd.read_parquet(args.scores, columns=columns).rename(
        columns={args.score_column: "final_score"}
    )
    validation = pd.read_parquet(
        Path(args.processed) / args.samples_file, columns=["session", "target_ts"]
    ).sort_values(["target_ts", "session"], kind="mergesort").reset_index(drop=True)
    validation["sample_id"] = validation.index.astype("int64")
    sessions = pd.read_parquet(
        Path(args.processed) / "sessions.parquet", columns=["session", "visitorid"]
    ).drop_duplicates("session")
    sample_users = validation[["sample_id", "session"]].merge(
        sessions, on="session", how="left", validate="many_to_one"
    )[["sample_id", "visitorid"]]
    metrics = evaluate_unified_toplist(scored, k=args.topk, sample_users=sample_users)
    metrics.update({
        "scores": args.scores,
        "score_column": args.score_column,
        "test_evaluated": Path(args.samples_file).name == "test_samples.parquet",
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
