"""Train and evaluate Item2Vec V2 checkpoints on one strict temporal OOF fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recall.item2vec_ann import Item2VecANN, train_item2vec_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--fold-report", default="outputs/rolling_oof_candidates_compact/fold_3_report.json")
    parser.add_argument("--fold-candidates", default="outputs/rolling_oof_candidates_compact/fold_3_candidates.parquet")
    parser.add_argument("--output", default="outputs/item2vec_v2_checkpoints/fold_3")
    parser.add_argument("--subsample", type=float, default=1e-5)
    args = parser.parse_args()
    output = ROOT / args.output; output.mkdir(parents=True, exist_ok=True)
    fold = json.loads((ROOT / args.fold_report).read_text())
    cutoff, end = int(fold["cutoff_ts"]), int(fold["end_ts"])
    processed = ROOT / args.processed
    events = pd.read_parquet(processed / "events_all.parquet")
    reference = events[events.ts < cutoff].copy()
    samples = pd.read_parquet(processed / "train_samples.parquet").sort_values(
        ["target_ts", "session"], kind="mergesort"
    ).reset_index(drop=True)
    samples["sample_id"] = samples.index.astype("int64")
    samples = samples[(samples.target_ts >= cutoff) & (samples.target_ts < end)].copy()
    base_sources = ("recent", "category", "itemcf", "transition", "popular_30d", "two_tower")
    columns = ["sample_id", "label"] + [f"source_rank_{s}" for s in base_sources]
    positives = pd.read_parquet(ROOT / args.fold_candidates, columns=columns, filters=[[('label', '=', 1)]]).drop_duplicates("sample_id")
    samples = samples.merge(positives.drop(columns="label"), on="sample_id", how="left")
    base = samples[[f"source_rank_{s}" for s in base_sources]].notna().any(axis=1).to_numpy()
    target_counts = reference["aid"].value_counts()
    frequent_target = samples["target_aid"].map(target_counts).fillna(0).ge(5).to_numpy()
    reports = []
    action_weights = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}

    def evaluate(epoch, embeddings):
        embeddings.save(output / f"item2vec_epoch_{epoch}.npz")
        scoring_samples = samples.copy()
        scoring_samples["session"] = scoring_samples["sample_id"]
        candidates = Item2VecANN(embeddings).recall(scoring_samples, 250)
        targets = set(zip(samples.sample_id.astype(int), samples.target_aid.astype(int)))
        pairs = set(zip(candidates.session.astype(int), candidates.aid.astype(int)))
        hit_ids = {sid for sid, aid in targets if (sid, aid) in pairs}
        hit = samples.sample_id.astype(int).isin(hit_ids).to_numpy()
        union = base | hit
        by_action = {a: float(hit[samples.target_type.eq(a).to_numpy()].mean()) for a in action_weights}
        report = {
            "epoch": epoch, "recall_at_250": float(hit.mean()),
            "frequent_target_recall_at_250": float(hit[frequent_target].mean()),
            "weighted_recall_at_250": float(sum(action_weights[a] * by_action[a] for a in action_weights)),
            "exclusive_hits": int((hit & ~base).sum()),
            "union_lift": float((hit & ~base).mean()),
            "recall_by_action": by_action,
        }
        reports.append(report)
        (output / "metrics.json").write_text(json.dumps({"reports": reports, "test_evaluated": False}, indent=2, ensure_ascii=False))
        print(json.dumps(report, ensure_ascii=False), flush=True)

    train_item2vec_embeddings(
        reference, dimensions=128, window=10, negative_samples=15, epochs=15,
        batch_size=4096, learning_rate=0.01, min_count=5, seed=2026,
        subsample=args.subsample, adaptive_window=True, checkpoint_epochs=(5, 10, 15),
        checkpoint_callback=evaluate,
    )


if __name__ == "__main__":
    main()
