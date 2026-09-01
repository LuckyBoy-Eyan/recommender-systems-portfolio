"""Compare ItemCF seed weighting and Top-50/100 on one temporal OOF fold."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_SOURCES = ("recent", "category", "transition", "popular_30d", "item2vec", "two_tower")


def ranked(history: list[int], index: dict, seeds: int, mode: str) -> list[int]:
    scores = Counter()
    recent = list(dict.fromkeys(map(int, history[::-1])))[:seeds]
    for position, aid in enumerate(recent, 1):
        weight = 1.0 / position if mode == "reciprocal" else math.exp(-0.1 * (position - 1))
        for candidate, similarity in index.get(aid, []):
            scores[int(candidate)] += float(similarity) * weight
    return [aid for aid, _ in scores.most_common(100)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--fold-candidates", required=True)
    parser.add_argument("--fold-report", required=True)
    parser.add_argument("--heuristic-index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.processed)
    fold = json.loads(Path(args.fold_report).read_text())
    cutoff, end = int(fold["cutoff_ts"]), int(fold["end_ts"])
    samples = pd.read_parquet(root / "train_samples.parquet").sort_values(
        ["target_ts", "session"], kind="mergesort"
    ).reset_index(drop=True)
    samples["sample_id"] = samples.index.astype("int64")
    samples = samples[(samples.target_ts >= cutoff) & (samples.target_ts < end)].copy()
    samples["novel"] = [int(t) not in set(map(int, h)) for t, h in zip(samples.target_aid, samples.history_aids)]

    columns = ["sample_id", "aid", "label", "source_rank_itemcf"]
    columns += [f"source_rank_{source}" for source in BASE_SOURCES]
    positives = pd.read_parquet(args.fold_candidates, columns=columns, filters=[[('label', '=', 1)]]).drop_duplicates("sample_id")
    samples = samples.merge(positives.drop(columns="label"), on="sample_id", how="left", validate="one_to_one")
    base = samples[[f"source_rank_{source}" for source in BASE_SOURCES]].notna().any(axis=1).to_numpy()
    index = joblib.load(args.heuristic_index)["itemcf"]
    targets = samples.target_aid.astype(int).to_numpy()
    rankings = {
        "seed5_reciprocal": [ranked(h, index, 5, "reciprocal") for h in samples.history_aids],
        "seed10_exp01": [ranked(h, index, 10, "exponential") for h in samples.history_aids],
    }
    weights = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}
    base_macro = sum(weights[a] * base[samples.target_type.eq(a).to_numpy()].mean() for a in weights)
    reports = []
    for variant, lists in rankings.items():
        for topk in (50, 100):
            hit = np.fromiter((target in items[:topk] for target, items in zip(targets, lists)), bool, len(samples))
            union = base | hit
            macro = sum(weights[a] * union[samples.target_type.eq(a).to_numpy()].mean() for a in weights)
            reports.append({
                "variant": variant, "topk": topk, "recall": float(hit.mean()),
                "novel_recall": float(hit[samples.novel.to_numpy()].mean()),
                "exclusive_hits_vs_other_six": int((hit & ~base).sum()),
                "union_lift_vs_other_six": float((hit & ~base).mean()),
                "union_macro_weighted_lift": float(macro - base_macro),
            })
    current50 = next(r for r in reports if r["variant"] == "seed5_reciprocal" and r["topk"] == 50)
    stored = samples.source_rank_itemcf.notna().to_numpy()
    current_lists = rankings["seed5_reciprocal"]
    reconstructed = np.fromiter((target in items[:50] for target, items in zip(targets, current_lists)), bool, len(samples))
    result = {"protocol": {"fold": int(fold["fold_id"]), "samples": len(samples), "formal_validation_evaluated": False, "test_evaluated": False}, "current_reconstruction_mismatches": int(np.sum(stored != reconstructed)), "reports": reports}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pd.DataFrame(reports).to_csv(output / "metrics.csv", index=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
