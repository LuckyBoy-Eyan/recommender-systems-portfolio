"""Compare causal directional-transition variants on one temporal OOF fold."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ACTION_WEIGHTS = {"clicks": 1.0, "carts": 3.0, "orders": 6.0}
BASE_SOURCES = ("recent", "category", "itemcf", "popular_30d", "item2vec", "two_tower")


def build_variants(events: pd.DataFrame, cutoff: int, neighbors: int = 200) -> tuple[dict, dict]:
    adjacent, multistep = defaultdict(Counter), defaultdict(Counter)
    ordered = events[events["ts"] < cutoff].sort_values(["session", "ts", "event_order"], kind="mergesort")
    for _, group in ordered.groupby("session", sort=False):
        rows = list(group[["aid", "ts", "type"]].itertuples(index=False, name=None))
        for left_index, (left, left_ts, _) in enumerate(rows):
            for distance, (right, right_ts, right_type) in enumerate(rows[left_index + 1:left_index + 4], 1):
                if right_ts <= left_ts or int(left) == int(right):
                    continue
                gap = (int(right_ts) - int(left_ts)) / 60_000
                score = ACTION_WEIGHTS[str(right_type)] * (
                    math.exp(-gap / 30.0) + 0.2 * math.exp(-gap / 1440.0)
                ) / distance
                multistep[int(left)][int(right)] += score
                if distance == 1:
                    adjacent[int(left)][int(right)] += score
    compact = lambda index: {aid: scores.most_common(neighbors) for aid, scores in index.items()}
    return compact(adjacent), compact(multistep)


def recall(history: list[int], index: dict[int, list[tuple[int, float]]]) -> set[int]:
    scores = Counter()
    recent = list(dict.fromkeys(map(int, history[::-1])))[:5]
    for recency, aid in enumerate(recent, 1):
        for candidate, score in index.get(aid, []):
            scores[int(candidate)] += float(score) / recency
    return {aid for aid, _ in scores.most_common(50)}


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
    samples = samples[(samples["target_ts"] >= cutoff) & (samples["target_ts"] < end)].copy()
    samples["novel"] = [int(t) not in set(map(int, h)) for t, h in zip(samples.target_aid, samples.history_aids)]

    columns = ["sample_id", "aid", "label", "source_rank_transition"]
    columns += [f"source_rank_{source}" for source in BASE_SOURCES]
    positives = pd.read_parquet(args.fold_candidates, columns=columns, filters=[[('label', '=', 1)]]).drop_duplicates("sample_id")
    samples = samples.merge(positives.drop(columns="label"), on="sample_id", how="left", validate="one_to_one")
    base = samples[[f"source_rank_{source}" for source in BASE_SOURCES]].notna().any(axis=1).to_numpy()

    current = joblib.load(args.heuristic_index)["transition"]
    events = pd.read_parquet(root / "events_all.parquet", columns=["session", "aid", "ts", "type", "event_order"])
    adjacent, multistep = build_variants(events, cutoff)
    variants = {"current": current, "adjacent_mixed_decay": adjacent, "multistep_mixed_decay": multistep}
    weights = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}
    base_macro = sum(weights[a] * base[samples.target_type.eq(a).to_numpy()].mean() for a in weights)
    reports = []
    stored = samples["source_rank_transition"].notna().to_numpy()
    mismatch = None
    for name, index in variants.items():
        pools = [recall(history, index) for history in samples["history_aids"]]
        hit = np.fromiter((int(target) in pool for target, pool in zip(samples.target_aid, pools)), bool, len(samples))
        if name == "current":
            mismatch = int(np.sum(hit != stored))
        union = base | hit
        macro = sum(weights[a] * union[samples.target_type.eq(a).to_numpy()].mean() for a in weights)
        reports.append({
            "variant": name, "recall_at_50": float(hit.mean()),
            "novel_recall_at_50": float(hit[samples.novel.to_numpy()].mean()),
            "exclusive_hits_vs_base_six": int((hit & ~base).sum()),
            "union_lift_vs_base_six": float((hit & ~base).mean()),
            "union_macro_weighted_lift": float(macro - base_macro),
        })
    result = {"protocol": {"fold": int(fold["fold_id"]), "samples": len(samples), "formal_validation_evaluated": False, "test_evaluated": False}, "current_reconstruction_mismatches": mismatch, "variants": reports}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pd.DataFrame(reports).to_csv(output / "variants.csv", index=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
