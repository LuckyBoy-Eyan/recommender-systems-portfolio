"""运行冻结索引的全量目录多路召回开发验证。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import candidate_recall, candidate_recall_by_action, source_recall
from src.evaluation.recall_diagnostics import conditional_auc_gauc, route_diagnostics
from src.recall.full_catalog import build_category_popularity, build_directional_transitions, build_frozen_indexes, build_hybrid_popularity, recall_from_frozen_indexes
from src.recall.item2vec_ann import Item2VecANN, Item2VecEmbeddings, train_item2vec_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="运行全量目录冻结索引召回")
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--output", default="outputs/full_catalog_recall_dev")
    parser.add_argument("--validation-limit", type=int, default=3000)
    parser.add_argument("--per-source", type=int, default=50)
    parser.add_argument("--item2vec-topk", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--with-item2vec", action="store_true")
    parser.add_argument("--item2vec-dimensions", type=int, default=128)
    parser.add_argument("--item2vec-window", type=int, default=10)
    parser.add_argument("--item2vec-negative-samples", type=int, default=15)
    parser.add_argument("--item2vec-epochs", type=int, default=10)
    parser.add_argument("--item2vec-batch-size", type=int, default=4096)
    parser.add_argument("--item2vec-min-count", type=int, default=5)
    parser.add_argument("--item2vec-subsample", type=float, default=3e-5)
    args = parser.parse_args()
    started = time.perf_counter()
    processed = ROOT / args.processed
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(processed / "events_all.parquet")
    enriched = pd.read_parquet(processed / "events_enriched.parquet")
    category = pd.read_parquet(processed / "item_category_changes.parquet")
    labels = pd.read_parquet(processed / "labels.parquet")
    validation = pd.read_parquet(processed / "validation_samples.parquet")
    validation_labels = labels[labels["split"].eq("validation")]
    if args.validation_limit and len(validation) > args.validation_limit:
        selected = validation.sample(args.validation_limit, random_state=args.seed)
        validation = selected.sort_values(["target_ts", "session"]).reset_index(drop=True)
        validation_labels = validation_labels[
            validation_labels["session"].isin(set(validation["session"]))
        ]
    cutoff_ts = int(labels.loc[labels["split"].eq("validation"), "target_ts"].min())
    cache_path = processed / "frozen_recall_indexes.joblib"
    if cache_path.exists():
        indexes = joblib.load(cache_path)
        if indexes["cutoff_ts"] != cutoff_ts:
            raise ValueError("冻结索引缓存与当前验证边界不一致")
        if indexes.get("index_version", 0) < 2:
            indexes = build_frozen_indexes(events, enriched, category, cutoff_ts)
            joblib.dump(indexes, cache_path, compress=3)
        elif indexes.get("index_version", 0) < 3:
            indexes["category_popularity"] = build_category_popularity(
                enriched[enriched["ts"] < cutoff_ts], cutoff_ts
            )
            indexes["index_version"] = 3
            joblib.dump(indexes, cache_path, compress=3)
        if indexes.get("index_version", 0) < 4:
            indexes["transition"] = build_directional_transitions(events[events["ts"] < cutoff_ts])
            indexes["index_version"] = 4
            joblib.dump(indexes, cache_path, compress=3)
        if indexes.get("index_version", 0) < 5:
            indexes["hybrid_popularity"] = build_hybrid_popularity(events[events["ts"] < cutoff_ts], cutoff_ts)
            indexes["index_version"] = 5
            joblib.dump(indexes, cache_path, compress=3)
    else:
        indexes = build_frozen_indexes(events, enriched, category, cutoff_ts)
        joblib.dump(indexes, cache_path, compress=3)
    recalled = recall_from_frozen_indexes(validation, indexes)
    item2vec_metadata = None
    if args.with_item2vec:
        embedding_path = processed / "frozen_item2vec_embeddings_v2.npz"
        if embedding_path.exists():
            embeddings = Item2VecEmbeddings.load(embedding_path)
        else:
            reference = events[events["ts"] < cutoff_ts]
            embeddings = train_item2vec_embeddings(
                reference,
                dimensions=args.item2vec_dimensions,
                window=args.item2vec_window,
                negative_samples=args.item2vec_negative_samples,
                epochs=args.item2vec_epochs,
                batch_size=args.item2vec_batch_size,
                min_count=args.item2vec_min_count,
                subsample=args.item2vec_subsample,
                adaptive_window=True,
                seed=args.seed,
            )
            embeddings.save(embedding_path)
        ann_started = time.perf_counter()
        ann = Item2VecANN(embeddings)
        item2vec_candidates = ann.recall(validation, args.item2vec_topk)
        recalled = pd.concat([recalled, item2vec_candidates], ignore_index=True)
        item2vec_metadata = dict(embeddings.metadata)
        item2vec_metadata["ann_and_query_seconds"] = time.perf_counter() - ann_started
        item2vec_metadata["sampled_auc_gauc"] = ann.sampled_auc_gauc(
            validation, seed=args.seed
        )
    diagnostics = route_diagnostics(recalled, validation_labels)
    diagnostics["conditional_auc_gauc"] = conditional_auc_gauc(
        recalled, validation_labels
    )
    diagnostics["candidate_recall"] = candidate_recall(recalled, validation_labels)
    diagnostics["candidate_recall_by_action"] = candidate_recall_by_action(
        recalled, validation_labels
    )
    history_by_session = dict(
        zip(validation["session"].astype(int), validation["history_aids"])
    )
    novel_labels = validation_labels[
        [
            int(row.target_aid)
            not in set(map(int, history_by_session[int(row.session)]))
            for row in validation_labels.itertuples(index=False)
        ]
    ]
    diagnostics["novel_target"] = {
        "sessions": int(len(novel_labels)),
        "candidate_recall": candidate_recall(recalled, novel_labels),
        "source_recall_at_50": source_recall(recalled, novel_labels, 50),
    }
    diagnostics["source_recall_at_20"] = source_recall(
        recalled, validation_labels, 20
    )
    diagnostics["source_recall_at_50"] = source_recall(
        recalled, validation_labels, 50
    )
    diagnostics["source_recall_at_50_by_action"] = {
        action: source_recall(
            recalled,
            validation_labels[validation_labels["target_type"].eq(action)],
            50,
        )
        for action in ("clicks", "carts", "orders")
    }
    diagnostics["source_catalog_coverage"] = {
        source: float(group["aid"].nunique() / len(indexes["catalog"]))
        for source, group in recalled.groupby("source")
    }
    diagnostics["protocol"] = {
        "validation_samples": len(validation),
        "validation_limit": args.validation_limit,
        "cutoff_ts": cutoff_ts,
        "reference_events": indexes["reference_events"],
        "catalog_items": len(indexes["catalog"]),
        "route_topk": {
            "recent": 30, "hybrid_popular": 100, "category": 150,
            "itemcf": 200, "transition": 150, "item2vec": args.item2vec_topk,
        },
        "index_policy": "frozen_at_validation_start",
        "test_evaluated": False,
        "item2vec_enabled": bool(args.with_item2vec),
    }
    if item2vec_metadata is not None:
        diagnostics["item2vec"] = item2vec_metadata
    diagnostics["runtime_seconds"] = time.perf_counter() - started
    recalled.to_parquet(output / "recall_candidates.parquet", index=False)
    (output / "metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False)
    )
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
