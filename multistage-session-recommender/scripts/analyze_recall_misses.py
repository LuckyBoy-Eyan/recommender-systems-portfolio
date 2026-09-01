"""Diagnose frozen-validation candidate misses without touching the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCES = ("category", "item2vec", "itemcf", "hybrid_popular", "recent", "transition", "two_tower")


def grouped(frame: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for value, group in frame.groupby(column, dropna=False, observed=True):
        rows.append({
            "segment": str(value),
            "samples": int(len(group)),
            "misses": int((~group["recalled"]).sum()),
            "recall": float(group["recalled"].mean()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--features", default="outputs/ranker_datasets/features/validation.parquet")
    parser.add_argument("--output", default="outputs/recall_miss_diagnostics/metrics.json")
    parser.add_argument("--cutoff-ts", type=int, default=1438311330967)
    args = parser.parse_args()

    root = Path(args.processed)
    samples = pd.read_parquet(root / "validation_samples.parquet").sort_values(
        ["target_ts", "session"], kind="mergesort"
    ).reset_index(drop=True)
    samples["sample_id"] = np.arange(len(samples), dtype=np.int64)

    source_columns = [f"source_present_{source}" for source in SOURCES]
    candidates = pd.read_parquet(
        args.features, columns=["sample_id", "label"] + source_columns
    )
    positive = candidates[candidates["label"].eq(1)].drop_duplicates("sample_id")
    samples["recalled"] = samples["sample_id"].isin(positive["sample_id"])
    samples["target_in_history"] = [
        int(target) in set(map(int, history))
        for target, history in zip(samples["target_aid"], samples["history_aids"])
    ]
    samples["history_bucket"] = pd.cut(
        samples["history_length_full"], [0, 2, 5, 10, 20, np.inf],
        labels=["2", "3-5", "6-10", "11-20", "21+"], include_lowest=True,
    )

    first_seen = pd.read_parquet(root / "item_first_seen.parquet")
    first_seen_map = dict(zip(first_seen["aid"].astype(int), first_seen["first_seen_ts"].astype(int)))
    samples["first_seen_ts"] = samples["target_aid"].map(first_seen_map)
    samples["catalog_status"] = np.where(
        samples["first_seen_ts"] < args.cutoff_ts, "known_at_cutoff", "first_seen_after_cutoff"
    )

    events = pd.read_parquet(root / "events_all.parquet", columns=["aid", "ts"])
    popularity = events.loc[events["ts"] < args.cutoff_ts, "aid"].value_counts()
    samples["reference_events"] = samples["target_aid"].map(popularity).fillna(0).astype(int)
    samples["popularity_bucket"] = pd.cut(
        samples["reference_events"], [-1, 0, 1, 4, 19, np.inf],
        labels=["0", "1", "2-4", "5-19", "20+"],
    )

    category = pd.read_parquet(root / "item_category_changes.parquet")
    category_first = category.groupby("itemid")["timestamp"].min()
    category = category[category["timestamp"] < args.cutoff_ts].sort_values(
        ["itemid", "timestamp"]
    ).groupby("itemid").tail(1)
    samples["category_known"] = samples["target_aid"].isin(category["itemid"])
    item_category = dict(zip(category["itemid"].astype(int), category["categoryid"].astype(int)))
    paths = pd.read_parquet(root / "category_paths.parquet")
    category_path = {
        int(row.categoryid): tuple(map(int, row.category_path))
        for row in paths.itertuples(index=False) if bool(row.in_tree)
    }

    def category_relation(row) -> str:
        target_category = item_category.get(int(row.target_aid))
        if target_category is None:
            return "target_unknown"
        history_categories = {
            item_category[int(aid)] for aid in row.history_aids if int(aid) in item_category
        }
        if target_category in history_categories:
            return "exact_category"
        target_path = category_path.get(target_category)
        history_paths = [category_path[value] for value in history_categories if value in category_path]
        if target_path and len(target_path) > 1 and any(
            len(path) > 1 and path[-2] == target_path[-2] for path in history_paths
        ):
            return "same_parent"
        if target_path and any(path[0] == target_path[0] for path in history_paths):
            return "same_root"
        return "different_root_or_history_unknown"

    samples["category_relation"] = [category_relation(row) for row in samples.itertuples(index=False)]

    availability = pd.read_parquet(root / "item_availability_changes.parquet")
    availability_first = availability.groupby("itemid")["timestamp"].min()
    metadata_first = pd.concat([category_first, availability_first], axis=1).min(axis=1)
    samples["metadata_first_seen_ts"] = samples["target_aid"].map(metadata_first)
    samples["online_catalog_status"] = np.select(
        [
            samples["first_seen_ts"] < args.cutoff_ts,
            samples["first_seen_ts"] < samples["target_ts"],
            samples["metadata_first_seen_ts"] < args.cutoff_ts,
            samples["metadata_first_seen_ts"] < samples["target_ts"],
        ],
        [
            "behavior_known_at_cutoff",
            "behavior_seen_before_target",
            "metadata_known_at_cutoff",
            "metadata_seen_before_target",
        ],
        default="unknown_before_target",
    )
    availability = availability[availability["timestamp"] < args.cutoff_ts].sort_values(
        ["itemid", "timestamp"]
    ).groupby("itemid").tail(1).set_index("itemid")["available"]
    state = samples["target_aid"].map(availability)
    samples["availability_at_cutoff"] = np.select(
        [state.eq(1), state.eq(0)], ["available", "unavailable"], default="unknown"
    )

    hits = positive.set_index("sample_id")[source_columns].astype(bool)
    union_ids = set(hits.index.astype(int))
    source_report = {}
    for source, column in zip(SOURCES, source_columns):
        source_ids = set(hits.index[hits[column]])
        other_columns = [name for name in source_columns if name != column]
        other_ids = set(hits.index[hits[other_columns].any(axis=1)])
        source_report[source] = {
            "recall": len(source_ids) / len(samples),
            "hits": len(source_ids),
            "exclusive_hits": len(source_ids - other_ids),
            "leave_one_out_recall_drop": len(union_ids - other_ids) / len(samples),
        }

    novel_ids = set(samples.loc[~samples["target_in_history"], "sample_id"].astype(int))
    novel_source_report = {}
    for source, column in zip(SOURCES, source_columns):
        source_ids = set(hits.index[hits[column]]) & novel_ids
        other_columns = [name for name in source_columns if name != column]
        other_ids = set(hits.index[hits[other_columns].any(axis=1)]) & novel_ids
        novel_source_report[source] = {
            "recall": len(source_ids) / len(novel_ids),
            "hits": len(source_ids),
            "exclusive_hits": len(source_ids - other_ids),
        }

    misses = samples[~samples["recalled"]]
    known_misses = misses[misses["catalog_status"].eq("known_at_cutoff")]
    novel = samples[~samples["target_in_history"]]
    report = {
        "protocol": {"split": "validation", "test_evaluated": False, "cutoff_ts": args.cutoff_ts},
        "summary": {
            "samples": int(len(samples)),
            "recalled": int(samples["recalled"].sum()),
            "misses": int(len(misses)),
            "candidate_recall": float(samples["recalled"].mean()),
            "novel_targets": int((~samples["target_in_history"]).sum()),
            "novel_misses": int((~misses["target_in_history"]).sum()),
            "catalog_unseen_targets": int(samples["catalog_status"].eq("first_seen_after_cutoff").sum()),
            "catalog_unseen_misses": int(misses["catalog_status"].eq("first_seen_after_cutoff").sum()),
            "known_catalog_misses": int(len(known_misses)),
        },
        "by_action": grouped(samples, "target_type"),
        "by_history_length": grouped(samples, "history_bucket"),
        "by_repeat_status": grouped(samples.assign(
            repeat_status=np.where(samples["target_in_history"], "repeat", "novel_to_session")
        ), "repeat_status"),
        "by_catalog_status": grouped(samples, "catalog_status"),
        "by_online_catalog_status": grouped(samples, "online_catalog_status"),
        "by_reference_popularity": grouped(samples, "popularity_bucket"),
        "by_category_metadata": grouped(samples.assign(
            category_metadata=np.where(samples["category_known"], "known", "unknown")
        ), "category_metadata"),
        "by_availability": grouped(samples, "availability_at_cutoff"),
        "sources": source_report,
        "novel_only": {
            "by_action": grouped(novel, "target_type"),
            "by_history_length": grouped(novel, "history_bucket"),
            "by_catalog_status": grouped(novel, "catalog_status"),
            "by_online_catalog_status": grouped(novel, "online_catalog_status"),
            "by_reference_popularity": grouped(novel, "popularity_bucket"),
            "by_category_relation": grouped(novel, "category_relation"),
            "sources": novel_source_report,
        },
        "known_catalog_miss_breakdown": {
            "samples": int(len(known_misses)),
            "novel_to_session": int((~known_misses["target_in_history"]).sum()),
            "repeat": int(known_misses["target_in_history"].sum()),
            "reference_events_le_4": int(known_misses["reference_events"].le(4).sum()),
            "category_unknown": int((~known_misses["category_known"]).sum()),
            "unavailable_at_cutoff": int(known_misses["availability_at_cutoff"].eq("unavailable").sum()),
            "category_relation": {
                str(key): int(value)
                for key, value in known_misses["category_relation"].value_counts().items()
            },
        },
        "frozen_catalog_miss_breakdown": {
            str(key): int(value)
            for key, value in misses["online_catalog_status"].value_counts().items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
