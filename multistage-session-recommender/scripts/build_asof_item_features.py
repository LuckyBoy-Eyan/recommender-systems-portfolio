"""按事件时间连接最近商品状态，并审计全量因果目录。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd


DAY_MS = 86_400_000


def _merge_state(
    left: pd.DataFrame,
    state: pd.DataFrame,
    *,
    value_column: str,
    state_ts_column: str,
) -> pd.DataFrame:
    """按商品执行 backward as-of join，绝不读取查询时间之后的状态。"""
    ordered = left.copy()
    ordered["_row_order"] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values(["ts", "aid", "_row_order"], kind="mergesort")
    right = state.rename(columns={"itemid": "aid", "timestamp": state_ts_column})
    right = right.sort_values([state_ts_column, "aid"], kind="mergesort")
    merged = pd.merge_asof(
        ordered,
        right,
        left_on="ts",
        right_on=state_ts_column,
        by="aid",
        direction="backward",
        allow_exact_matches=True,
    )
    if (
        merged[state_ts_column].notna()
        & (merged[state_ts_column] > merged["ts"])
    ).any():
        raise AssertionError(f"{value_column} 连接使用了未来状态")
    return merged.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)


def enrich_asof(
    frame: pd.DataFrame,
    category_changes: pd.DataFrame,
    availability_changes: pd.DataFrame,
    category_paths: pd.DataFrame,
) -> pd.DataFrame:
    """连接时序类目、可用状态和静态类目祖先路径。"""
    enriched = _merge_state(
        frame,
        category_changes,
        value_column="categoryid",
        state_ts_column="category_state_ts",
    )
    enriched = _merge_state(
        enriched,
        availability_changes,
        value_column="available",
        state_ts_column="availability_state_ts",
    )
    enriched["categoryid"] = enriched["categoryid"].fillna(-1).astype("int64")
    enriched["available"] = enriched["available"].fillna(-1).astype("int8")
    enriched = enriched.merge(category_paths, on="categoryid", how="left")
    enriched["root_categoryid"] = enriched["root_categoryid"].fillna(-1).astype("int64")
    enriched["category_depth"] = enriched["category_depth"].fillna(-1).astype("int16")
    enriched["in_tree"] = enriched["in_tree"].eq(True)
    enriched["category_path"] = enriched["category_path"].map(
        lambda value: list(value)
        if isinstance(value, (list, tuple, np.ndarray))
        else [-1]
    )
    return enriched


def availability_audit(frame: pd.DataFrame, action_column: str) -> dict:
    status_names = {-1: "unknown", 0: "unavailable", 1: "available"}
    counts = frame["available"].value_counts().to_dict()
    by_action = {}
    for action, group in frame.groupby(action_column):
        action_counts = group["available"].value_counts().to_dict()
        by_action[str(action)] = {
            status_names[status]: int(action_counts.get(status, 0))
            for status in (-1, 0, 1)
        }
    known = frame[frame["availability_state_ts"].notna()].copy()
    age_days = (known["ts"] - known["availability_state_ts"]) / DAY_MS
    return {
        "rows": int(len(frame)),
        "status": {
            status_names[status]: int(counts.get(status, 0)) for status in (-1, 0, 1)
        },
        "by_action": by_action,
        "state_age_days": {
            "median": float(age_days.median()) if len(age_days) else None,
            "p90": float(age_days.quantile(0.9)) if len(age_days) else None,
            "p99": float(age_days.quantile(0.99)) if len(age_days) else None,
        },
        "category_known_rate": float(frame["categoryid"].ne(-1).mean()),
    }


def build_catalog_snapshot_summary(
    first_seen: pd.DataFrame, target_times: pd.Series
) -> pd.DataFrame:
    snapshots = np.sort((target_times.astype(np.int64) // DAY_MS * DAY_MS).unique())
    first_times = np.sort(first_seen["first_seen_ts"].astype(np.int64).to_numpy())
    sizes = np.searchsorted(first_times, snapshots, side="left")
    output = pd.DataFrame({"snapshot_ts": snapshots, "catalog_items": sizes})
    output["new_items_since_previous_snapshot"] = output["catalog_items"].diff().fillna(
        output["catalog_items"]
    ).astype(int)
    return output


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建时序商品特征与目录审计")
    parser.add_argument("--processed", default="data/processed/retailrocket")
    args = parser.parse_args()
    started = time.perf_counter()
    root = Path(args.processed)
    events = pd.read_parquet(root / "events_all.parquet")
    labels = pd.read_parquet(root / "labels.parquet")
    category = pd.read_parquet(root / "item_category_changes.parquet")
    availability = pd.read_parquet(root / "item_availability_changes.parquet")
    paths = pd.read_parquet(root / "category_paths.parquet")

    events_enriched = enrich_asof(events, category, availability, paths)
    label_queries = labels.rename(
        columns={
            "target_aid": "aid",
            "target_type": "type",
            "target_ts": "ts",
        }
    )
    labels_enriched = enrich_asof(label_queries, category, availability, paths).rename(
        columns={"aid": "target_aid", "type": "target_type", "ts": "target_ts"}
    )
    first_seen = (
        events.groupby("aid", as_index=False)["ts"]
        .min()
        .rename(columns={"ts": "first_seen_ts"})
    )
    snapshots = build_catalog_snapshot_summary(first_seen, labels["target_ts"])

    write_parquet_atomic(events_enriched, root / "events_enriched.parquet")
    write_parquet_atomic(labels_enriched, root / "labels_enriched.parquet")
    write_parquet_atomic(first_seen, root / "item_first_seen.parquet")
    write_parquet_atomic(snapshots, root / "catalog_snapshot_summary.parquet")
    target_for_audit = labels_enriched.rename(
        columns={"target_type": "type", "target_ts": "ts"}
    )
    development_targets = target_for_audit[~target_for_audit["split"].eq("test")]
    report = {
        "status": "completed",
        "events": availability_audit(events_enriched, "type"),
        "development_targets": availability_audit(development_targets, "type"),
        "targets_by_split": {
            split: availability_audit(group, "type")
            for split, group in development_targets.groupby("split")
        },
        "catalog": {
            "items": int(len(first_seen)),
            "snapshots": int(len(snapshots)),
            "first_snapshot_items": int(snapshots.iloc[0]["catalog_items"]),
            "last_snapshot_items": int(snapshots.iloc[-1]["catalog_items"]),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (root / "asof_item_features_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
