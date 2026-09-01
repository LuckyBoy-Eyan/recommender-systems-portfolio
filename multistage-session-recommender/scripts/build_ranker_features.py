"""Build point-in-time candidate features for PLE/MMoE training and validation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare_samples(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.read_parquet(path).sort_values(["target_ts", "session"], kind="mergesort").reset_index(drop=True)
    samples["sample_id"] = np.arange(len(samples), dtype=np.int64)
    session_rows = []
    pair_rows = []
    for row in samples.itertuples(index=False):
        aids = [int(value) for value in row.history_aids]
        types = [int(value) for value in row.history_type_ids]
        counts = Counter(aids)
        recency = {}
        for rank, aid in enumerate(reversed(aids), 1):
            recency.setdefault(aid, rank)
        for aid, count in counts.items():
            pair_rows.append((int(row.sample_id), aid, count, recency[aid]))
        session_rows.append(
            (
                int(row.sample_id), int(row.target_ts), int(row.target_aid), str(row.target_type),
                len(aids), len(counts), aids[-1], types[-1],
                float(max(row.history_time_deltas_ms) / 60_000),
            )
        )
    session_features = pd.DataFrame(
        session_rows,
        columns=[
            "sample_id", "target_ts", "target_aid", "target_type", "history_length",
            "history_unique_items", "last_aid", "last_type_id", "session_span_minutes",
        ],
    )
    pair_features = pd.DataFrame(
        pair_rows,
        columns=["sample_id", "aid", "history_candidate_count", "history_candidate_recency"],
    )
    return session_features, pair_features


def snapshot_item_features(
    snapshot_ts: int,
    category_changes: pd.DataFrame,
    availability_changes: pd.DataFrame,
    category_paths: pd.DataFrame,
    first_seen: pd.DataFrame,
) -> pd.DataFrame:
    category = category_changes[category_changes["timestamp"] < snapshot_ts].sort_values(
        ["itemid", "timestamp"]
    ).groupby("itemid").tail(1).rename(
        columns={"itemid": "aid", "timestamp": "category_state_ts"}
    )
    available = availability_changes[availability_changes["timestamp"] < snapshot_ts].sort_values(
        ["itemid", "timestamp"]
    ).groupby("itemid").tail(1).rename(
        columns={"itemid": "aid", "timestamp": "availability_state_ts"}
    )
    features = category.merge(category_paths, on="categoryid", how="left").merge(
        available, on="aid", how="outer"
    ).merge(first_seen, on="aid", how="outer")
    features["candidate_age_days"] = (snapshot_ts - features["first_seen_ts"]) / 86_400_000
    features["category_state_age_days"] = (
        snapshot_ts - features["category_state_ts"]
    ) / 86_400_000
    features["availability_state_age_days"] = (
        snapshot_ts - features["availability_state_ts"]
    ) / 86_400_000
    return features.drop(columns=["category_path"], errors="ignore")


def enrich_file(
    candidate_path: Path,
    output_path: Path,
    session_features: pd.DataFrame,
    pair_features: pd.DataFrame,
    item_features: pd.DataFrame,
    *,
    training: bool,
    batch_size: int,
) -> dict:
    last_category = dict(zip(item_features["aid"].astype(int), item_features["categoryid"].fillna(-1).astype(int)))
    session = session_features.copy()
    session["last_categoryid"] = session["last_aid"].map(last_category).fillna(-1).astype(int)
    first_seen_map = dict(zip(item_features["aid"].astype(int), item_features["first_seen_ts"]))
    impossible = set()
    if training:
        impossible = set(
            session.loc[
                session["target_aid"].map(first_seen_map).fillna(np.inf) >= session["target_ts"],
                "sample_id",
            ].astype(int)
        )
    writer = None
    rows = 0
    samples = set()
    removed_samples = set()
    try:
        parquet = pq.ParquetFile(candidate_path)
        for record_batch in parquet.iter_batches(batch_size=batch_size):
            frame = record_batch.to_pandas()
            if impossible:
                removed_samples.update(
                    set(frame.loc[frame["sample_id"].isin(impossible), "sample_id"].astype(int))
                )
                frame = frame[~frame["sample_id"].isin(impossible)]
            frame = frame.merge(session, on="sample_id", how="left", validate="many_to_one")
            frame = frame.merge(pair_features, on=["sample_id", "aid"], how="left", validate="many_to_one")
            frame = frame.merge(item_features, on="aid", how="left", validate="many_to_one")
            frame["candidate_is_last"] = frame["aid"].eq(frame["last_aid"]).astype(np.int8)
            frame["candidate_in_history"] = frame["history_candidate_count"].notna().astype(np.int8)
            frame["same_category_as_last"] = (
                frame["categoryid"].fillna(-1).astype(int).eq(frame["last_categoryid"])
                & frame["categoryid"].fillna(-1).astype(int).ne(-1)
            ).astype(np.int8)
            frame[["history_candidate_count", "history_candidate_recency"]] = frame[
                ["history_candidate_count", "history_candidate_recency"]
            ].fillna(0)
            rank_columns = [column for column in frame if column.startswith("source_rank_")]
            score_columns = [column for column in frame if column.startswith("source_score_")]
            for column in rank_columns:
                frame[column.replace("source_rank_", "source_present_")] = frame[column].notna().astype(np.int8)
            # 各路预算已扩到100--300；旧哨兵51会让“该路未召回”看起来优于真实的深位候选。
            # source_present_*负责表达是否命中，缺失排名使用稳定的大哨兵。
            frame[rank_columns] = frame[rank_columns].fillna(10_000)
            frame[score_columns] = frame[score_columns].fillna(0.0)
            # 累积序数标签：click <= cart <= order。
            positive = frame["label"].astype(int)
            frame["label_clicks"] = positive
            frame["label_carts"] = positive * frame["target_type_x"].isin(["carts", "orders"]).astype(int)
            frame["label_orders"] = positive * frame["target_type_x"].eq("orders").astype(int)
            frame = frame.rename(columns={"target_type_x": "target_type"}).drop(
                columns=["target_type_y"], errors="ignore"
            )
            if "target_ts_x" in frame.columns:
                frame = frame.rename(columns={"target_ts_x": "target_ts"}).drop(
                    columns=["target_ts_y"], errors="ignore"
                )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
            samples.update(frame["sample_id"].astype(int).unique())
    finally:
        if writer is not None:
            writer.close()
    return {"rows": rows, "samples": len(samples), "removed_impossible_samples": len(removed_samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 PLE/MMoE 严格时间点特征")
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--oof", default="outputs/rolling_oof_candidates_compact")
    parser.add_argument("--validation", default="outputs/ranker_datasets/validation_candidates.parquet")
    parser.add_argument("--train-candidates", default=None, help="直接训练候选文件；设置后不读取OOF fold文件")
    parser.add_argument("--output", default="outputs/ranker_datasets/features")
    parser.add_argument("--batch-size", type=int, default=250000)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    processed = ROOT / args.processed
    output = ROOT / args.output
    category_changes = pd.read_parquet(processed / "item_category_changes.parquet")
    availability_changes = pd.read_parquet(processed / "item_availability_changes.parquet")
    category_paths = pd.read_parquet(processed / "category_paths.parquet")
    first_seen = pd.read_parquet(processed / "item_first_seen.parquet")
    validation_session, validation_pairs = prepare_samples(processed / "validation_samples.parquet")
    reports = {}
    if not args.validation_only and args.train_candidates:
        train_session, train_pairs = prepare_samples(processed / "train_samples.parquet")
        train_snapshot = int(pd.read_parquet(processed / "labels.parquet").query(
            "split == 'validation'"
        )["target_ts"].min())
        train_item_features = snapshot_item_features(
            train_snapshot, category_changes, availability_changes, category_paths, first_seen
        )
        reports["train"] = enrich_file(
            ROOT / args.train_candidates, output / "train.parquet",
            train_session, train_pairs, train_item_features,
            training=True, batch_size=args.batch_size,
        )
    elif not args.validation_only:
        train_session, train_pairs = prepare_samples(processed / "train_samples.parquet")
        for fold in range(4):
            candidate_path = ROOT / args.oof / f"fold_{fold}_candidates.parquet"
            snapshot = int(pq.read_table(candidate_path, columns=["snapshot_ts"]).column(0)[0].as_py())
            item_features = snapshot_item_features(
                snapshot, category_changes, availability_changes, category_paths, first_seen
            )
            reports[f"train_fold_{fold}"] = enrich_file(
                candidate_path, output / f"train_fold_{fold}.parquet",
                train_session, train_pairs, item_features, training=True, batch_size=args.batch_size,
            )
    validation_path = ROOT / args.validation
    validation_snapshot = int(pd.read_parquet(processed / "labels.parquet").query(
        "split == 'validation'"
    )["target_ts"].min())
    item_features = snapshot_item_features(
        validation_snapshot, category_changes, availability_changes, category_paths, first_seen
    )
    reports["validation"] = enrich_file(
        validation_path, output / "validation.parquet", validation_session,
        validation_pairs, item_features, training=False, batch_size=args.batch_size,
    )
    pd.Series(reports).to_json(output / "manifest.json", indent=2, force_ascii=False)
    print(reports)


if __name__ == "__main__":
    main()
