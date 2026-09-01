"""全量 RetailRocket 事件预处理：保留参考事件，只限制监督标签资格。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


TYPE_MAP = {"view": "clicks", "addtocart": "carts", "transaction": "orders"}
QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_time_splits(
    labels: pd.DataFrame, train_ratio: float, valid_ratio: float
) -> pd.Series:
    """按目标时间严格切分，边界并列样本统一进入较晚集合。"""
    ordered = labels.sort_values(["target_ts", "session"], kind="mergesort")
    train_position = int(len(ordered) * train_ratio)
    valid_position = train_position + int(len(ordered) * valid_ratio)
    if not 0 < train_position < valid_position < len(ordered):
        raise ValueError("可标注 Session 数不足，无法切分训练/验证/测试")
    valid_start = int(ordered.iloc[train_position]["target_ts"])
    test_start = int(ordered.iloc[valid_position]["target_ts"])
    split = pd.Series("test", index=labels.index, dtype="string")
    split.loc[labels["target_ts"] < test_start] = "validation"
    split.loc[labels["target_ts"] < valid_start] = "train"
    if set(split.unique()) != {"train", "validation", "test"}:
        raise ValueError("时间戳并列导致某个数据分区为空")
    return split


def session_length_report(sessions: pd.DataFrame) -> dict:
    eligible = sessions[sessions["label_eligible"]]
    lengths = eligible["event_count"]
    report = {
        "all_sessions": int(len(sessions)),
        "label_eligible_sessions": int(len(eligible)),
        "reference_only_sessions": int((~sessions["label_eligible"]).sum()),
        "mean": float(lengths.mean()),
        "max": int(lengths.max()),
        "quantiles": {
            f"p{int(q * 100)}": float(lengths.quantile(q)) for q in QUANTILES
        },
        "length_buckets": {
            "3_5": int(lengths.between(3, 5).sum()),
            "6_10": int(lengths.between(6, 10).sum()),
            "11_20": int(lengths.between(11, 20).sum()),
            "21_50": int(lengths.between(21, 50).sum()),
            "51_100": int(lengths.between(51, 100).sum()),
            "over_100": int((lengths > 100).sum()),
        },
    }
    # 训练前缀的历史长度为 2..N-1；报告各上限能完整保留多少样本上下文。
    total_samples = int((lengths - 2).sum())
    report["prefix_training_samples"] = total_samples
    report["history_cap_coverage"] = {}
    for cap in (20, 50, 100):
        fully_covered = int((np.minimum(lengths - 1, cap) - 1).clip(lower=0).sum())
        report["history_cap_coverage"][str(cap)] = (
            fully_covered / total_samples if total_samples else 0.0
        )
    return report


def preprocess_events(
    raw: pd.DataFrame,
    *,
    session_gap_ms: int = 30 * 60 * 1000,
    min_label_events: int = 3,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """清洗事件、切 Session，并生成 Session 资格与唯一末事件标签。"""
    required = {"timestamp", "visitorid", "event", "itemid", "transactionid"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"原始事件缺少字段: {sorted(missing)}")
    input_rows = len(raw)
    raw = raw[raw["event"].isin(TYPE_MAP)].copy()
    invalid_rows = input_rows - len(raw)
    duplicate_mask = raw.duplicated(
        ["timestamp", "visitorid", "event", "itemid", "transactionid"], keep="first"
    )
    duplicate_rows = int(duplicate_mask.sum())
    raw = raw[~duplicate_mask].copy()
    raw["event_order"] = np.arange(len(raw), dtype=np.int64)
    raw = raw.sort_values(
        ["visitorid", "timestamp", "event_order"], kind="mergesort"
    ).reset_index(drop=True)

    visitor_change = raw["visitorid"].ne(raw["visitorid"].shift())
    gap = raw.groupby("visitorid", sort=False)["timestamp"].diff()
    new_session = visitor_change | gap.gt(session_gap_ms)
    raw["session"] = new_session.cumsum().astype(np.int64) - 1
    raw["type"] = raw["event"].map(TYPE_MAP).astype("string")
    raw["transactionid"] = raw["transactionid"].fillna(-1).astype(np.int64)

    grouped = raw.groupby("session", sort=False)
    sessions = grouped.agg(
        visitorid=("visitorid", "first"),
        event_count=("itemid", "size"),
        unique_item_count=("itemid", "nunique"),
        start_ts=("timestamp", "min"),
        end_ts=("timestamp", "max"),
    ).reset_index()
    max_ts = grouped["timestamp"].transform("max")
    at_last_ts = raw["timestamp"].eq(max_ts)
    last_counts = at_last_ts.groupby(raw["session"], sort=False).sum()
    sessions["last_timestamp_count"] = sessions["session"].map(last_counts).astype(int)
    short = sessions["event_count"] < min_label_events
    ambiguous = sessions["last_timestamp_count"] > 1
    sessions["label_eligible"] = ~(short | ambiguous)
    sessions["ineligible_reason"] = ""
    sessions.loc[short, "ineligible_reason"] = "too_short"
    sessions.loc[ambiguous, "ineligible_reason"] = "ambiguous_last_timestamp"
    sessions.loc[short & ambiguous, "ineligible_reason"] = (
        "too_short+ambiguous_last_timestamp"
    )

    eligible_ids = set(sessions.loc[sessions["label_eligible"], "session"])
    labels = raw[at_last_ts & raw["session"].isin(eligible_ids)][
        ["session", "itemid", "type", "timestamp"]
    ].rename(
        columns={
            "itemid": "target_aid",
            "type": "target_type",
            "timestamp": "target_ts",
        }
    )
    if labels["session"].duplicated().any() or len(labels) != len(eligible_ids):
        raise AssertionError("可标注 Session 没有生成唯一标签")
    labels["split"] = assign_time_splits(labels, train_ratio, valid_ratio)
    sessions = sessions.merge(labels[["session", "split"]], on="session", how="left")
    sessions["split"] = sessions["split"].fillna("reference_only")

    events = raw.rename(
        columns={"itemid": "aid", "timestamp": "ts"}
    )[
        [
            "session",
            "visitorid",
            "aid",
            "ts",
            "type",
            "transactionid",
            "event_order",
        ]
    ]
    report = {
        "input_rows": int(input_rows),
        "output_rows": int(len(events)),
        "invalid_event_rows": int(invalid_rows),
        "exact_duplicate_rows": duplicate_rows,
        "users": int(events["visitorid"].nunique()),
        "items": int(events["aid"].nunique()),
        "event_types": {
            key: int(value) for key, value in events["type"].value_counts().items()
        },
        "label_splits": {
            key: int(value) for key, value in labels["split"].value_counts().items()
        },
        "ineligible_reasons": {
            key: int(value)
            for key, value in sessions.loc[
                ~sessions["label_eligible"], "ineligible_reason"
            ].value_counts().items()
        },
        "session_lengths": session_length_report(sessions),
    }
    return events, sessions, labels, report


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="预处理全量 RetailRocket 事件")
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", default="data/processed/retailrocket")
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--min-label-events", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    args = parser.parse_args()
    started = time.perf_counter()
    source = Path(args.events)
    output = Path(args.output)
    raw = pd.read_csv(
        source,
        dtype={
            "timestamp": "int64",
            "visitorid": "int64",
            "event": "string",
            "itemid": "int64",
            "transactionid": "float64",
        },
    )
    events, sessions, labels, report = preprocess_events(
        raw,
        session_gap_ms=args.session_gap_minutes * 60 * 1000,
        min_label_events=args.min_label_events,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
    write_parquet_atomic(events, output / "events_all.parquet")
    write_parquet_atomic(sessions, output / "sessions.parquet")
    write_parquet_atomic(labels, output / "labels.parquet")
    manifest = {
        "status": "completed",
        "source": str(source),
        "source_sha256": sha256(source),
        "parameters": {
            "session_gap_minutes": args.session_gap_minutes,
            "min_label_events": args.min_label_events,
            "train_ratio": args.train_ratio,
            "valid_ratio": args.valid_ratio,
        },
        "report": report,
        "runtime_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
