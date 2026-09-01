"""从已处理 Session 生成因果滑动前缀训练样本和末事件评估样本。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd


TYPE_TO_ID = {"clicks": 0, "carts": 1, "orders": 2}


def _sample_row(
    session: int,
    history: pd.DataFrame,
    target,
    max_history: int,
) -> dict:
    full_length = len(history)
    history = history.tail(max_history)
    target_ts = int(target.ts)
    return {
        "session": int(session),
        "target_aid": int(target.aid),
        "target_type": str(target.type),
        "target_type_id": TYPE_TO_ID[str(target.type)],
        "target_ts": target_ts,
        "history_aids": history["aid"].astype(int).tolist(),
        "history_type_ids": history["type"].map(TYPE_TO_ID).astype(int).tolist(),
        "history_time_deltas_ms": (target_ts - history["ts"]).astype(int).tolist(),
        "history_length_full": int(full_length),
        "history_length_used": int(len(history)),
    }


def build_training_samples(
    events: pd.DataFrame,
    train_sessions: set[int],
    *,
    min_history: int = 2,
    max_history: int = 30,
    max_samples_per_session: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """每个唯一目标时间生成前缀样本；并列目标时间不强造顺序。"""
    rows = []
    candidate_samples = 0
    skipped_tied_targets = 0
    capped_sessions = 0
    selected_samples = 0
    train_events = events[events["session"].isin(train_sessions)].sort_values(
        ["session", "ts", "event_order"], kind="mergesort"
    )
    for session, group in train_events.groupby("session", sort=False):
        target_positions = []
        timestamp_counts = group["ts"].value_counts()
        for position in range(min_history, len(group)):
            target_ts = int(group.iloc[position]["ts"])
            history = group[group["ts"] < target_ts]
            if len(history) < min_history:
                continue
            if int(timestamp_counts[target_ts]) != 1:
                skipped_tied_targets += 1
                continue
            target_positions.append((position, history))
        candidate_samples += len(target_positions)
        if len(target_positions) > max_samples_per_session:
            capped_sessions += 1
            chosen = np.rint(
                np.linspace(0, len(target_positions) - 1, max_samples_per_session)
            ).astype(int)
            target_positions = [target_positions[index] for index in chosen]
        for position, history in target_positions:
            rows.append(
                _sample_row(
                    int(session), history, group.iloc[position], max_history
                )
            )
        selected_samples += len(target_positions)
    return pd.DataFrame(rows), {
        "candidate_samples": int(candidate_samples),
        "output_samples": int(selected_samples),
        "skipped_tied_targets": int(skipped_tied_targets),
        "capped_sessions": int(capped_sessions),
    }


def build_evaluation_samples(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    split: str,
    *,
    min_history: int = 2,
    max_history: int = 30,
) -> pd.DataFrame:
    """验证/测试每个 Session 只使用唯一末事件作为目标。"""
    split_labels = labels[labels["split"].eq(split)].set_index("session")
    split_events = events[events["session"].isin(split_labels.index)].sort_values(
        ["session", "ts", "event_order"], kind="mergesort"
    )
    rows = []
    for session, group in split_events.groupby("session", sort=False):
        label = split_labels.loc[session]
        history = group[group["ts"] < int(label.target_ts)]
        if len(history) < min_history:
            raise AssertionError("评估 Session 的严格历史少于 min_history")
        target = pd.Series(
            {
                "aid": int(label.target_aid),
                "type": str(label.target_type),
                "ts": int(label.target_ts),
            }
        )
        rows.append(_sample_row(int(session), history, target, max_history))
    return pd.DataFrame(rows)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RetailRocket 序列样本")
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--min-history", type=int, default=2)
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--max-samples-per-session", type=int, default=50)
    args = parser.parse_args()
    if not 0 < args.min_history <= args.max_history:
        raise ValueError("历史长度参数非法")
    if args.max_samples_per_session < 1:
        raise ValueError("max_samples_per_session 必须为正数")
    started = time.perf_counter()
    root = Path(args.processed)
    events = pd.read_parquet(root / "events_all.parquet")
    sessions = pd.read_parquet(root / "sessions.parquet")
    labels = pd.read_parquet(root / "labels.parquet")
    train_sessions = set(
        sessions.loc[sessions["split"].eq("train"), "session"].astype(int)
    )
    train, train_report = build_training_samples(
        events,
        train_sessions,
        min_history=args.min_history,
        max_history=args.max_history,
        max_samples_per_session=args.max_samples_per_session,
    )
    validation = build_evaluation_samples(
        events,
        labels,
        "validation",
        min_history=args.min_history,
        max_history=args.max_history,
    )
    test = build_evaluation_samples(
        events,
        labels,
        "test",
        min_history=args.min_history,
        max_history=args.max_history,
    )
    write_parquet_atomic(train, root / "train_samples.parquet")
    write_parquet_atomic(validation, root / "validation_samples.parquet")
    write_parquet_atomic(test, root / "test_samples.parquet")
    report = {
        "status": "completed",
        "parameters": {
            "min_history": args.min_history,
            "max_history": args.max_history,
            "max_samples_per_session": args.max_samples_per_session,
        },
        "train": train_report,
        "validation_samples": len(validation),
        "test_samples": len(test),
        "runtime_seconds": time.perf_counter() - started,
    }
    (root / "sequence_samples_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
