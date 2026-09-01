"""统一加载 OTTO 风格的会话行为数据。"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

TYPE_MAP = {0: "clicks", 1: "carts", 2: "orders"}
REQUIRED_COLUMNS = {"session", "aid", "ts", "type"}


def load_events(path: str | Path, max_sessions: int | None = None) -> pd.DataFrame:
    """读取并标准化 Parquet、CSV、JSONL 或按行 JSON 事件。"""
    path = Path(path)
    if path.suffix == ".parquet":
        events = pd.read_parquet(path)
    elif path.suffix == ".csv":
        events = pd.read_csv(path)
    elif path.suffix in {".jsonl", ".json"}:
        events = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported data format: {path.suffix}")
    missing = REQUIRED_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    events = events[["session", "aid", "ts", "type"]].copy()
    events["type"] = events["type"].replace(TYPE_MAP).astype(str)
    events = events[events["type"].isin(TYPE_MAP.values())]
    if max_sessions is not None:
        sessions = events.groupby("session")["ts"].min().nsmallest(max_sessions).index
        events = events[events["session"].isin(sessions)]
    return events.sort_values(["session", "ts"]).reset_index(drop=True)
