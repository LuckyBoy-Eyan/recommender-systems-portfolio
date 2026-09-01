"""构造无歧义预测标签，并按目标时间切分训练、验证和测试集。"""

from __future__ import annotations
import pandas as pd


def drop_ambiguous_target_sessions(events: pd.DataFrame, ts_column: str = "ts") -> pd.DataFrame:
    """删除最大时间戳对应多个事件的整个 Session。"""
    max_ts = events.groupby("session")[ts_column].transform("max")
    at_max = events[ts_column].eq(max_ts)
    max_counts = at_max.groupby(events["session"]).sum()
    ambiguous_sessions = set(max_counts[max_counts > 1].index)
    return events[~events["session"].isin(ambiguous_sessions)].reset_index(drop=True)


def leave_last_event_out(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把每个 Session 的唯一最后事件留作监督标签。"""
    events = drop_ambiguous_target_sessions(events)
    ordered = events.sort_values(["session", "ts"]).copy()
    last_indices = ordered.groupby("session").tail(1).index
    labels = ordered.loc[last_indices, ["session", "aid", "type", "ts"]].rename(
        columns={"aid": "target_aid", "type": "target_type", "ts": "target_ts"}
    )
    history = ordered.drop(last_indices).merge(
        labels[["session", "target_ts"]], on="session", how="left"
    )
    history = history[history["ts"] < history["target_ts"]].drop(columns="target_ts")
    return history.reset_index(drop=True), labels.reset_index(drop=True)


def split_sessions(
    events: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按照 Session 最后事件时间切分训练、验证和测试集。"""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than 1")
    target_times = events.groupby("session")["ts"].max().sort_values(kind="mergesort")
    train_boundary = int(len(target_times) * train_ratio)
    valid_boundary = train_boundary + int(len(target_times) * valid_ratio)
    if train_boundary == 0 or valid_boundary == train_boundary or valid_boundary == len(target_times):
        raise ValueError("not enough sessions to create non-empty train/validation/test sets")
    valid_start_ts = target_times.iloc[train_boundary]
    test_start_ts = target_times.iloc[valid_boundary]
    train_sessions = set(target_times[target_times < valid_start_ts].index)
    valid_sessions = set(target_times[(target_times >= valid_start_ts) & (target_times < test_start_ts)].index)
    test_sessions = set(target_times[target_times >= test_start_ts].index)
    if not train_sessions or not valid_sessions or not test_sessions:
        raise ValueError("timestamp ties produced an empty train/validation/test split")
    return (
        events[events["session"].isin(train_sessions)].copy().reset_index(drop=True),
        events[events["session"].isin(valid_sessions)].copy().reset_index(drop=True),
        events[events["session"].isin(test_sessions)].copy().reset_index(drop=True),
    )
