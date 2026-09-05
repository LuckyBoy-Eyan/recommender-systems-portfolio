"""完整正负事件序列及围绕正目标的时间切分。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeedbackSequence:
    items: tuple[int, ...]
    positive: tuple[int, ...]

    def __post_init__(self):
        if len(self.items) != len(self.positive):
            raise ValueError("items and positive flags must have equal lengths")
        if any(flag not in (0, 1) for flag in self.positive):
            raise ValueError("feedback flags must be binary")

    def prefix(self, end: int) -> "FeedbackSequence":
        return FeedbackSequence(self.items[:end], self.positive[:end])


def load_feedback_sequences(
    interactions_path: str | Path,
    item_features_path: str | Path,
    *,
    min_positive_events: int = 5,
) -> tuple[np.ndarray, list[FeedbackSequence]]:
    """加载完整事件，保留正负标记并映射到连续 item index。"""
    columns = set(pd.read_csv(interactions_path, nrows=0).columns)
    required = {"user_id", "item_id", "timestamp", "is_positive"}
    missing = required - columns
    if missing:
        raise ValueError(f"Missing interaction columns: {sorted(missing)}")
    frame = pd.read_csv(
        interactions_path,
        usecols=sorted(required),
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "timestamp": "float64",
            "is_positive": "int8",
        },
    ).sort_values(["user_id", "timestamp"])
    item_ids = np.sort(frame["item_id"].unique())
    item_to_index = {int(item): index for index, item in enumerate(item_ids)}
    sequences = []
    for _, group in frame.groupby("user_id", sort=False):
        flags = tuple(int(value) for value in group["is_positive"].tolist())
        if sum(flags) < min_positive_events:
            continue
        items = tuple(item_to_index[int(item)] for item in group["item_id"])
        sequences.append(FeedbackSequence(items, flags))
    features = np.load(item_features_path).astype(np.float32)
    if len(features) != len(item_ids):
        raise ValueError("Feature rows must match sorted unique item IDs")
    return features, sequences


def positive_target_leave_two_out(
    sequences: list[FeedbackSequence],
    *,
    min_train_positive: int = 3,
) -> tuple[list[FeedbackSequence], list[FeedbackSequence], list[FeedbackSequence]]:
    """最后两个正事件作为验证/测试目标，期间负事件留在上下文中。"""
    train, validation, test = [], [], []
    for sequence in sequences:
        positive_positions = [
            index for index, flag in enumerate(sequence.positive) if flag == 1
        ]
        if len(positive_positions) < min_train_positive + 2:
            continue
        validation_target, test_target = positive_positions[-2:]
        train.append(sequence.prefix(validation_target))
        validation.append(sequence.prefix(validation_target + 1))
        test.append(sequence.prefix(test_target + 1))
    return train, validation, test
