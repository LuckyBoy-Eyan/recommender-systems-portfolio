"""完整反馈序列上的生成式正目标样本与 masked SASRec 样本。"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.data.feedback import FeedbackSequence


def _padded_history(items, flags, max_history: int):
    items = items[-max_history:]
    flags = flags[-max_history:]
    padding = max_history - len(items)
    item_tokens = torch.zeros(max_history, dtype=torch.long)
    feedback_tokens = torch.zeros(max_history, dtype=torch.long)
    history_items = torch.full((max_history,), -1, dtype=torch.long)
    if items:
        item_tokens[padding:] = torch.tensor(items, dtype=torch.long) + 1
        # 0=padding, 1=negative, 2=positive
        feedback_tokens[padding:] = torch.tensor(flags, dtype=torch.long) + 1
        history_items[padding:] = torch.tensor(items, dtype=torch.long)
    return item_tokens, feedback_tokens, history_items


class PositiveTargetDataset(Dataset):
    """只为正目标建样本，但完整正负历史全部进入上下文。"""

    def __init__(
        self,
        sequences: list[FeedbackSequence],
        item_codes,
        max_history: int,
        *,
        last_only: bool = False,
        max_samples_per_sequence: int | None = None,
    ):
        self.item_codes = torch.as_tensor(item_codes, dtype=torch.long)
        self.max_history = max_history
        self.samples = []
        for sequence in sequences:
            indices = [
                index
                for index in range(1, len(sequence.items))
                if sequence.positive[index] == 1
            ]
            if last_only:
                indices = indices[-1:]
            elif max_samples_per_sequence is not None:
                indices = indices[-max_samples_per_sequence:]
            for target_index in indices:
                self.samples.append((sequence, target_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence, target_index = self.samples[index]
        items = sequence.items[:target_index]
        flags = sequence.positive[:target_index]
        _, feedback_tokens, history_items = _padded_history(
            items, flags, self.max_history
        )
        recent_items = items[-self.max_history:]
        history_codes = self.item_codes[list(recent_items)]
        padding = self.max_history - len(recent_items)
        if padding:
            history_codes = torch.cat(
                [
                    torch.zeros(
                        padding, history_codes.shape[1], dtype=torch.long
                    ),
                    history_codes + 1,
                ]
            )
        else:
            history_codes = history_codes + 1
        target = sequence.items[target_index]
        return (
            history_codes,
            feedback_tokens,
            self.item_codes[target],
            torch.tensor(target),
            history_items,
        )


class MaskedSASRecDataset(Dataset):
    """滑窗内并行预测下一事件，仅正目标位置保留标签。"""

    def __init__(
        self,
        sequences: list[FeedbackSequence],
        max_history: int,
        *,
        target_stride: int | None = None,
    ):
        self.max_history = max_history
        self.target_stride = target_stride or max_history
        if not 1 <= self.target_stride <= max_history:
            raise ValueError("target_stride must be in [1, max_history]")
        self.windows = []
        for sequence in sequences:
            # target positions are 1..n-1; overlapping context does not duplicate loss.
            for block_start in range(1, len(sequence.items), self.target_stride):
                block_end = min(block_start + self.target_stride, len(sequence.items))
                window_start = max(0, block_end - (max_history + 1))
                self.windows.append(
                    (sequence, window_start, block_start, block_end)
                )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        sequence, window_start, block_start, block_end = self.windows[index]
        event_items = sequence.items[window_start:block_end]
        event_flags = sequence.positive[window_start:block_end]
        inputs = event_items[:-1]
        input_flags = event_flags[:-1]
        targets = event_items[1:]
        target_flags = event_flags[1:]
        padding = self.max_history - len(inputs)
        item_tokens = torch.zeros(self.max_history, dtype=torch.long)
        feedback_tokens = torch.zeros(self.max_history, dtype=torch.long)
        labels = torch.full((self.max_history,), -100, dtype=torch.long)
        if inputs:
            item_tokens[padding:] = torch.tensor(inputs) + 1
            feedback_tokens[padding:] = torch.tensor(input_flags) + 1
            target_positions = range(window_start + 1, block_end)
            values = [
                target
                if flag == 1 and position >= block_start
                else -100
                for target, flag, position in zip(
                    targets, target_flags, target_positions
                )
            ]
            labels[padding:] = torch.tensor(values, dtype=torch.long)
        return item_tokens, feedback_tokens, labels


class SelectedPositiveTargetSASRecDataset(Dataset):
    """与生成主线共享正目标，每个目标保留此前完整的定长上下文。"""

    def __init__(
        self,
        sequences: list[FeedbackSequence],
        max_history: int,
        *,
        max_targets_per_sequence: int | None = None,
    ):
        self.max_history = max_history
        self.samples = []
        for sequence in sequences:
            indices = [
                index
                for index in range(1, len(sequence.items))
                if sequence.positive[index] == 1
            ]
            if max_targets_per_sequence is not None:
                indices = indices[-max_targets_per_sequence:]
            self.samples.extend((sequence, target_index) for target_index in indices)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence, target_index = self.samples[index]
        item_tokens, feedback_tokens, _ = _padded_history(
            sequence.items[:target_index],
            sequence.positive[:target_index],
            self.max_history,
        )
        labels = torch.full((self.max_history,), -100, dtype=torch.long)
        # SASRec 最后一个有效状态预测当前目标；目标本身从未进入输入。
        labels[-1] = sequence.items[target_index]
        return item_tokens, feedback_tokens, labels


class SASRecPositiveEvalDataset(Dataset):
    """每位用户最后一个正事件的全历史评价样本。"""

    def __init__(self, sequences: list[FeedbackSequence], max_history: int):
        self.max_history = max_history
        self.samples = []
        for sequence in sequences:
            positive = [i for i, flag in enumerate(sequence.positive) if flag == 1]
            if positive:
                self.samples.append((sequence, positive[-1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence, target_index = self.samples[index]
        item_tokens, feedback_tokens, history_items = _padded_history(
            sequence.items[:target_index],
            sequence.positive[:target_index],
            self.max_history,
        )
        return (
            item_tokens,
            feedback_tokens,
            torch.tensor(sequence.items[target_index]),
            history_items,
        )
