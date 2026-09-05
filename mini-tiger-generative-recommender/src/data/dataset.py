"""把用户物品序列转换为 PyTorch 的 next-item 监督学习样本。

调用关系：
    scripts/run_demo.py 创建 NextItemDataset
        ├─ training/train.py 的 DataLoader 迭代训练样本
        └─ training/evaluate.py 的 DataLoader 迭代评估样本

这里完成两个关键转换：
1. 把 ``[item1, item2, item3]`` 切成“历史 -> 下一物品”；
2. 把原始 item index 查表替换成多级 Semantic ID。
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class NextItemDataset(Dataset):
    """
    把用户行为序列切成 next-item prediction 样本。

    原始序列类似 [item1, item2, item3, item4]。
    训练样本会变成：
        历史 [item1, item2] -> 目标 item3
        历史 [item1, item2, item3] -> 目标 item4

    模型真正接收的不是原始 item_id，而是每个 item 对应的 Semantic ID。
    """

    def __init__(
        self,
        sequences: list[list[int]],
        item_codes,
        max_history: int,
        last_only: bool = False,
        max_samples_per_sequence: int | None = None,
    ):
        """预先建立所有可用的“历史、目标物品”索引。

        参数:
            sequences: 用户行为序列列表；每条序列中的整数都是从 0 开始的
                item index，例如 ``[[3, 8, 2], [1, 5, 7, 4]]``。
            item_codes: 商品目录编码表，形状为 [num_items, num_levels]；
                第 i 行是 item i 的完整 Semantic ID 或 Random ID。
            max_history: 单条样本最多保留最近多少个历史物品。
            last_only: False 时从一条序列产生多个滑窗样本，适合训练；
                True 时只预测最后一个物品，适合 leave-one-out 评估。
            max_samples_per_sequence: 可选。限制每个用户最多贡献多少条训练样本；
                超出时保留时间上最近的目标。KuaiRec 单用户行为很多，这个参数
                可以避免少数超长序列主导训练并控制训练时间。

        调用:
            scripts/run_demo.py 分别为训练集、测试集以及 Random ID 对照组调用。
        """
        # samples 里保存的是 (历史 item 列表, 目标 item)。
        self.samples = []
        # item_codes[index] 可以把一个 item 编号转换成它的 Semantic ID。
        self.item_codes = torch.as_tensor(item_codes, dtype=torch.long)
        # 最多保留多少个历史行为，太长的历史只截取最近 max_history 个。
        self.max_history = max_history
        for sequence in sequences:
            # 用滑动窗口构造“历史行为 -> 下一个物品”的训练样本。
            # last_only=True 时只取最后一个预测点，常用于 leave-one-out 评估。
            indices = [len(sequence) - 1] if last_only else list(range(2, len(sequence)))
            if (
                not last_only
                and max_samples_per_sequence is not None
                and len(indices) > max_samples_per_sequence
            ):
                indices = indices[-max_samples_per_sequence:]
            for index in indices:
                # index 左边是历史，sequence[index] 是这条样本要预测的下一个物品。
                self.samples.append((sequence[max(0, index - max_history) : index], sequence[index]))

    def __len__(self):
        """返回 next-item 样本总数，供 PyTorch DataLoader 决定迭代长度。"""
        return len(self.samples)

    def __getitem__(self, index):
        """取一条已经完成定长 padding 和编码转换的监督样本。

        参数:
            index: DataLoader 请求的样本下标，范围是 ``[0, len(self))``。

        返回:
            history_codes: LongTensor，形状 [max_history, num_levels]；
                左侧 padding 为 0，真实 token 整体加 1。
            target_codes: LongTensor，形状 [num_levels]；目标物品的原始编码，
                不加 1，因为它直接作为分类标签。
            target_item: 标量 LongTensor；目标物品的原始连续 item index，
                只在推荐指标计算时使用。
            history_items: LongTensor，形状 [max_history]；原始历史 item index，
                左侧 padding 为 -1。评估时用它排除已经看过的候选物品。

        调用:
            training/train.py 和 training/evaluate.py 中的 DataLoader 自动调用。
        """
        # 取出第 index 条样本的历史 item 和目标 item。
        history, target = self.samples[index]
        # 把历史 item_id 转换成历史 Semantic ID 序列。
        history_codes = self.item_codes[history]
        padding = self.max_history - len(history)
        # 原始 item index 也做定长左 padding，供评估阶段屏蔽已看物品。
        history_items = torch.full((self.max_history,), -1, dtype=torch.long)
        history_items[padding:] = torch.as_tensor(history, dtype=torch.long)
        # 左侧 padding 到固定长度，保证同一个 batch 的历史序列形状一致。
        if padding:
            # padding 位置全是 0；真实历史 token 加 1，避免和 padding 的 0 混淆。
            history_codes = torch.cat(
                [torch.zeros((padding, history_codes.shape[1]), dtype=torch.long), history_codes + 1]
            )
        else:
            # 真实历史 token 整体加 1，把 0 留给 padding。
            history_codes = history_codes + 1
        # 返回三部分：
        # history_codes 给模型做输入；target_codes 用来训练生成 Semantic ID；
        # target 原始 item_id 用来评估推荐列表是否命中真实物品。
        return (
            history_codes,
            self.item_codes[target],
            torch.tensor(target),
            history_items,
        )


class SASRecDataset(Dataset):
    """把相同的 next-item 样本转换为 SASRec 使用的 Item ID 序列。

    输入中的 0 表示 padding，真实 item index 整体加 1；监督标签仍使用从 0
    开始的 item index。样本切分规则与 NextItemDataset 完全相同。
    """

    def __init__(
        self,
        sequences: list[list[int]],
        max_history: int,
        last_only: bool = False,
        max_samples_per_sequence: int | None = None,
    ):
        self.samples = []
        self.max_history = max_history
        for sequence in sequences:
            indices = [len(sequence) - 1] if last_only else list(range(2, len(sequence)))
            if (
                not last_only
                and max_samples_per_sequence is not None
                and len(indices) > max_samples_per_sequence
            ):
                indices = indices[-max_samples_per_sequence:]
            for index in indices:
                history = sequence[max(0, index - max_history) : index]
                self.samples.append((history, sequence[index]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        history, target = self.samples[index]
        padding = self.max_history - len(history)
        history_tokens = torch.zeros(self.max_history, dtype=torch.long)
        history_tokens[padding:] = torch.as_tensor(history, dtype=torch.long) + 1
        history_items = torch.full((self.max_history,), -1, dtype=torch.long)
        history_items[padding:] = torch.as_tensor(history, dtype=torch.long)
        return history_tokens, torch.tensor(target), history_items
