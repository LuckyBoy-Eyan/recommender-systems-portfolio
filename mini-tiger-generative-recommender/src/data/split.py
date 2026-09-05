"""推荐序列的时间切分工具。

KuaiRec 是带时间戳的行为数据。序列推荐应使用用户过去预测未来，因此不能再把
前 80% 用户用于训练、后 20% 用户用于测试。这里采用 leave-last-two-out：
每个用户最后一次交互做测试，倒数第二次做验证，更早的交互用于训练。
"""

from __future__ import annotations


def temporal_leave_two_out(
    sequences: list[list[int]],
    min_train_length: int = 3,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """按用户时间顺序构造训练、验证和测试序列。

    参数:
        sequences: 已按时间升序排列的用户 item index 序列。
        min_train_length: 去掉验证与测试目标后，训练前缀至少保留的长度。

    返回:
        train_sequences: 每条原序列去掉最后两个物品。
        validation_sequences: 每条原序列只去掉最后一个物品；配合
            NextItemDataset(last_only=True) 时，目标是倒数第二个物品。
        test_sequences: 完整序列；配合 last_only=True 时，目标是最后一个物品。

    例如:
        ``[A, B, C, D, E]`` 会变成训练 ``[A, B, C]``、
        验证 ``[A, B, C, D] -> D``、测试 ``[A, B, C, D, E] -> E``。
    """
    required = min_train_length + 2
    retained = [sequence for sequence in sequences if len(sequence) >= required]
    return (
        [sequence[:-2] for sequence in retained],
        [sequence[:-1] for sequence in retained],
        retained,
    )


def user_holdout_split(
    sequences: list[list[int]],
    train_fraction: float = 0.8,
) -> tuple[list[list[int]], list[list[int]]]:
    """保留旧 Demo 使用的按用户切分方式，便于复现实验结果。

    这不是 KuaiRec 的主评估协议。它仅用于兼容已有合成数据与 MovieLens 指标。
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split = int(len(sequences) * train_fraction)
    return sequences[:split], sequences[split:]
