"""读取外部真实交互与物品特征，并转换为项目统一的内部格式。

scripts/run_demo.py 仅在配置同时提供 interactions_path 和 item_features_path
时调用 load_interactions；否则会改用 data/synthetic.py 的合成数据。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_interactions(
    interactions_path: str | Path,
    item_features_path: str | Path,
    min_sequence_length: int = 3,
    max_sequence_length: int | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    读取真实交互数据和物品特征，并转换成项目内部使用的格式。

    interactions.csv 需要包含 user_id、item_id、timestamp 三列。
    item_features.npy 需要和排序后的唯一 item_id 一一对齐。

    参数:
        interactions_path: CSV 路径。每行是一条用户行为，至少包含
            user_id、item_id、timestamp。
        item_features_path: NumPy ``.npy`` 文件路径，形状应为
            [num_items, feature_dim]。
        min_sequence_length: 过滤阈值；行为少于该值的用户不会进入结果。
        max_sequence_length: 可选。每个用户只保留时间上最近的这些交互；
            KuaiRec 用户序列很长时可控制训练规模。

    返回:
        features: float32 物品特征矩阵，形状 [num_items, feature_dim]。
        sequences: 按用户组织、按时间升序排列的 item index 序列。

    调用:
        scripts/run_demo.py 的 main；内部调用 pandas.read_csv、groupby 和
        numpy.load。

    注意:
        函数会把任意形式的原始 item_id 映射成从 0 开始的连续整数。特征文件
        的行顺序必须与“排序后的唯一 item_id”一致，否则编码会错配。
    """
    # 读取用户交互表，每一行是一条用户对物品的行为记录。
    required = {"user_id", "item_id", "timestamp"}
    # 先只读取表头验证字段，再只加载真正需要的三列，避免 KuaiRec 的大 CSV
    # 把预处理统计列全部占入内存。
    columns = set(pd.read_csv(interactions_path, nrows=0).columns)
    # 检查必需列是否存在，避免后面排序和分组时报难懂的错误。
    missing = required - columns
    if missing:
        raise ValueError(f"Missing interaction columns: {sorted(missing)}")
    interactions = pd.read_csv(
        interactions_path,
        usecols=["user_id", "item_id", "timestamp"],
        dtype={"user_id": "int64", "item_id": "int64", "timestamp": "float64"},
    )
    # 把原始 item_id 排序后映射成从 0 开始的连续编号，方便做数组索引。
    item_ids = sorted(interactions["item_id"].unique())
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    sequences = []
    # 先按时间排序，再按用户分组，得到每个用户的行为序列。
    interactions = interactions.sort_values(["user_id", "timestamp"])
    for _, group in interactions.groupby("user_id", sort=False):
        sequence = [item_to_index[item] for item in group["item_id"]]
        if max_sequence_length is not None:
            sequence = sequence[-max_sequence_length:]
        # 太短的序列无法构造有效的“历史 -> 下一个物品”样本，因此过滤掉。
        if len(sequence) >= min_sequence_length:
            sequences.append(sequence)
    # 读取物品特征矩阵；每一行对应一个排序后的唯一 item。
    features = np.load(item_features_path).astype(np.float32)
    # 特征行数必须和 item 数一致，否则 Semantic ID 会和 item 对不上。
    if len(features) != len(item_ids):
        raise ValueError("Feature rows must match sorted unique item IDs")
    return features, sequences
