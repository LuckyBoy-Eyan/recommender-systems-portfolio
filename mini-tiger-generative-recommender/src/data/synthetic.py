"""生成带有可学习兴趣结构的合成推荐数据。

该数据仅用于快速验证端到端代码是否工作，不用于宣称真实推荐效果。
scripts/run_demo.py 在配置中没有真实数据路径时调用 make_synthetic_data。
"""

from __future__ import annotations

import numpy as np


def make_synthetic_data(
    num_users: int,
    num_items: int,
    num_topics: int,
    feature_dim: int,
    min_seq_len: int,
    max_seq_len: int,
    seed: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    生成一份可控的合成推荐数据，用来快速验证完整训练和评估链路。

    思路是先造出若干个隐藏 topic，再让每个物品属于某个 topic；
    每个用户偏好两个 topic，所以用户序列里会反复出现相关 topic 的物品。

    参数:
        num_users: 要生成的用户数量，也就是 sequences 的长度。
        num_items: 商品目录大小，也就是 item_features 的行数。
        num_topics: 隐藏主题数量；主题同时影响物品特征和用户点击。
        feature_dim: 每个物品连续特征向量的维度。
        min_seq_len: 单个用户行为序列的最短长度。
        max_seq_len: 单个用户行为序列的最长长度，包含上界。
        seed: NumPy 随机种子。

    返回:
        item_features: float32 数组，形状 [num_items, feature_dim]。
        sequences: 用户行为列表，每条序列由从 0 开始的 item index 组成。

    调用:
        scripts/run_demo.py 的 main；内部只调用 NumPy 随机采样函数。
    """
    rng = np.random.default_rng(seed)
    # 每个 topic 有一个中心向量，代表一类物品的共同语义。
    topic_centers = rng.normal(size=(num_topics, feature_dim)).astype(np.float32)
    # 给每个物品随机分配一个隐藏 topic。
    item_topics = rng.integers(0, num_topics, size=num_items)
    # 物品特征 = 所属 topic 中心 + 少量噪声，让同 topic 物品相似但不完全一样。
    item_features = topic_centers[item_topics] + 0.15 * rng.normal(
        size=(num_items, feature_dim)
    )
    sequences: list[list[int]] = []
    for _ in range(num_users):
        # 每个用户偏好两个 topic，第一个 topic 权重更高。
        preferred = rng.choice(num_topics, size=2, replace=False)
        length = int(rng.integers(min_seq_len, max_seq_len + 1))
        sequence = []
        for _ in range(length):
            # 按用户偏好采样一个 topic，再从该 topic 的物品中随机选一个。
            topic = int(rng.choice(preferred, p=[0.75, 0.25]))
            candidates = np.flatnonzero(item_topics == topic)
            sequence.append(int(rng.choice(candidates)))
        sequences.append(sequence)
    # 返回物品特征矩阵和用户行为序列列表。
    return item_features.astype(np.float32), sequences
