"""把连续物品特征转换为生成模型可以预测的离散 Semantic ID。

调用关系：
    scripts/run_demo.py
        ├─ build_hierarchical_codes：为 Semantic ID 实验构造语义编码
        ├─ append_collision_token：让完整编码唯一对应一个物品
        ├─ build_random_codes：构造不含语义的公平对照组
        └─ collision_rate：记录编码碰撞率

这一模块不训练推荐模型。它只负责建立 ``item_index -> token 序列`` 的目录，
例如把第 17 个物品映射成 ``[2, 5, 0]``。
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def build_hierarchical_codes(
    features: np.ndarray,
    codebook_sizes: list[int],
    seed: int,
    minibatch_threshold: int = 5000,
    method: str = "residual",
) -> np.ndarray:
    """
    使用残差量化或树式层次 KMeans 把连续物品特征转换成离散 Semantic ID。

    输入:
        features: 每个物品的连续特征向量，形状为 [num_items, feature_dim]。
        codebook_sizes: 每一层 KMeans 的聚类数量，例如 [8, 8] 表示两层 token。
        seed: 随机种子，用于保证 KMeans 结果可复现。
        minibatch_threshold: 物品数达到该阈值时改用 MiniBatchKMeans，降低
            KuaiRec 万级目录上残差量化的内存和时间。
        method: ``residual`` 保留旧 Demo 的全局残差量化；
            ``hierarchical`` 让每一级在上一层前缀内部继续聚类，适合 KuaiRec。

    输出:
        codes: 每个物品的层次编码，形状为 [num_items, num_levels]。

    调用:
        在 scripts/run_demo.py 的 main 中调用；内部调用 sklearn.cluster.KMeans
        的 fit_predict 完成每一级量化。

    原理:
        第一级聚类原始特征，得到粗粒度 token；第二级及以后聚类上一级没有
        表达掉的残差，继续细化编码。这是一个轻量版残差量化过程。
    """
    if method == "hierarchical":
        return _build_tree_codes(
            features, codebook_sizes, seed, minibatch_threshold
        )
    if method != "residual":
        raise ValueError(f"Unknown semantic ID method: {method}")

    # 旧实验协议：第一层量化原始特征，后续层量化未解释的全局残差。
    residual = features.copy()
    codes = []
    for level, size in enumerate(codebook_sizes):
        if len(features) >= minibatch_threshold:
            model = MiniBatchKMeans(
                n_clusters=size,
                random_state=seed + level,
                n_init=10,
                batch_size=min(2048, len(features)),
            )
        else:
            model = KMeans(n_clusters=size, random_state=seed + level, n_init=10)
        code = model.fit_predict(residual)
        # 用当前层的簇中心近似 residual，再把这部分从 residual 中扣掉。
        # 下一层 KMeans 会学习更细粒度的剩余差异。
        residual = residual - model.cluster_centers_[code]
        codes.append(code)
    # 输出形状为 [num_items, num_levels]，每一行就是一个 item 的 Semantic ID 前缀。
    return np.stack(codes, axis=1).astype(np.int64)


def _build_tree_codes(
    features: np.ndarray,
    codebook_sizes: list[int],
    seed: int,
    minibatch_threshold: int,
) -> np.ndarray:
    """在每个已有语义前缀内部继续聚类，构造层次化 Semantic ID。

    全局残差 KMeans 在高维稀疏文本特征上可能反复产生相似分桶，造成大量完整
    前缀碰撞。树式方法保证第 l 级是在相同前 l-1 级前缀的物品中做细分。

    当一个前缀下的物品数不超过当前词表大小时，直接给它们唯一的局部 token。
    这个 token 类似最末级局部门牌号，可以避免没有必要的 KMeans 空簇。
    """
    features = np.asarray(features, dtype=np.float32)
    codes = np.zeros((len(features), len(codebook_sizes)), dtype=np.int64)
    groups = [np.arange(len(features), dtype=np.int64)]

    for level, size in enumerate(codebook_sizes):
        next_groups = []
        for group_number, indices in enumerate(groups):
            if len(indices) <= size:
                # 在当前父前缀内唯一即可。不同父节点使用不同的确定性排列，
                # 避免每个小分组都从 token 0 开始而造成全局 token 极度偏斜。
                local_rng = np.random.default_rng(
                    seed + level * 100_003 + group_number
                )
                labels = local_rng.permutation(size)[: len(indices)].astype(np.int64)
            else:
                model_seed = seed + level * 100_003 + group_number
                if len(indices) >= minibatch_threshold:
                    model = MiniBatchKMeans(
                        n_clusters=size,
                        random_state=model_seed,
                        n_init=10,
                        batch_size=min(2048, len(indices)),
                    )
                else:
                    model = KMeans(
                        n_clusters=size,
                        random_state=model_seed,
                        n_init=10,
                    )
                labels = model.fit_predict(features[indices]).astype(np.int64)
            codes[indices, level] = labels
            for token in np.unique(labels):
                next_groups.append(indices[labels == token])
        groups = next_groups
    return codes


def collision_rate(codes: np.ndarray) -> float:
    """
    计算编码碰撞率，也就是有多少比例的物品没有拿到唯一 Semantic ID。

    如果两个或多个物品的整行 code 完全相同，它们就发生了碰撞。
    返回值越接近 0，说明编码越能唯一地区分商品目录中的物品。

    参数:
        codes: 物品编码矩阵，形状为 [num_items, num_levels]。

    返回:
        ``1 - 唯一编码数 / 物品数``。例如 4 个物品只有 3 个唯一编码，
        碰撞率就是 0.25。

    调用:
        scripts/run_demo.py 用它记录追加 tail token 前后的碰撞率；
        tests/test_semantic_ids.py 用它验证碰撞处理。
    """
    # 多个 item 可能被量化成完全相同的 code；这里统计这种碰撞占比。
    return 1.0 - len({tuple(row) for row in codes.tolist()}) / len(codes)


def append_collision_token(codes: np.ndarray) -> tuple[np.ndarray, int]:
    """
    给 Semantic ID 追加一层 tail token，消除多个物品共享同一编码前缀的问题。

    同一个前缀第一次出现追加 0，第二次出现追加 1，以此类推。
    这样模型仍然可以利用前缀语义，同时最终完整 code 能唯一对应到具体物品。

    返回:
        resolved_codes: 追加 tail token 后的完整编码。
        tail_size: tail token 这一层的词表大小。

    参数:
        codes: 尚未保证唯一的编码矩阵，形状为 [num_items, num_levels]。

    调用:
        scripts/run_demo.py 直接处理 Semantic ID；build_random_codes 也调用它
        处理随机编码，从而保证两个实验的完整编码都能唯一定位物品。
    """
    counts: dict[tuple[int, ...], int] = {}
    tails = []
    for row in codes.tolist():
        key = tuple(row)
        # 同一个 Semantic ID 第一次出现 tail=0，第二次出现 tail=1，以此类推。
        # 这样即使前缀碰撞，追加 tail 后也能唯一定位具体 item。
        tail = counts.get(key, 0)
        counts[key] = tail + 1
        tails.append(tail)
    tail_size = max(tails, default=0) + 1
    # 返回完整 catalog code，以及 tail token 这一层需要的词表大小。
    return np.column_stack([codes, np.asarray(tails, dtype=np.int64)]), tail_size


def build_random_codes(
    num_items: int, codebook_sizes: list[int], seed: int
) -> tuple[np.ndarray, list[int]]:
    """
    生成随机层次 ID，作为 Semantic ID 的对照实验基线。

    随机 ID 和 Semantic ID 使用相同的层数与词表规模，因此模型容量基本一致；
    区别在于随机 ID 不包含物品特征语义。两者指标差距可以用来观察语义编码
    是否真的带来了推荐效果提升。

    返回:
        codes: 追加 tail token 后的随机层次编码。
        codebook_sizes_with_tail: 包含 tail 层词表大小的完整词表配置。

    参数:
        num_items: 商品目录中的物品总数。
        codebook_sizes: 随机编码每一级的 token 取值范围。
        seed: NumPy 随机数种子，保证对照实验可复现。

    调用:
        scripts/run_demo.py 用它建立 Random ID baseline；内部调用
        append_collision_token 消除随机编码碰撞。
    """
    # 随机层次 ID 作为对照组：模型容量相同，但输出 token 不携带物品语义。
    rng = np.random.default_rng(seed)
    codes = np.column_stack(
        [rng.integers(0, size, size=num_items, dtype=np.int64) for size in codebook_sizes]
    )
    # 随机 ID 也可能碰撞，因此同样追加 tail token，保证对比公平。
    codes, tail_size = append_collision_token(codes)
    return codes, [*codebook_sizes, tail_size]


def codebook_diagnostics(codes: np.ndarray, codebook_sizes: list[int]) -> dict:
    """统计各级 token 利用率、熵以及完整前缀碰撞情况。

    这些诊断比只看最终 Recall 更能判断 Semantic ID 是否退化。若某一级只使用
    少量 token，或少数 token 承担绝大多数物品，理论组合容量就没有真正利用。
    """
    codes = np.asarray(codes)
    levels = []
    for level, vocabulary_size in enumerate(codebook_sizes):
        counts = np.bincount(codes[:, level], minlength=vocabulary_size)
        probabilities = counts[counts > 0] / counts.sum()
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        max_entropy = float(np.log2(vocabulary_size)) if vocabulary_size > 1 else 0.0
        levels.append(
            {
                "level": level,
                "vocabulary_size": vocabulary_size,
                "used_tokens": int(np.count_nonzero(counts)),
                "utilization": float(np.count_nonzero(counts) / vocabulary_size),
                "entropy_bits": entropy,
                "normalized_entropy": entropy / max_entropy if max_entropy else 1.0,
                "largest_bucket": int(counts.max()),
            }
        )
    unique, prefix_counts = np.unique(codes, axis=0, return_counts=True)
    return {
        "items": len(codes),
        "unique_codes": len(unique),
        "collision_rate": collision_rate(codes),
        "largest_collision_bucket": int(prefix_counts.max()),
        "levels": levels,
    }
