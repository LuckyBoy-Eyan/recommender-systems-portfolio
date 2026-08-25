"""实现四路会话商品召回。

全局召回结构（Popularity、ItemCF、Item2Vec）由历史事件构建；
Recent 召回只依赖当前 Session 的可见历史。所有函数默认输入已经满足 Point-in-Time
约束，本模块本身不负责过滤未来事件。
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import torch
from torch import nn
from torch.nn import functional as F


def build_popularity(events: pd.DataFrame) -> list[int]:
    """按照历史事件次数生成全局热门商品列表。

    参数：
        events:
            可见历史事件表，至少包含 ``aid``。

    返回：
        商品 ID 列表，按照事件出现次数从高到低排列。该列表不区分用户或 Session，
        主要用作冷启动和候选不足时的热门兜底。
    """
    return events["aid"].value_counts().index.tolist()


def build_itemcf(events: pd.DataFrame, max_neighbors: int = 100) -> dict[int, list[tuple[int, float]]]:
    """构建基于 Session 共现的 ItemCF 相似商品表。

    参数：
        events:
            可见历史事件表，包含 ``session`` 和 ``aid``。
        max_neighbors:
            每个商品最多保留的相似邻居数。

    返回：
        字典 ``{aid: [(neighbor_aid, similarity), ...]}``，相似度从高到低排列。

    计算方式：
        1. 同一 Session 内对商品去重；
        2. 使用 ``1 / log2(2 + session商品数)`` 降低超长 Session 的贡献；
        3. 用 ``sqrt(N_i * N_j)`` 做余弦形式归一化，缓解热门商品共现次数过高的问题。
    """
    co_counts: dict[int, Counter] = defaultdict(Counter)
    item_sessions = Counter()
    for _, group in events.groupby("session"):
        items = group["aid"].drop_duplicates().astype(int).tolist()
        for aid in items:
            item_sessions[aid] += 1
        # 长 Session 中任意两个商品偶然共现的概率更高，因此降低其贡献。
        weight = 1.0 / math.log2(2 + len(items))
        for left in items:
            for right in items:
                if left != right:
                    co_counts[left][right] += weight
    neighbors = {}
    for left, counter in co_counts.items():
        # N(i) 表示包含商品 i 的 Session 数。
        scored = [
            (right, score / math.sqrt(item_sessions[left] * item_sessions[right]))
            for right, score in counter.items()
        ]
        neighbors[left] = sorted(scored, key=lambda pair: pair[1], reverse=True)[:max_neighbors]
    return neighbors


def build_svd_neighbors(
    events: pd.DataFrame, max_neighbors: int = 100, dimensions: int = 32, seed: int = 2026
) -> dict[int, list[tuple[int, float]]]:
    """通过 Session-Item 矩阵低秩分解构建 SVD 消融邻居表。

    参数：
        events:
            可见历史事件表，包含 ``session、aid、type``。
        max_neighbors:
            每个商品最多保留的向量近邻数。
        dimensions:
            目标向量维度。若矩阵规模不足，会自动降低到合法维度。
        seed:
            ``TruncatedSVD`` 随机种子。

    返回：
        字典 ``{aid: [(neighbor_aid, cosine_similarity), ...]}``。

    实现说明：
        Session-Item 矩阵中点击、加购、购买分别累加 1、3、6；对矩阵转置执行
        ``TruncatedSVD`` 得到商品向量，L2 归一化后通过内积计算余弦相似度。
        当前实现直接生成完整商品两两相似度矩阵，并不是 ANN 检索。
    """
    # factorize 同时得到连续矩阵下标和下标到原始 ID 的映射。
    sessions, session_ids = pd.factorize(events["session"])
    items, item_ids = pd.factorize(events["aid"])
    matrix = np.zeros((len(session_ids), len(item_ids)), dtype=np.float32)
    type_weight = events["type"].map({"clicks": 1.0, "carts": 3.0, "orders": 6.0}).to_numpy()
    # 同一 Session 对同一商品的多次行为会在矩阵对应位置累加。
    np.add.at(matrix, (sessions, items), type_weight)
    components = min(dimensions, max(1, min(matrix.shape) - 1))
    embeddings = TruncatedSVD(components, random_state=seed).fit_transform(matrix.T)
    # 归一化后，向量内积等价于余弦相似度。
    embeddings = normalize(embeddings)
    similarities = embeddings @ embeddings.T
    output = {}
    for index, aid in enumerate(item_ids):
        order = np.argsort(-similarities[index])
        neighbors = [
            (int(item_ids[right]), float(similarities[index, right]))
            for right in order
            if right != index
        ][:max_neighbors]
        output[int(aid)] = neighbors
    return output


def _build_skipgram_pairs(
    events: pd.DataFrame,
    item_to_index: dict[int, int],
    window: int,
    action_weights: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从按时间排序的 Session 序列构造带行为权重的 Skip-Gram 正样本对。

    参数：
        events:
            仅包含当前快照可见历史的事件表。
        item_to_index:
            原始商品 ID 到连续词表下标的映射。
        window:
            中心商品左右两侧最多纳入的上下文距离。
        action_weights:
            点击、加购、购买对应的正数权重。

    返回：
        ``(centers, contexts, pair_weights)``。前两个是一维 LongTensor，
        每个位置表示一个有序正样本；第三个是一维 FloatTensor，权重为中心行为和
        上下文行为权重的几何平均数。
    """
    centers: list[int] = []
    contexts: list[int] = []
    pair_weights: list[float] = []
    ordered = events.sort_values(["session", "ts"], kind="stable")
    for _, group in ordered.groupby("session", sort=False):
        sequence = [
            (item_to_index[int(row.aid)], float(action_weights[row.type]))
            for row in group.itertuples(index=False)
            if int(row.aid) in item_to_index
        ]
        for position, (center, center_weight) in enumerate(sequence):
            left = max(0, position - window)
            right = min(len(sequence), position + window + 1)
            for context_position in range(left, right):
                if context_position != position:
                    context, context_weight = sequence[context_position]
                    centers.append(center)
                    contexts.append(context)
                    pair_weights.append(math.sqrt(center_weight * context_weight))
    return (
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(pair_weights, dtype=torch.float32),
    )


def _sample_negative_items(
    center_batch: torch.Tensor,
    negative_distribution: torch.Tensor,
    blocked_contexts: torch.Tensor,
    negative_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """确定性抽取不属于中心商品真实上下文集合的负商品。

    参数：
        center_batch:
            一维中心商品下标，形状为 ``[B]``。
        negative_distribution:
            商品频次0.75次方归一化后的采样概率，形状为 ``[V]``。
        blocked_contexts:
            布尔矩阵 ``[V, V]``。若商品 ``j`` 是中心商品 ``i`` 的任一已观察
            真实上下文，或 ``i == j``，则 ``blocked_contexts[i, j]=True``。
        negative_samples:
            每个中心商品需要抽取的负样本数量。
        generator:
            固定种子的本地 PyTorch 随机数生成器。

    返回：
        形状为 ``[B, negative_samples]`` 的负商品下标。函数使用拒绝采样，
        直到所有结果均不在对应中心商品的屏蔽集合中。
    """
    negatives = torch.multinomial(
        negative_distribution,
        len(center_batch) * negative_samples,
        replacement=True,
        generator=generator,
    ).view(len(center_batch), negative_samples)
    invalid = blocked_contexts[center_batch.unsqueeze(1), negatives]
    while invalid.any():
        replacement_count = int(invalid.sum().item())
        negatives[invalid] = torch.multinomial(
            negative_distribution,
            replacement_count,
            replacement=True,
            generator=generator,
        )
        invalid = blocked_contexts[center_batch.unsqueeze(1), negatives]
    return negatives


def build_item2vec_neighbors(
    events: pd.DataFrame,
    max_neighbors: int = 100,
    dimensions: int = 32,
    window: int = 5,
    negative_samples: int = 5,
    epochs: int = 5,
    batch_size: int = 2048,
    learning_rate: float = 0.025,
    min_count: int = 1,
    action_weights: dict[str, float] | None = None,
    exclude_positive_contexts: bool = True,
    seed: int = 2026,
) -> dict[int, list[tuple[int, float]]]:
    """使用 Skip-Gram Negative Sampling 训练 Item2Vec 商品近邻。

    参数：
        events:
            当前 Point-in-Time 快照严格可见的历史事件，包含
            ``session、aid、ts``。
        max_neighbors:
            每个商品最多保留的余弦近邻数量。
        dimensions:
            输入和输出商品向量的维度。
        window:
            Session 序列中中心商品左右两侧的上下文窗口。
        negative_samples:
            每个正样本配套抽取的负商品数量。
        epochs:
            对当前快照正样本对训练的轮数。
        batch_size:
            每次梯度更新包含的正样本对数量。
        learning_rate:
            Adam 优化器学习率。
        min_count:
            商品进入词表所需的最少历史出现次数。
        action_weights:
            行为权重字典。正式配置使用点击1、加购3、购买6；为 ``None`` 时三类行为
            权重均为1，用于复现旧版未加权实验。
        exclude_positive_contexts:
            是否禁止把中心商品自身及它在当前快照中全部已观察上下文抽成负样本。
        seed:
            PyTorch 初始化、乱序和负采样共同使用的随机种子。

    返回：
        字典 ``{aid: [(neighbor_aid, cosine_similarity), ...]}``。

    因果与确定性：
        函数不会自行读取其他数据；调用方必须传入严格早于快照时点的事件。
        实现使用固定种子的本地 ``torch.Generator`` 和单线程 CPU，同一输入及参数
        重复训练会得到相同的邻居和分数。

    训练目标：
        对正样本 ``(i, j)`` 最大化 ``log σ(v_i·u_j)``，并对从商品出现频率
        ``count(aid)^0.75`` 分布抽取的负样本最大化 ``log σ(-v_i·u_neg)``。
        每个正样本损失乘以中心行为和上下文行为权重的几何平均数，再统一归一化到
        均值1。若开启 ``exclude_positive_contexts``，负例不能是商品自身或该中心
        在当前快照中任一已观察真实上下文。
    """
    if events.empty:
        return {}
    if dimensions < 1 or window < 1 or negative_samples < 1 or epochs < 1:
        raise ValueError("Item2Vec dimensions/window/negative_samples/epochs 必须为正数")
    if batch_size < 1 or learning_rate <= 0 or min_count < 1:
        raise ValueError("Item2Vec batch_size/min_count 必须为正数，learning_rate 必须大于 0")
    action_weights = action_weights or {"clicks": 1.0, "carts": 1.0, "orders": 1.0}
    required_actions = {"clicks", "carts", "orders"}
    if set(action_weights) != required_actions or any(
        float(action_weights[action]) <= 0 for action in required_actions
    ):
        raise ValueError("action_weights 必须为 clicks/carts/orders 提供正数权重")

    counts = events["aid"].astype(int).value_counts()
    item_ids = np.array(sorted(int(aid) for aid, count in counts.items() if count >= min_count))
    if len(item_ids) < 2:
        return {}
    item_to_index = {int(aid): index for index, aid in enumerate(item_ids)}
    centers, contexts, pair_weights = _build_skipgram_pairs(
        events, item_to_index, window, action_weights
    )
    if centers.numel() == 0:
        return {}
    vocabulary_size = len(item_ids)
    blocked_contexts = torch.zeros(
        (vocabulary_size, vocabulary_size), dtype=torch.bool
    )
    if exclude_positive_contexts:
        blocked_contexts[centers, contexts] = True
        blocked_contexts.fill_diagonal_(True)
        # 极小词表中某个中心可能与全部商品都形成过正关系，此时不存在合法负例。
        # 删除这些中心对应的训练对，避免拒绝采样无法结束。
        has_allowed_negative = (~blocked_contexts).any(dim=1)
        usable_pairs = has_allowed_negative[centers]
        centers = centers[usable_pairs]
        contexts = contexts[usable_pairs]
        pair_weights = pair_weights[usable_pairs]
        if centers.numel() == 0:
            return {}
    # 把均值归一化到1，仅保留行为间相对重要性，不改变整体损失和学习率尺度。
    pair_weights = pair_weights / pair_weights.mean()

    # 限制为 CPU 单线程并固定所有随机源，确保不同运行之间的结果可复现。
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.manual_seed(seed)
    input_embeddings = nn.Embedding(vocabulary_size, dimensions)
    output_embeddings = nn.Embedding(vocabulary_size, dimensions)
    bound = 0.5 / dimensions
    with torch.no_grad():
        input_embeddings.weight.uniform_(-bound, bound, generator=generator)
        output_embeddings.weight.zero_()
    optimizer = torch.optim.Adam(
        list(input_embeddings.parameters()) + list(output_embeddings.parameters()),
        lr=learning_rate,
    )
    negative_distribution = torch.tensor(
        [float(counts.get(int(aid), 0)) ** 0.75 for aid in item_ids],
        dtype=torch.float32,
    )
    negative_distribution /= negative_distribution.sum()

    try:
        for _ in range(epochs):
            permutation = torch.randperm(len(centers), generator=generator)
            for start in range(0, len(centers), batch_size):
                batch_indices = permutation[start : start + batch_size]
                center_batch = centers[batch_indices]
                context_batch = contexts[batch_indices]
                weight_batch = pair_weights[batch_indices]
                if exclude_positive_contexts:
                    negative_batch = _sample_negative_items(
                        center_batch,
                        negative_distribution,
                        blocked_contexts,
                        negative_samples,
                        generator,
                    )
                else:
                    negative_batch = torch.multinomial(
                        negative_distribution,
                        len(batch_indices) * negative_samples,
                        replacement=True,
                        generator=generator,
                    ).view(len(batch_indices), negative_samples)

                center_vectors = input_embeddings(center_batch)
                context_vectors = output_embeddings(context_batch)
                positive_logits = (center_vectors * context_vectors).sum(dim=1)
                negative_vectors = output_embeddings(negative_batch)
                negative_logits = torch.bmm(
                    negative_vectors, center_vectors.unsqueeze(2)
                ).squeeze(2)
                per_pair_loss = -(
                    F.logsigmoid(positive_logits)
                    + F.logsigmoid(-negative_logits).sum(dim=1)
                )
                loss = (per_pair_loss * weight_batch).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    finally:
        torch.set_num_threads(previous_threads)

    embeddings = F.normalize(input_embeddings.weight.detach(), p=2, dim=1).cpu().numpy()
    similarities = embeddings @ embeddings.T
    output: dict[int, list[tuple[int, float]]] = {}
    for index, aid in enumerate(item_ids):
        # aid 作为相似度并列时的稳定次序，保证输出完全可复现。
        order = np.lexsort((item_ids, -similarities[index]))
        output[int(aid)] = [
            (int(item_ids[right]), float(similarities[index, right]))
            for right in order
            if right != index
        ][:max_neighbors]
    return output


def build_embedding_neighbors(
    events: pd.DataFrame,
    method: str = "item2vec",
    max_neighbors: int = 100,
    dimensions: int = 32,
    seed: int = 2026,
    **item2vec_parameters,
) -> dict[int, list[tuple[int, float]]]:
    """按配置构建 Item2Vec、SVD 或关闭向量召回。

    参数：
        events:
            当前 Point-in-Time 快照可见的历史事件。
        method:
            ``item2vec`` 为正式方案，``svd`` 用于旧方案消融，``none`` 表示关闭。
        max_neighbors、dimensions、seed:
            两种向量方案共用的近邻数、维度和随机种子。
        **item2vec_parameters:
            仅传给 ``build_item2vec_neighbors`` 的训练参数。
    """
    if method == "none":
        return {}
    if method == "svd":
        return build_svd_neighbors(events, max_neighbors, dimensions, seed)
    if method == "item2vec":
        return build_item2vec_neighbors(
            events,
            max_neighbors=max_neighbors,
            dimensions=dimensions,
            seed=seed,
            **item2vec_parameters,
        )
    raise ValueError(f"不支持的 embedding method: {method}")


def recall_candidates(
    history: pd.DataFrame,
    popularity: list[int],
    itemcf: dict[int, list[tuple[int, float]]],
    per_source: int,
    embedding: dict[int, list[tuple[int, float]]] | None = None,
    embedding_source: str = "item2vec",
) -> pd.DataFrame:
    """根据当前 Session 历史执行四路召回。

    参数：
        history:
            当前批次的可见 Session 历史，包含 ``session、aid、ts``。
        popularity:
            ``build_popularity`` 生成的热门商品有序列表。
        itemcf:
            ``build_itemcf`` 生成的 ItemCF 邻居表。
        per_source:
            每一路、每个 Session 最多返回的候选数。
        embedding:
            可选的向量近邻表。为 ``None`` 时不执行向量召回；
            为空字典时仍会产生该召回源，但通常没有候选。
        embedding_source:
            向量召回在候选长表中的来源名称，例如 ``item2vec`` 或 ``svd``。

    返回：
        长表 ``DataFrame``，列为：

        - ``session``：候选所属 Session；
        - ``aid``：候选商品；
        - ``source``：召回来源；
        - ``source_rank``：该来源内部排名，从 1 开始；
        - ``source_score``：该来源内部原始分数。

        同一 ``(session, aid)`` 可能因多路命中而出现多行，后续特征模块会合并并保留
        各路分数。理论最大记录数为 ``Session数 × 路数 × per_source``。
    """
    rows = []
    for session, group in history.sort_values("ts").groupby("session"):
        # 倒序读取并保持第一次出现，从而得到按新近性排列的不同商品。
        recent = list(dict.fromkeys(group["aid"].tolist()[::-1]))
        sources = {
            "recent": [(aid, 1.0 / (rank + 1)) for rank, aid in enumerate(recent[:per_source])],
            "popular": [(aid, 1.0 / (rank + 1)) for rank, aid in enumerate(popularity[:per_source])],
        }
        # ItemCF、Item2Vec/SVD 都只使用最近 5 个商品作为查询种子。
        itemcf_scores = Counter()
        for recency, aid in enumerate(recent[:5]):
            for neighbor, score in itemcf.get(int(aid), []):
                # 越新的种子贡献越大；同一邻居被多个种子命中时分数累加。
                itemcf_scores[neighbor] += score / (recency + 1)
        sources["itemcf"] = itemcf_scores.most_common(per_source)
        if embedding is not None:
            embedding_scores = Counter()
            for recency, aid in enumerate(recent[:5]):
                for neighbor, score in embedding.get(int(aid), []):
                    embedding_scores[neighbor] += score / (recency + 1)
            sources[embedding_source] = embedding_scores.most_common(per_source)
        for source, candidates in sources.items():
            for rank, (aid, score) in enumerate(candidates, start=1):
                rows.append((int(session), int(aid), source, rank, float(score)))
    return pd.DataFrame(rows, columns=["session", "aid", "source", "source_rank", "source_score"])
