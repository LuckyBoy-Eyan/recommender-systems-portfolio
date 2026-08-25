"""训练和调用Shared-Bottom及其神经排序消融。"""

from __future__ import annotations

import pandas as pd
from src.features.build import RRF_K
from src.ranking.neural import score_neural_candidates, train_neural_ranker


# 这些列是标识、标签或审计字段，不能作为模型输入，否则会造成无意义学习或目标泄漏。
NON_FEATURES = {"session", "aid", "label", "target_type", "target_ts", "snapshot_ts"}

def attach_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """把 Session 级真实目标绑定到候选特征表。

    参数：
        features:
            每个 ``(session, aid)`` 一行的候选特征表。
        labels:
            每个 Session 一行的标签表，至少包含 ``session、target_aid、
            target_type、target_ts``。

    返回：
        带监督标签的候选表。若候选 ``aid == target_aid``，则 ``label=1``；
        否则 ``label=0``。辅助比较列 ``target_aid`` 在计算后删除。

    注意：
        如果某个 Session 的真实目标没有被召回，该 Session 不会产生正候选，
        排序模型无法在后续阶段把它找回来。
    """
    labeled = features.merge(labels, on="session", how="left")
    labeled["label"] = (labeled["aid"] == labeled["target_aid"]).astype(int)
    return labeled.drop(columns=["target_aid"])


def sample_hard_negatives(labeled: pd.DataFrame, max_negatives: int, seed: int) -> pd.DataFrame:
    """按 Session 保留全部正例并限制召回负例数量。

    参数：
        labeled:
            ``attach_labels`` 生成的候选训练表，必须包含 ``session`` 和 ``label``。
        max_negatives:
            每个 Session 最多保留的负候选数。
        seed:
            Pandas 随机抽样种子，保证负样本选择可复现。

    返回：
        采样后的训练表。所有 ``label=1`` 的召回正例都会保留；若负例超过上限，
        则随机保留 ``max_negatives`` 个。

    “Hard” 的含义：
        这些负例不是从全商品库均匀随机抽取，而是已经被某一路召回认为相关的候选，
        因此通常比完全随机商品更难区分；当前实现并没有再按模型分数选择最难负例。
    """
    groups = []
    for _, group in labeled.groupby("session", sort=False):
        positive = group[group["label"] == 1]
        negative = group[group["label"] == 0]
        if len(negative) > max_negatives:
            negative = negative.sample(max_negatives, random_state=seed)
        groups.append(pd.concat([positive, negative]))
    return pd.concat(groups, ignore_index=True)


def train_ranker_system(
    labeled: pd.DataFrame, seed: int, config: dict | None = None
) -> dict:
    """按照配置训练Shared-Bottom。

    参数：
        labeled:
            Hard Negative采样后的训练候选表。
        seed:
            模型初始化和批次乱序使用的随机种子。
        config:
            正式方案固定为 ``shared_bottom``。
    """
    config = dict(config or {})
    method = config.pop("method", "shared_bottom")
    feature_columns = [
        column for column in labeled.columns if column not in NON_FEATURES
    ]
    hidden_dims = tuple(config.pop("hidden_dims", [64, 32]))
    bundle = train_neural_ranker(
        labeled,
        feature_columns,
        method=method,
        seed=seed,
        hidden_dims=hidden_dims,
        **config,
    )
    return {"method": method, "bundle": bundle}


def score_ranker_system(
    system: dict, features: pd.DataFrame, action: str
) -> pd.DataFrame:
    """使用统一接口调用神经多任务排序模型。"""
    return score_neural_candidates(system["bundle"], features, action)


def ranker_system_audit(system: dict) -> dict:
    """返回排序器类型、神经参数量、任务观测行数和正例权重等审计信息。"""
    bundle = system["bundle"]
    audit = {
        "method": bundle.method,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in bundle.model.parameters()
            if parameter.requires_grad
        ),
        "observed_rows": {
            action: rows
            for action, rows in zip(("clicks", "carts", "orders"), bundle.observed_rows)
        },
        "positive_weights": {
            action: weight
            for action, weight in zip(
                ("clicks", "carts", "orders"), bundle.positive_weights
            )
        },
        "positive_rows": {
            action: rows
            for action, rows in zip(
                ("clicks", "carts", "orders"), bundle.positive_rows
            )
        },
    }
    return audit


def heuristic_score(features: pd.DataFrame) -> pd.DataFrame:
    """使用 Reciprocal Rank Fusion 计算不训练模型的启发式基线。

    参数：
        features:
            候选特征表，包含以 ``source_rank_`` 开头的各路召回内部排名列。
            排名从 1 开始，0 表示该候选没有被对应召回源召回。

    返回：
        ``session、aid、score`` 三列表。融合分数公式为：

        ``score(session, aid) = Σ 1 / (RRF_K + source_rank)``

        未命中该候选的召回源贡献 0。所有召回源在统一的排名尺度上参与融合，
        因而某一路原始分数数值较大时不会支配最终排序。

    注意：
        当前各路权重相同。这解决了量纲不一致，但不代表权重最优；若要学习不同
        召回源的重要性，应只在验证集调权，不能根据测试集结果选择权重。
    """
    rank_columns = sorted(
        column for column in features if column.startswith("source_rank_")
    )
    if not rank_columns:
        raise ValueError("启发式 RRF 基线至少需要一个 source_rank_ 特征")

    ranks = features[rank_columns].astype(float)
    valid_ranks = ranks.where(ranks > 0)
    scored = features[["session", "aid"]].copy()
    scored["score"] = (1.0 / (RRF_K + valid_ranks)).fillna(0.0).sum(axis=1)
    # aid 只用于同分时提供确定性顺序，不参与分数计算。
    return scored.sort_values(
        ["session", "score", "aid"], ascending=[True, False, True]
    )
