"""端到端关键正确性测试。

测试重点不是模型效果，而是标签唯一性、时间切分、Point-in-Time 因果边界、
召回图防泄漏以及负采样行为。
"""

import pandas as pd
import torch

from src.data.split import drop_ambiguous_target_sessions, leave_last_event_out, split_sessions
from src.features.point_in_time import build_point_in_time_dataset
from src.evaluation.metrics import candidate_recall_by_action
from src.ranking.model import heuristic_score, sample_hard_negatives
from src.recall.sources import (
    _build_skipgram_pairs,
    _sample_negative_items,
    build_item2vec_neighbors,
    build_itemcf,
)
from scripts.prepare_retailrocket import prepare_events
from scripts.run_pipeline import (
    configured_rankers,
    select_formal_and_warmup_events,
    validate_data_profile,
)


def _make_boundary_test_events(num_sessions: int = 10) -> pd.DataFrame:
    """构造仅供单元测试检查时间边界的最小事件表。

    参数：
        num_sessions:
            需要构造的 Session 数量。每个 Session 固定包含 4 条行为。

    返回：
        包含 ``session``、``aid``、``ts`` 和 ``type`` 四列的事件表。

    说明：
        这不是实验数据生成器，不参与模型训练或指标报告；它只让切分测试拥有
        可手工验证的确定输入。
    """
    rows = []
    action_types = ("clicks", "clicks", "carts", "orders")
    for session in range(num_sessions):
        session_start = session * 100
        for position, action_type in enumerate(action_types):
            rows.append(
                (session, session * 10 + position, session_start + position, action_type)
            )
    return pd.DataFrame(rows, columns=["session", "aid", "ts", "type"])


def test_leave_last_event_out():
    """每个无歧义 Session 应产生一个标签，且事件总数守恒。"""
    events = _make_boundary_test_events()
    history, labels = leave_last_event_out(events)
    assert len(labels) == 10
    assert len(history) + len(labels) == len(events)
    assert "target_ts" in labels


def test_ambiguous_target_timestamp_drops_entire_session():
    """最大时间戳并列时应删除整个 Session，而不是任意选择一行。"""
    events = pd.DataFrame(
        [
            (0, 10, 1, "clicks"),
            (0, 11, 2, "carts"),
            (0, 12, 2, "orders"),
            (1, 20, 3, "clicks"),
            (1, 21, 4, "orders"),
        ],
        columns=["session", "aid", "ts", "type"],
    )
    cleaned = drop_ambiguous_target_sessions(events)
    history, labels = leave_last_event_out(events)
    assert set(cleaned["session"]) == {1}
    assert set(history["session"]) == {1}
    assert set(labels["session"]) == {1}


def test_sessionization_happens_before_catalog_filtering():
    """商品目录过滤不能改变原始行为流上的 Session 边界。"""
    minute = 60 * 1000
    raw = pd.DataFrame(
        [
            (0, 0, "view", 1),
            (20 * minute, 0, "view", 999),
            (40 * minute, 0, "view", 2),
            (0, 1, "view", 1),
            (minute, 1, "view", 1),
            (2 * minute, 1, "view", 2),
        ],
        columns=["timestamp", "visitorid", "event", "itemid"],
    )
    prepared, metadata = prepare_events(
        raw, catalog_size=2, max_sessions=10, min_session_length=2
    )
    item_sets = prepared.groupby("session")["aid"].apply(set).tolist()
    assert metadata["sessionization_before_catalog_filter"] is True
    assert metadata["sessions"] == 2
    assert {1, 2} in item_sets
    assert 999 not in set(prepared["aid"])


def test_retailrocket_defaults_use_current_data_scale():
    """默认预处理口径必须保持 Top3000 和 20000 个 Session。"""
    defaults = prepare_events.__defaults__
    assert defaults == (3000, 20000, 3)


def test_data_profile_rejects_mismatched_preprocessed_data(tmp_path):
    """主配置不能静默读取与声明口径不一致的数据。"""
    events = pd.DataFrame(
        [(0, 1, 1, "clicks"), (0, 2, 2, "orders")],
        columns=["session", "aid", "ts", "type"],
    )
    data_path = tmp_path / "events.csv"
    data_path.with_suffix(".metadata.json").write_text(
        '{"sessions": 19999, "catalog_size_limit": 2999, '
        '"max_sessions": 19999, "min_session_length": 3}'
    )
    try:
        validate_data_profile(
            data_path,
            events,
            {
                "sessions": 20000,
                "catalog_size_limit": 3000,
                "max_sessions": 20000,
                "min_session_length": 3,
            },
        )
    except ValueError as error:
        assert "数据口径与配置不一致" in str(error)
    else:
        raise AssertionError("stale data profile should be rejected")


def test_warmup_ablation_keeps_formal_labels_fixed_and_uses_complete_sessions():
    """0/3/7/14天组只能改变历史池，正式标签Session必须完全一致。"""
    day = 100
    rows = []
    # 每个Session完整落在单独一天内；最后20个Session位于正式标签起点之后。
    for session in range(40):
        start = session * day + 10
        rows.extend(
            [
                (session, session * 10, start, "clicks"),
                (session, session * 10 + 1, start + 1, "orders"),
            ]
        )
    events = pd.DataFrame(rows, columns=["session", "aid", "ts", "type"])
    label_start = 20 * day
    formal_sets = []
    warmup_sizes = []
    for warmup_days in (0, 3, 7, 14):
        formal, warmup, profile = select_formal_and_warmup_events(
            events,
            label_start,
            warmup_days,
            day,
            max_formal_sessions=20,
        )
        formal_sets.append(set(formal["session"]))
        warmup_sizes.append(warmup["session"].nunique())
        assert set(formal["session"]).isdisjoint(set(warmup["session"]))
        if not warmup.empty:
            assert warmup["ts"].min() >= profile["warmup_start_ts"]
            assert warmup["ts"].max() < label_start
    assert all(formal_set == formal_sets[0] for formal_set in formal_sets)
    assert warmup_sizes == [0, 3, 7, 14]


def test_formal_configuration_uses_only_shared_bottom():
    """正式实验只训练Shared-Bottom排序器。"""
    rankers, primary = configured_rankers(
        {
            "primary_ranker": "shared_bottom",
            "rankers": {
                "shared_bottom": {"method": "shared_bottom", "epochs": 2},
            },
        }
    )
    assert primary == "shared_bottom"
    assert set(rankers) == {"shared_bottom"}
    assert rankers["shared_bottom"]["method"] == "shared_bottom"


def test_weighted_candidate_recall_matches_task_weights():
    """候选上限必须使用与最终Weighted Recall相同的任务权重。"""
    recalled = pd.DataFrame(
        [(0, 10), (1, 20), (2, 99)],
        columns=["session", "aid"],
    )
    labels = pd.DataFrame(
        [
            (0, 10, "clicks"),
            (1, 20, "carts"),
            (2, 30, "orders"),
        ],
        columns=["session", "target_aid", "target_type"],
    )
    metrics = candidate_recall_by_action(recalled, labels)
    assert metrics["candidate_recall_clicks"] == 1.0
    assert metrics["candidate_recall_carts"] == 1.0
    assert metrics["candidate_recall_orders"] == 0.0
    assert metrics["weighted_candidate_recall"] == 0.4


def test_split_sessions_has_no_overlap():
    """三个集合的 Session 必须互斥，目标时间必须严格前后有序。"""
    events = _make_boundary_test_events()
    train, valid, test = split_sessions(events, 0.7, 0.15)
    assert set(train["session"]).isdisjoint(set(valid["session"]))
    assert set(train["session"]).isdisjoint(set(test["session"]))
    assert set(valid["session"]).isdisjoint(set(test["session"]))
    assert train["session"].nunique() == 7
    assert valid["session"].nunique() == 1
    assert test["session"].nunique() == 2
    assert train.groupby("session")["ts"].max().max() < valid.groupby("session")["ts"].max().min()
    assert valid.groupby("session")["ts"].max().max() < test.groupby("session")["ts"].max().min()


def test_point_in_time_snapshot_excludes_future_events():
    """快照不得使用目标时间之后的商品 999，并应记录正确最大参考时间。"""
    events = pd.DataFrame(
        [
            (0, 1, 10, "clicks"),
            (0, 2, 20, "orders"),
            (1, 3, 110, "clicks"),
            (1, 4, 120, "orders"),
            (2, 999, 210, "clicks"),
            (2, 998, 220, "orders"),
        ],
        columns=["session", "aid", "ts", "type"],
    )
    history, labels = leave_last_event_out(events[events["session"] == 1])
    _, recalled, audit = build_point_in_time_dataset(
        events, history, labels, snapshot_interval=100, candidates_per_source=10, seed=0
    )
    assert audit.loc[0, "max_reference_ts"] == 20
    assert audit.loc[0, "max_reference_ts"] < audit.loc[0, "min_target_ts"]
    assert 999 not in set(recalled["aid"])


def test_target_is_not_used_in_recall_graph():
    """Leave-Last-Event-Out 的目标商品不能进入 ItemCF 或 Item2Vec 图。"""
    events = pd.DataFrame(
        [
            (0, 10, 1, "clicks"),
            (0, 11, 2, "clicks"),
            (0, 99, 3, "orders"),
            (1, 20, 4, "clicks"),
            (1, 21, 5, "carts"),
            (1, 98, 6, "orders"),
        ],
        columns=["session", "aid", "ts", "type"],
    )
    history, _ = leave_last_event_out(events)
    graph = build_itemcf(history)
    all_graph_items = set(graph)
    all_graph_items.update(neighbor for values in graph.values() for neighbor, _ in values)
    assert 99 not in all_graph_items
    assert 98 not in all_graph_items
    item2vec = build_item2vec_neighbors(
        history,
        dimensions=4,
        window=1,
        negative_samples=2,
        epochs=1,
        batch_size=4,
        seed=0,
    )
    all_item2vec_items = set(item2vec)
    all_item2vec_items.update(
        neighbor for values in item2vec.values() for neighbor, _ in values
    )
    assert 99 not in all_item2vec_items
    assert 98 not in all_item2vec_items


def test_hard_negative_sampling_keeps_positive_and_caps_negatives():
    """负采样必须保留全部正例，并把每个 Session 的负例限制到上限。"""
    labeled = pd.DataFrame(
        {"session": [0] * 6, "aid": range(6), "label": [1, 0, 0, 0, 0, 0]}
    )
    sampled = sample_hard_negatives(labeled, 2, 0)
    assert sampled["label"].sum() == 1
    assert len(sampled) == 3


def test_heuristic_rrf_ignores_incompatible_raw_score_scales():
    """启发式融合应依据来源内部排名，而不是直接相加不可比的原始分数。"""
    features = pd.DataFrame(
        {
            "session": [0, 0],
            "aid": [10, 20],
            "source_rank_recent": [1, 2],
            "source_score_recent": [0.01, 10_000.0],
        }
    )
    ranked = heuristic_score(features)
    assert ranked["aid"].tolist() == [10, 20]


def test_item2vec_is_deterministic_for_same_snapshot_and_seed():
    """相同快照和随机种子重复训练应得到完全一致的 Item2Vec 邻居。"""
    events = pd.DataFrame(
        [
            (0, 10, 1, "clicks"),
            (0, 11, 2, "clicks"),
            (0, 12, 3, "orders"),
            (1, 10, 4, "clicks"),
            (1, 11, 5, "carts"),
            (1, 13, 6, "orders"),
            (2, 12, 7, "clicks"),
            (2, 11, 8, "clicks"),
            (2, 13, 9, "orders"),
        ],
        columns=["session", "aid", "ts", "type"],
    )
    parameters = {
        "max_neighbors": 3,
        "dimensions": 8,
        "window": 2,
        "negative_samples": 2,
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 0.02,
        "seed": 7,
    }
    first = build_item2vec_neighbors(events, **parameters)
    second = build_item2vec_neighbors(events, **parameters)
    assert first == second


def test_skipgram_pairs_use_click_cart_order_weights():
    """正样本权重应同时反映中心行为和上下文行为的重要性。"""
    events = pd.DataFrame(
        [(0, 10, 1, "clicks"), (0, 11, 2, "orders")],
        columns=["session", "aid", "ts", "type"],
    )
    centers, contexts, weights = _build_skipgram_pairs(
        events,
        {10: 0, 11: 1},
        window=1,
        action_weights={"clicks": 1.0, "carts": 3.0, "orders": 6.0},
    )
    assert centers.tolist() == [0, 1]
    assert contexts.tolist() == [1, 0]
    assert torch.allclose(weights, torch.full((2,), 6.0**0.5))


def test_negative_sampling_excludes_all_observed_contexts_and_self():
    """每个中心商品的负例不得属于其全部真实上下文集合或商品自身。"""
    blocked = torch.zeros((5, 5), dtype=torch.bool)
    blocked[0, [0, 1, 2]] = True
    blocked[1, [1, 3]] = True
    centers = torch.tensor([0] * 50 + [1] * 50)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9)
    negatives = _sample_negative_items(
        centers,
        torch.full((5,), 0.2),
        blocked,
        negative_samples=5,
        generator=generator,
    )
    assert not blocked[centers.unsqueeze(1), negatives].any()
