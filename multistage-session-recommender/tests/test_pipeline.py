"""端到端关键正确性测试。

测试重点不是模型效果，而是标签唯一性、时间切分、Point-in-Time 因果边界、
召回图防泄漏以及负采样行为。
"""

import numpy as np
import pandas as pd
import torch

from src.data.split import drop_ambiguous_target_sessions, leave_last_event_out, split_sessions
from src.features.point_in_time import build_point_in_time_dataset
from src.evaluation.metrics import candidate_recall_by_action
from src.ranking.model import heuristic_score, sample_hard_negatives
from src.recall.sources import (
    _build_skipgram_pairs,
    _exact_topk_cosine_neighbors,
    _sample_negative_items,
    build_item2vec_neighbors,
    build_itemcf,
)
from scripts.prepare_retailrocket import prepare_events
from scripts.preprocess_full_retailrocket import preprocess_events
from scripts.preprocess_item_metadata import build_category_paths, clean_state_changes
from scripts.build_sequence_samples import (
    build_evaluation_samples,
    build_training_samples,
)
from scripts.build_asof_item_features import enrich_asof
from src.evaluation.recall_diagnostics import conditional_auc_gauc, route_diagnostics
from src.recall.full_catalog import build_directional_transitions
from src.recall.item2vec_ann import Item2VecANN, Item2VecEmbeddings, train_item2vec_embeddings
from src.recall.two_tower import causal_inbatch_mask
from src.ranking.neural import MMoE, PLE
from scripts.run_pipeline import (
    configured_rankers,
    select_formal_and_warmup_events,
    validate_data_profile,
)
from scripts.build_rolling_oof_candidates import compact_candidates, temporal_folds


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


def test_full_catalog_preparation_keeps_long_tail_items():
    """全量模式不得因商品频次删除长尾事件。"""
    minute = 60 * 1000
    raw = pd.DataFrame(
        [
            (0, 0, "view", 1),
            (minute, 0, "view", 2),
            (2 * minute, 0, "transaction", 999),
            (0, 1, "view", 1),
            (minute, 1, "view", 2),
            (2 * minute, 1, "transaction", 3),
        ],
        columns=["timestamp", "visitorid", "event", "itemid"],
    )
    prepared, metadata = prepare_events(
        raw, catalog_size=None, max_sessions=10, min_session_length=3
    )
    assert set(prepared["aid"]) == {1, 2, 3, 999}
    assert metadata["catalog_policy"] == "full"
    assert metadata["catalog_size_limit"] is None


def test_full_preprocessing_keeps_reference_only_sessions():
    """短 Session 和末时间并列 Session 保留作参考，但不产生标签。"""
    raw = pd.DataFrame(
        [
            (0, 1, "view", 10, np.nan),
            (1, 1, "view", 11, np.nan),
            (10, 2, "view", 20, np.nan),
            (11, 2, "view", 21, np.nan),
            (11, 2, "addtocart", 22, np.nan),
            (20, 3, "view", 30, np.nan),
            (21, 3, "view", 31, np.nan),
            (22, 3, "transaction", 32, 7),
            (30, 4, "view", 40, np.nan),
            (31, 4, "view", 41, np.nan),
            (32, 4, "transaction", 42, 8),
            (40, 5, "view", 50, np.nan),
            (41, 5, "view", 51, np.nan),
            (42, 5, "transaction", 52, 9),
        ],
        columns=["timestamp", "visitorid", "event", "itemid", "transactionid"],
    )
    events, sessions, labels, report = preprocess_events(
        raw, train_ratio=0.34, valid_ratio=0.34
    )
    assert len(events) == len(raw)
    assert len(labels) == 3
    reasons = set(sessions.loc[~sessions["label_eligible"], "ineligible_reason"])
    assert reasons == {"too_short", "ambiguous_last_timestamp"}
    assert report["session_lengths"]["reference_only_sessions"] == 2


def test_item_state_cleaning_removes_conflicts_and_repeated_states():
    frame = pd.DataFrame(
        [(1, 10, 2), (1, 10, 2), (2, 10, 2), (3, 10, 3), (3, 10, 4)],
        columns=["timestamp", "itemid", "categoryid"],
    )
    cleaned, report = clean_state_changes(frame, "categoryid")
    assert cleaned[["timestamp", "categoryid"]].values.tolist() == [[1, 2]]
    assert report["exact_duplicates"] == 1
    assert report["conflicting_rows_removed"] == 2
    assert report["consecutive_repeats_removed"] == 1


def test_category_paths_include_ancestors_and_detect_unknown_categories():
    tree = pd.DataFrame([(1, np.nan), (2, 1), (3, 2)], columns=["categoryid", "parentid"])
    paths, report = build_category_paths(tree, {3, 99})
    by_category = paths.set_index("categoryid")
    assert by_category.loc[3, "category_path"] == [1, 2, 3]
    assert by_category.loc[3, "category_depth"] == 2
    assert by_category.loc[99, "category_path"] == [99]
    assert report["observed_categories_missing_from_tree"] == 1


def test_sequence_samples_use_strict_history_and_cap_long_sessions():
    events = pd.DataFrame(
        [
            (0, 10, 1, "clicks", 0),
            (0, 11, 2, "clicks", 1),
            (0, 12, 3, "carts", 2),
            (0, 13, 4, "orders", 3),
            (0, 14, 5, "clicks", 4),
        ],
        columns=["session", "aid", "ts", "type", "event_order"],
    )
    samples, report = build_training_samples(
        events, {0}, min_history=2, max_history=2, max_samples_per_session=2
    )
    assert len(samples) == 2
    assert report["candidate_samples"] == 3
    assert report["capped_sessions"] == 1
    assert samples.iloc[-1]["history_aids"] == [12, 13]


def test_sequence_samples_skip_tied_target_timestamps():
    events = pd.DataFrame(
        [
            (0, 10, 1, "clicks", 0),
            (0, 11, 2, "clicks", 1),
            (0, 12, 3, "carts", 2),
            (0, 13, 3, "orders", 3),
            (0, 14, 4, "clicks", 4),
        ],
        columns=["session", "aid", "ts", "type", "event_order"],
    )
    samples, report = build_training_samples(events, {0})
    assert samples["target_aid"].tolist() == [14]
    assert report["skipped_tied_targets"] == 2


def test_asof_item_features_never_use_future_state():
    queries = pd.DataFrame([(10, 100, 5), (11, 100, 15), (12, 200, 5)], columns=["session", "aid", "ts"])
    categories = pd.DataFrame(
        [(10, 100, 1), (20, 100, 2)], columns=["timestamp", "itemid", "categoryid"]
    )
    availability = pd.DataFrame(
        [(12, 100, 1)], columns=["timestamp", "itemid", "available"]
    )
    paths = pd.DataFrame(
        [(1, 1, 0, [1], True), (2, 2, 0, [2], True)],
        columns=["categoryid", "root_categoryid", "category_depth", "category_path", "in_tree"],
    )
    enriched = enrich_asof(queries, categories, availability, paths)
    assert enriched["categoryid"].tolist() == [-1, 1, -1]
    assert enriched["available"].tolist() == [-1, 1, -1]
    assert enriched.loc[1, "category_state_ts"] == 10
    assert enriched.loc[1, "availability_state_ts"] == 12


def test_directional_transition_respects_order_and_action_weight():
    events = pd.DataFrame(
        [(0, 1, 1, "clicks", 0), (0, 2, 2, "orders", 1), (0, 3, 3, "clicks", 2)],
        columns=["session", "aid", "ts", "type", "event_order"],
    )
    transitions = build_directional_transitions(events)
    assert transitions[1][0][0] == 2
    assert transitions[2][0][0] == 3
    assert 1 not in {aid for aid, _ in transitions.get(2, [])}


def test_recall_diagnostics_report_exclusive_hits_and_conditional_auc():
    recalled = pd.DataFrame(
        [
            (0, 10, "a", 1, 0.9), (0, 11, "a", 2, 0.1),
            (0, 10, "b", 1, 0.8), (1, 20, "b", 1, 0.7),
            (1, 21, "b", 2, 0.6),
        ],
        columns=["session", "aid", "source", "source_rank", "source_score"],
    )
    labels = pd.DataFrame([(0, 10), (1, 20)], columns=["session", "target_aid"])
    diagnostics = route_diagnostics(recalled, labels)
    auc = conditional_auc_gauc(recalled, labels)
    assert diagnostics["union_recall"] == 1.0
    assert diagnostics["sources"]["b"]["exclusive_hits"] == 1
    assert auc["a"]["candidate_auc"] == 1.0
    assert auc["b"]["session_gauc"] == 1.0


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
    assert audit.loc[0, "catalog_items"] == 3
    assert 0.0 <= audit.loc[0, "candidate_catalog_coverage"] <= 1.0
    assert 999 not in set(recalled["aid"])


def test_exact_neighbor_search_is_blocked_and_deterministic():
    """分块精确检索应返回正确近邻，并用商品 ID 稳定处理并列。"""
    embeddings = np.array(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    item_ids = np.array([10, 20, 30, 15])
    neighbors = _exact_topk_cosine_neighbors(
        embeddings, item_ids, max_neighbors=2, max_similarity_bytes=16
    )
    assert [aid for aid, _ in neighbors[10]] == [15, 20]
    assert all(aid != source for source, values in neighbors.items() for aid, _ in values)


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


def test_scalable_item2vec_ann_trains_caches_and_recalls(tmp_path):
    events = pd.DataFrame(
        [
            (0, 10, 1, "clicks"), (0, 11, 2, "clicks"), (0, 12, 3, "orders"),
            (1, 10, 4, "clicks"), (1, 11, 5, "carts"), (1, 13, 6, "orders"),
            (2, 12, 7, "clicks"), (2, 11, 8, "clicks"), (2, 13, 9, "orders"),
        ],
        columns=["session", "aid", "ts", "type"],
    )
    embeddings = train_item2vec_embeddings(
        events,
        dimensions=8,
        window=2,
        negative_samples=2,
        epochs=1,
        batch_size=8,
        min_count=1,
        seed=7,
        num_threads=1,
    )
    cache = tmp_path / "item2vec.npz"
    embeddings.save(cache)
    restored = Item2VecEmbeddings.load(cache)
    assert restored.item_ids.tolist() == embeddings.item_ids.tolist()
    assert np.allclose(restored.vectors, embeddings.vectors)

    samples = pd.DataFrame(
        {
            "session": [100, 101],
            "history_aids": [np.array([10, 11]), np.array([12, 13])],
            "target_aid": [12, 11],
        }
    )
    ann = Item2VecANN(restored, hnsw_m=4, ef_construction=20, ef_search=20)
    recalled = ann.recall(samples, topk=3)
    assert set(recalled["source"]) == {"item2vec"}
    assert recalled.groupby("session").size().to_dict() == {100: 3, 101: 3}
    auc = ann.sampled_auc_gauc(samples, negatives_per_session=5, seed=9)
    assert auc["eligible_sessions"] == 2
    assert 0.0 <= auc["sampled_auc"] <= 1.0


def test_two_tower_inbatch_mask_blocks_future_and_duplicate_items():
    aids = torch.tensor([10, 20, 10])
    timestamps = torch.tensor([100, 200, 300])
    mask = causal_inbatch_mask(aids, timestamps)
    assert mask.diagonal().all()
    assert not mask[0, 1]  # future target occurrence
    assert not mask[0, 2]  # future and duplicate
    assert mask[1, 0]      # past item is a valid negative
    assert not mask[2, 0]  # duplicate item is not a false negative


def test_rolling_oof_folds_are_ordered_after_warmup():
    samples = pd.DataFrame({"target_ts": np.arange(100, 200)})
    folds = temporal_folds(samples, folds=4, warmup_fraction=0.2)
    assert len(folds) == 4
    assert folds[0][0] >= int(samples["target_ts"].quantile(0.2))
    assert all(start < end for start, end in folds)
    assert all(left[1] <= right[0] for left, right in zip(folds, folds[1:]))


def test_oof_compaction_injects_positive_and_caps_negatives():
    raw = pd.DataFrame(
        [(7, aid, "itemcf", rank, 1.0 / rank) for rank, aid in enumerate(range(20, 26), 1)],
        columns=["session", "aid", "source", "source_rank", "source_score"],
    )
    samples = pd.DataFrame(
        {"session": [7], "target_aid": [99], "target_type": ["orders"]}
    )
    compact, hits = compact_candidates(raw, samples, max_negatives=2, topk=50)
    assert hits == 0
    assert len(compact) == 3
    assert compact["label"].sum() == 1
    assert 99 in set(compact["aid"])


def test_validation_compaction_is_label_blind():
    raw = pd.DataFrame(
        [(7, aid, "itemcf", rank, 1.0 / rank) for rank, aid in enumerate([20, 21, 99], 1)],
        columns=["session", "aid", "source", "source_rank", "source_score"],
    )
    samples = pd.DataFrame(
        {"session": [7], "target_aid": [99], "target_type": ["orders"]}
    )
    compact, hits = compact_candidates(
        raw, samples, max_negatives=2, topk=50,
        inject_missing_positives=False, prioritize_positives=False,
    )
    assert hits == 1
    assert len(compact) == 2
    assert compact["label"].sum() == 0
    assert 99 not in set(compact["aid"])


def test_mmoe_and_ple_return_three_task_logits():
    features = torch.randn(7, 12)
    for model in (MMoE(12, hidden_dim=8, experts=3), PLE(12, hidden_dim=8)):
        logits = model(features)
        assert logits.shape == (7, 3)
        assert torch.isfinite(logits).all()
