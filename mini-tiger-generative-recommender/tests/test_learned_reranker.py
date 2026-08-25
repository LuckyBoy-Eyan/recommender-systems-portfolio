import numpy as np
import torch

from src.models.candidate_ranker import CandidateRanker
from src.training.learned_reranker import (
    CandidateCache,
    FEATURE_NAMES,
    diagnose_candidate_cache,
    item_popularity,
    load_candidate_cache,
    save_candidate_cache,
    train_candidate_ranker,
)


def _cache(rows=12, candidates=4):
    """构造一个第一维特征可以直接识别正例的候选缓存。"""
    rng = np.random.default_rng(7)
    items = np.tile(np.arange(candidates, dtype=np.int32), (rows, 1))
    positions = np.arange(rows, dtype=np.int32) % candidates
    features = rng.normal(
        size=(rows, candidates, len(FEATURE_NAMES))
    ).astype(np.float32)
    features[:, :, 0] = -1.0
    features[np.arange(rows), positions, 0] = 2.0
    return CandidateCache(
        candidate_items=items,
        features=features,
        mask=np.ones((rows, candidates), dtype=bool),
        targets=positions.copy(),
        target_positions=positions,
        metadata={
            "candidate_k": candidates,
            "feature_names": list(FEATURE_NAMES),
        },
    )


def test_candidate_ranker_masks_padding():
    model = CandidateRanker(len(FEATURE_NAMES), hidden_dims=(8,), dropout=0.0)
    features = torch.zeros(2, 3, len(FEATURE_NAMES))
    mask = torch.tensor([[True, True, False], [True, False, False]])
    scores = model(features, mask)
    assert scores.shape == (2, 3)
    assert torch.isneginf(scores[0, 2])
    assert torch.isfinite(scores[0, 0])


def test_candidate_ranker_starts_from_fixed_fusion():
    model = CandidateRanker(
        len(FEATURE_NAMES), hidden_dims=(8,), dropout=0.0, base_alpha=0.25
    )
    features = torch.randn(2, 3, len(FEATURE_NAMES))
    expected = 0.25 * features[..., 0] + 0.75 * features[..., 1]
    torch.testing.assert_close(model(features), expected)


def test_sasrec_short_left_padded_history_stays_finite():
    from src.models.sasrec import SASRec

    model = SASRec(8, 6, 8, 2, 2, dropout=0.0).eval()
    histories = torch.tensor(
        [[0, 0, 0, 0, 1, 2], [0, 0, 3, 4, 5, 6]]
    )
    assert torch.isfinite(model(histories)).all()


def test_candidate_cache_roundtrip_and_diagnostics(tmp_path):
    cache = _cache()
    path = tmp_path / "cache.npz"
    save_candidate_cache(cache, path)
    loaded = load_candidate_cache(path)
    np.testing.assert_array_equal(loaded.candidate_items, cache.candidate_items)
    np.testing.assert_allclose(loaded.features, cache.features)
    diagnostics = diagnose_candidate_cache(loaded)
    assert diagnostics["candidate_recall"] == 1.0
    assert diagnostics["candidate_hits"] == len(cache.targets)


def test_popularity_uses_only_passed_sequences():
    popularity = item_popularity([[0, 0, 1], [0, 2]], 4)
    assert popularity[0] > popularity[1] > popularity[3]
    assert popularity[3] == 0.0


def test_listwise_ranker_training_runs():
    torch.manual_seed(3)
    cache = _cache(rows=30)
    model = CandidateRanker(len(FEATURE_NAMES), hidden_dims=(8,), dropout=0.0)
    trained, history = train_candidate_ranker(
        model,
        cache,
        np.arange(20),
        np.arange(20, 30),
        epochs=2,
        batch_size=10,
        learning_rate=0.01,
        patience=2,
        monitor="ndcg@2",
        ks=(1, 2),
    )
    assert history
    assert isinstance(trained, CandidateRanker)
