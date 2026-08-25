"""Correctness tests for the versioned industrial RQ-KMeans indexer."""

import json

import numpy as np

from src.indexing.rq_kmeans import (
    INDEX_SCHEMA_VERSION,
    build_rq_kmeans_codes,
    encode_with_rq_kmeans,
    load_rq_kmeans_artifact,
    save_rq_kmeans_artifact,
)
from src.indexing.semantic_ids import collision_rate


def test_rq_kmeans_is_deterministic_and_respects_token_ranges():
    """The same seed/config must produce byte-identical catalog codes."""
    features = np.random.default_rng(4).normal(size=(60, 10)).astype(np.float32)
    kwargs = {
        "backend": "sklearn",
        "pca_dim": 8,
        "niter": 20,
        "nredo": 2,
        "max_balance_ratio": None,
    }
    left = build_rq_kmeans_codes(features, [6, 5, 4], 7, **kwargs)
    right = build_rq_kmeans_codes(features, [6, 5, 4], 7, **kwargs)

    assert np.array_equal(left.codes, right.codes)
    assert left.fingerprint == right.fingerprint
    assert left.codes.shape == (60, 3)
    for level, size in enumerate([6, 5, 4]):
        assert left.codes[:, level].min() >= 0
        assert left.codes[:, level].max() < size


def test_each_residual_codebook_reduces_unconstrained_error():
    """Nearest-centroid residual quantization should not increase reconstruction MSE."""
    features = np.random.default_rng(9).normal(size=(100, 12)).astype(np.float32)
    result = build_rq_kmeans_codes(
        features,
        [8, 8, 8],
        2,
        backend="sklearn",
        pca_dim=None,
        whiten=False,
        l2_normalize=True,
        niter=30,
        nredo=2,
        max_balance_ratio=None,
        resolve_collisions=False,
    )
    errors = [level["quantization_mse"] for level in result.level_metrics]
    assert errors[1] <= errors[0] + 1e-6
    assert errors[2] <= errors[1] + 1e-6


def test_capacity_constraint_bounds_every_level_bucket():
    """Balanced assignment must keep every learned bucket below its ceiling."""
    features = np.random.default_rng(3).normal(size=(73, 7)).astype(np.float32)
    result = build_rq_kmeans_codes(
        features,
        [7, 6],
        11,
        backend="sklearn",
        niter=15,
        nredo=1,
        max_balance_ratio=1.1,
        resolve_collisions=False,
    )
    for level, metrics in enumerate(result.level_metrics):
        counts = np.bincount(
            result.codes[:, level], minlength=[7, 6][level]
        )
        assert counts.max() <= metrics["capacity"]


def test_min_cost_final_reassignment_can_remove_prefix_collisions():
    """A final vocabulary as wide as each balanced prefix should make IDs unique."""
    rng = np.random.default_rng(13)
    features = np.vstack(
        [
            rng.normal(loc=(-4, 0), scale=0.1, size=(4, 2)),
            rng.normal(loc=(0, 4), scale=0.1, size=(4, 2)),
            rng.normal(loc=(4, 0), scale=0.1, size=(4, 2)),
        ]
    ).astype(np.float32)
    result = build_rq_kmeans_codes(
        features,
        [3, 4],
        5,
        backend="sklearn",
        whiten=False,
        l2_normalize=False,
        niter=30,
        nredo=3,
        max_balance_ratio=1.0,
        resolve_collisions=True,
    )
    assert collision_rate(result.codes) == 0
    assert result.unresolved_collisions == 0


def test_saved_artifact_contains_replay_state_and_manifest(tmp_path):
    """Publishing an index must save codebooks, preprocessing and identity."""
    features = np.random.default_rng(1).normal(size=(30, 6)).astype(np.float32)
    result = build_rq_kmeans_codes(
        features,
        [5, 5],
        8,
        backend="sklearn",
        pca_dim=4,
        niter=15,
        nredo=1,
    )
    save_rq_kmeans_artifact(result, tmp_path)

    manifest = json.loads((tmp_path / "rq_kmeans_manifest.json").read_text())
    arrays = np.load(tmp_path / "rq_kmeans_index.npz")
    assert manifest["schema_version"] == INDEX_SCHEMA_VERSION
    assert manifest["fingerprint"] == result.fingerprint
    assert manifest["codebook_sizes"] == [5, 5]
    assert np.array_equal(arrays["codes"], result.codes)
    assert {"centroids_0", "centroids_1", "pca_mean", "pca_components"} <= set(
        arrays.files
    )
    artifact = load_rq_kmeans_artifact(tmp_path)
    new_codes = encode_with_rq_kmeans(features[:3], artifact)
    assert new_codes.shape == (3, 2)
    assert new_codes[:, 0].max() < 5
