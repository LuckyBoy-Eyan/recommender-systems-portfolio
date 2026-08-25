import numpy as np
import pytest
import torch

from src.indexing.behavior_features import (
    FUSION_SCHEMA_VERSION,
    build_behavior_aware_features,
    extract_sasrec_item_embeddings,
    save_behavior_feature_artifact,
)
from src.models.generative import SemanticIDTransformer


def test_extract_sasrec_embedding_removes_padding_and_checks_alignment(tmp_path):
    weight = torch.arange(30, dtype=torch.float32).reshape(6, 5)
    path = tmp_path / "sasrec.pt"
    torch.save({"best_state": {"item_embedding.weight": weight}}, path)
    extracted = extract_sasrec_item_embeddings(path, expected_num_items=5)
    np.testing.assert_array_equal(extracted, weight[1:].numpy())
    with pytest.raises(ValueError, match="catalog size"):
        extract_sasrec_item_embeddings(path, expected_num_items=4)


def test_behavior_content_fusion_is_deterministic_and_unit_normalized(tmp_path):
    rng = np.random.default_rng(12)
    items = rng.normal(size=(80, 12)).astype(np.float32)
    behavior = rng.normal(size=(80, 10)).astype(np.float32)
    kwargs = {
        "behavior_weight": 0.7,
        "content_feature_dim": 8,
        "behavior_pca_dim": 6,
        "content_pca_dim": 4,
        "seed": 3,
    }
    left = build_behavior_aware_features(items, behavior, **kwargs)
    right = build_behavior_aware_features(items, behavior, **kwargs)
    np.testing.assert_allclose(left.features, right.features)
    np.testing.assert_allclose(
        np.linalg.norm(left.features, axis=1), np.ones(80), atol=1e-5
    )
    assert left.features.shape == (80, 10)
    assert left.manifest["schema_version"] == FUSION_SCHEMA_VERSION
    assert left.manifest["behavior_weight"] == 0.7
    save_behavior_feature_artifact(left, tmp_path)
    assert (tmp_path / "behavior_aware_features.npy").exists()
    assert (tmp_path / "behavior_feature_transform.npz").exists()
    assert (tmp_path / "behavior_feature_manifest.json").exists()


def test_behavior_content_fusion_rejects_misaligned_catalog():
    with pytest.raises(ValueError, match="rows must align"):
        build_behavior_aware_features(
            np.ones((5, 4), dtype=np.float32),
            np.ones((4, 3), dtype=np.float32),
            content_feature_dim=4,
            behavior_pca_dim=2,
            content_pca_dim=2,
        )


def test_equal_capacity_generator_is_within_one_percent_of_sasrec():
    generator = SemanticIDTransformer(
        [64, 64, 64, 1],
        max_history=100,
        hidden_dim=192,
        num_heads=4,
        num_layers=3,
        feedforward_dim=720,
    )
    parameters = sum(value.numel() for value in generator.parameters())
    sasrec_parameters = 1_627_230
    assert parameters == 1_633_009
    assert abs(parameters - sasrec_parameters) / sasrec_parameters < 0.01
