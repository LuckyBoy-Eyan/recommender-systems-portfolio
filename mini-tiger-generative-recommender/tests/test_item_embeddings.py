import numpy as np

from src.embeddings import (
    build_transition_embedding,
    build_weighted_response_embedding,
    fuse_item_embeddings,
)


def test_weighted_response_embedding_is_finite_and_normalized():
    rows = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    columns = np.asarray([0, 1, 0, 2, 1, 2, 0, 2])
    scores = np.asarray([0.1, 1.2, 1.4, 0.2, 0.8, 1.0, 0.3, 1.5])
    embedding, diagnostics = build_weighted_response_embedding(
        rows,
        columns,
        scores,
        num_items=4,
        num_users=3,
        output_dim=2,
        seed=3,
    )
    assert embedding.shape == (4, 2)
    assert np.isfinite(embedding).all()
    assert diagnostics["unique_user_item_pairs"] == 8


def test_transition_embedding_uses_only_passed_sequences():
    embedding, diagnostics = build_transition_embedding(
        [[0, 1, 2, 3], [0, 1, 3, 2]],
        4,
        output_dim=2,
        window=2,
        seed=4,
    )
    assert embedding.shape == (4, 2)
    assert diagnostics["raw_transitions"] == 10
    assert diagnostics["window"] == 2


def test_fusion_preserves_weighted_unit_norm_geometry():
    rng = np.random.default_rng(5)
    response = rng.normal(size=(8, 3)).astype(np.float32)
    transition = rng.normal(size=(8, 2)).astype(np.float32)
    fused, manifest = fuse_item_embeddings(
        {"response": response, "transition": transition},
        {"response": 3.0, "transition": 1.0},
    )
    assert fused.shape == (8, 5)
    assert np.allclose(np.linalg.norm(fused, axis=1), 1.0)
    assert manifest["weights"] == {"response": 0.75, "transition": 0.25}
