"""Production-oriented residual-quantized K-Means for Semantic IDs.

The module deliberately keeps the indexer independent from the recommendation
model.  It turns one continuous item vector into a fixed-length token sequence,
persists every transform and codebook, and exposes diagnostics needed to decide
whether an index is safe to publish.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

INDEX_SCHEMA_VERSION = "rq-kmeans-v1"


@dataclass
class RQKMeansResult:
    """All state produced while building one immutable RQ-KMeans index."""

    codes: np.ndarray
    centroids: list[np.ndarray]
    backend: str
    preprocessing: dict[str, Any]
    training: dict[str, Any]
    level_metrics: list[dict[str, Any]]
    collision_reassignments: int
    unresolved_collisions: int
    fingerprint: str

    def manifest(self) -> dict[str, Any]:
        """Return JSON-serializable index metadata, excluding large arrays."""
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "backend": self.backend,
            "num_items": int(len(self.codes)),
            "codebook_sizes": [int(len(centroids)) for centroids in self.centroids],
            "feature_dim": int(self.centroids[0].shape[1]),
            "preprocessing": {
                key: value
                for key, value in self.preprocessing.items()
                if not isinstance(value, np.ndarray)
            },
            "training": self.training,
            "level_metrics": self.level_metrics,
            "collision_reassignments": self.collision_reassignments,
            "unresolved_collisions": self.unresolved_collisions,
        }


def _index_fingerprint(
    codes: np.ndarray,
    centroids: list[np.ndarray],
    preprocessing: dict[str, Any],
    training: dict[str, Any],
) -> str:
    """Hash all state that can change the item-to-token mapping."""
    digest = hashlib.sha256()
    digest.update(INDEX_SCHEMA_VERSION.encode())
    digest.update(
        np.asarray([len(codebook) for codebook in centroids], dtype=np.int64).tobytes()
    )
    digest.update(
        json.dumps(
            {
                key: value
                for key, value in preprocessing.items()
                if not isinstance(value, np.ndarray)
            },
            sort_keys=True,
        ).encode()
    )
    digest.update(json.dumps(training, sort_keys=True).encode())
    for key in ("pca_mean", "pca_components", "pca_explained_variance"):
        if key in preprocessing:
            digest.update(preprocessing[key].tobytes())
    digest.update(codes.tobytes())
    for codebook in centroids:
        digest.update(codebook.tobytes())
    return digest.hexdigest()


def _squared_distances(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute the dense squared-L2 assignment matrix without a third axis."""
    vector_norms = np.sum(vectors * vectors, axis=1, keepdims=True)
    centroid_norms = np.sum(centroids * centroids, axis=1)[None, :]
    distances = vector_norms + centroid_norms - 2.0 * vectors @ centroids.T
    return np.maximum(distances, 0.0).astype(np.float32, copy=False)


def _fit_sklearn(
    vectors: np.ndarray,
    size: int,
    seed: int,
    niter: int,
    nredo: int,
    minibatch_threshold: int,
) -> np.ndarray:
    """Fit one codebook with sklearn; this is the portable CPU fallback."""
    if len(vectors) >= minibatch_threshold:
        model = MiniBatchKMeans(
            n_clusters=size,
            random_state=seed,
            n_init=nredo,
            max_iter=niter,
            batch_size=min(4096, len(vectors)),
            reassignment_ratio=0.0,
        )
    else:
        model = KMeans(
            n_clusters=size,
            random_state=seed,
            n_init=nredo,
            max_iter=niter,
            algorithm="lloyd",
        )
    model.fit(vectors)
    return np.asarray(model.cluster_centers_, dtype=np.float32)


def _fit_faiss(
    vectors: np.ndarray,
    size: int,
    seed: int,
    niter: int,
    nredo: int,
    use_gpu: bool,
) -> np.ndarray:
    """Fit one codebook with Faiss Kmeans, optionally on a CUDA GPU."""
    import faiss  # type: ignore[import-not-found]

    model = faiss.Kmeans(
        vectors.shape[1],
        size,
        niter=niter,
        nredo=nredo,
        seed=seed,
        gpu=use_gpu,
        verbose=False,
        spherical=False,
    )
    model.train(np.ascontiguousarray(vectors, dtype=np.float32))
    return np.asarray(model.centroids, dtype=np.float32).reshape(size, vectors.shape[1])


def _resolve_backend(requested: str) -> str:
    """Resolve ``auto`` to Faiss when importable, otherwise sklearn."""
    if requested not in {"auto", "faiss", "sklearn"}:
        raise ValueError("backend must be one of: auto, faiss, sklearn")
    if requested == "sklearn":
        return requested
    try:
        import faiss  # type: ignore[import-not-found]  # noqa: F401

        return "faiss"
    except ImportError:
        if requested == "faiss":
            raise RuntimeError(
                "rq_backend='faiss' requires the optional faiss-cpu/faiss-gpu package"
            ) from None
        return "sklearn"


def _capacity_assign(
    distances: np.ndarray, max_balance_ratio: float | None
) -> tuple[np.ndarray, int]:
    """Assign nearest centroids with an optional deterministic capacity ceiling.

    Points with the largest gap between their best and second-best centroid are
    placed first because moving them is most expensive.  This approximates a
    balanced assignment without constructing a catalog-sized min-cost-flow graph.
    """
    nearest = np.argmin(distances, axis=1).astype(np.int64)
    if max_balance_ratio is None or not math.isfinite(max_balance_ratio):
        return nearest, len(distances)
    if max_balance_ratio < 1.0:
        raise ValueError("max_balance_ratio must be at least 1.0")

    num_items, size = distances.shape
    capacity = max(1, math.ceil(num_items / size * max_balance_ratio))
    if capacity * size < num_items:
        capacity = math.ceil(num_items / size)

    preferences = np.argsort(distances, axis=1, kind="stable")
    if size == 1:
        margins = np.full(num_items, np.inf, dtype=np.float32)
    else:
        margins = (
            distances[np.arange(num_items), preferences[:, 1]]
            - distances[np.arange(num_items), preferences[:, 0]]
        )
    order = np.argsort(-margins, kind="stable")
    counts = np.zeros(size, dtype=np.int64)
    labels = np.empty(num_items, dtype=np.int64)
    for item in order:
        for token in preferences[item]:
            if counts[token] < capacity:
                labels[item] = token
                counts[token] += 1
                break
        else:  # Defensive fallback for floating-point/configuration edge cases.
            token = int(np.argmin(counts))
            labels[item] = token
            counts[token] += 1
    return labels, capacity


def _resolve_final_level_collisions(
    codes: np.ndarray,
    final_distances: np.ndarray,
    capacity: int,
) -> tuple[np.ndarray, int]:
    """Give equal-prefix items distinct final tokens at minimum added L2 cost."""
    if codes.shape[1] < 2:
        prefixes = np.zeros((len(codes), 1), dtype=np.int64)
    else:
        prefixes = codes[:, :-1]
    _, inverse, counts = np.unique(
        prefixes, axis=0, return_inverse=True, return_counts=True
    )
    global_counts = np.bincount(
        codes[:, -1], minlength=final_distances.shape[1]
    ).astype(np.int64)
    changed = 0

    # Resolve larger collision groups first; they have fewer feasible matchings.
    for group_id in np.argsort(-counts, kind="stable"):
        indices = np.flatnonzero(inverse == group_id)
        if len(indices) <= 1 or len(indices) > final_distances.shape[1]:
            continue
        current = codes[indices, -1]
        if len(np.unique(current)) == len(indices):
            continue

        np.subtract.at(global_counts, current, 1)
        available = np.flatnonzero(global_counts < capacity)
        # Capacity and per-prefix uniqueness can occasionally conflict. In that
        # rare case, uniqueness wins and the manifest exposes the imbalance.
        candidates = (
            available
            if len(available) >= len(indices)
            else np.arange(final_distances.shape[1])
        )
        rows, columns = linear_sum_assignment(
            final_distances[np.ix_(indices, candidates)]
        )
        reassigned = candidates[columns]
        changed += int(np.count_nonzero(current[rows] != reassigned))
        codes[indices[rows], -1] = reassigned
        np.add.at(global_counts, reassigned, 1)
    return codes, changed


def _preprocess_features(
    features: np.ndarray,
    seed: int,
    pca_dim: int | None,
    whiten: bool,
    l2_normalize: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit deterministic PCA/whitening and normalization for index training."""
    vectors = np.asarray(features, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("features must be a non-empty [num_items, feature_dim] matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("features contain NaN or infinite values")

    state: dict[str, Any] = {
        "input_dim": int(vectors.shape[1]),
        "pca_dim": None,
        "whiten": bool(whiten),
        "l2_normalize": bool(l2_normalize),
    }
    if pca_dim is not None:
        components = min(int(pca_dim), vectors.shape[0], vectors.shape[1])
        if components < 1:
            raise ValueError("pca_dim must be positive")
        pca = PCA(
            n_components=components,
            whiten=whiten,
            random_state=seed,
            svd_solver="auto",
        )
        vectors = pca.fit_transform(vectors).astype(np.float32)
        state.update(
            {
                "pca_dim": components,
                "pca_mean": np.asarray(pca.mean_, dtype=np.float32),
                "pca_components": np.asarray(pca.components_, dtype=np.float32),
                "pca_explained_variance": np.asarray(
                    pca.explained_variance_, dtype=np.float32
                ),
            }
        )
    if l2_normalize:
        vectors = normalize(vectors, norm="l2", axis=1).astype(np.float32)
    return np.ascontiguousarray(vectors), state


def build_rq_kmeans_codes(
    features: np.ndarray,
    codebook_sizes: list[int],
    seed: int,
    *,
    backend: str = "auto",
    pca_dim: int | None = None,
    whiten: bool = True,
    l2_normalize: bool = True,
    niter: int = 25,
    nredo: int = 3,
    use_gpu: bool = False,
    max_balance_ratio: float | None = 1.25,
    resolve_collisions: bool = True,
    minibatch_threshold: int = 5000,
) -> RQKMeansResult:
    """Build a multi-level residual-quantized Semantic ID index.

    Each level clusters the residual left by all previous codebooks.  Assignment
    is optionally capacity constrained, and the final level uses a Hungarian
    reassignment inside colliding prefixes to preserve semantics while avoiding
    unnecessary tail-token growth.
    """
    if not codebook_sizes or any(size < 2 for size in codebook_sizes):
        raise ValueError("codebook_sizes must contain integers greater than one")
    if any(size > len(features) for size in codebook_sizes):
        raise ValueError("a codebook cannot contain more centroids than items")

    vectors, preprocessing = _preprocess_features(
        features, seed, pca_dim, whiten, l2_normalize
    )
    resolved_backend = _resolve_backend(backend)
    residual = vectors.copy()
    codes = np.empty((len(vectors), len(codebook_sizes)), dtype=np.int64)
    centroids: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    final_distances: np.ndarray | None = None
    final_residual_input: np.ndarray | None = None
    final_capacity = len(vectors)

    for level, size in enumerate(codebook_sizes):
        level_input = residual
        if resolved_backend == "faiss":
            level_centroids = _fit_faiss(
                residual, size, seed + level, niter, nredo, use_gpu
            )
        else:
            level_centroids = _fit_sklearn(
                residual,
                size,
                seed + level,
                niter,
                nredo,
                minibatch_threshold,
            )
        distances = _squared_distances(residual, level_centroids)
        labels, capacity = _capacity_assign(distances, max_balance_ratio)
        residual = residual - level_centroids[labels]
        codes[:, level] = labels
        centroids.append(level_centroids)
        token_counts = np.bincount(labels, minlength=size)
        metrics.append(
            {
                "level": level,
                "quantization_mse": float(np.mean(np.sum(residual * residual, axis=1))),
                "residual_l2_mean": float(np.mean(np.linalg.norm(residual, axis=1))),
                "capacity": int(capacity),
                "largest_bucket": int(token_counts.max()),
                "smallest_bucket": int(token_counts.min()),
                "used_tokens": int(np.count_nonzero(token_counts)),
            }
        )
        if level == len(codebook_sizes) - 1:
            final_distances = distances
            final_residual_input = level_input
            final_capacity = capacity

    reassignments = 0
    if resolve_collisions and final_distances is not None:
        assert final_residual_input is not None
        codes, reassignments = _resolve_final_level_collisions(
            codes, final_distances, final_capacity
        )
        final_counts = np.bincount(codes[:, -1], minlength=codebook_sizes[-1])
        metrics[-1]["quantization_mse_before_collision_resolution"] = metrics[-1][
            "quantization_mse"
        ]
        resolved_residual = (
            final_residual_input - centroids[-1][codes[:, -1]]
        )
        metrics[-1]["quantization_mse"] = float(
            np.mean(np.sum(resolved_residual * resolved_residual, axis=1))
        )
        metrics[-1]["residual_l2_mean"] = float(
            np.mean(np.linalg.norm(resolved_residual, axis=1))
        )
        metrics[-1]["largest_bucket_after_collision_resolution"] = int(
            final_counts.max()
        )

    unresolved = len(codes) - len(np.unique(codes, axis=0))
    training = {
        "seed": int(seed),
        "niter": int(niter),
        "nredo": int(nredo),
        "use_gpu": bool(use_gpu),
        "max_balance_ratio": max_balance_ratio,
        "resolve_collisions": bool(resolve_collisions),
        "minibatch_threshold": int(minibatch_threshold),
    }
    fingerprint = _index_fingerprint(codes, centroids, preprocessing, training)
    return RQKMeansResult(
        codes=codes,
        centroids=centroids,
        backend=resolved_backend,
        preprocessing=preprocessing,
        training=training,
        level_metrics=metrics,
        collision_reassignments=reassignments,
        unresolved_collisions=unresolved,
        fingerprint=fingerprint,
    )


def save_rq_kmeans_artifact(result: RQKMeansResult, output_dir: str | Path) -> None:
    """Persist the replayable codebooks, preprocessing state, codes and manifest."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "codes": result.codes,
        **{
            f"centroids_{level}": centroids
            for level, centroids in enumerate(result.centroids)
        },
    }
    for key, value in result.preprocessing.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
    np.savez_compressed(output / "rq_kmeans_index.npz", **arrays)
    (output / "rq_kmeans_manifest.json").write_text(
        json.dumps(result.manifest(), ensure_ascii=False, indent=2)
    )


def load_rq_kmeans_artifact(input_dir: str | Path) -> dict[str, Any]:
    """Load a published index into memory and verify its schema."""
    source = Path(input_dir)
    manifest = json.loads((source / "rq_kmeans_manifest.json").read_text())
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported RQ-KMeans schema: {manifest.get('schema_version')}"
        )
    with np.load(source / "rq_kmeans_index.npz") as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    preprocessing = dict(manifest["preprocessing"])
    for key in ("pca_mean", "pca_components", "pca_explained_variance"):
        if key in arrays:
            preprocessing[key] = arrays[key]
    centroids = [
        arrays[f"centroids_{level}"]
        for level in range(len(manifest["codebook_sizes"]))
    ]
    actual_fingerprint = _index_fingerprint(
        arrays["codes"], centroids, preprocessing, manifest["training"]
    )
    if actual_fingerprint != manifest.get("fingerprint"):
        raise ValueError("RQ-KMeans artifact fingerprint mismatch")
    return {"manifest": manifest, "arrays": arrays}


def encode_with_rq_kmeans(
    features: np.ndarray, artifact: dict[str, Any]
) -> np.ndarray:
    """Encode new item features with a loaded immutable index.

    New-item insertion uses nearest-centroid assignment and does not mutate global
    balancing or collision state.  A collision therefore signals that the item
    needs a tail token or that the catalog index should be rebuilt.
    """
    manifest = artifact["manifest"]
    arrays = artifact["arrays"]
    preprocessing = manifest["preprocessing"]
    vectors = np.asarray(features, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != preprocessing["input_dim"]:
        raise ValueError(
            f"features must have shape [N, {preprocessing['input_dim']}]"
        )
    if preprocessing["pca_dim"] is not None:
        vectors = (vectors - arrays["pca_mean"]) @ arrays["pca_components"].T
        if preprocessing["whiten"]:
            vectors = vectors / np.sqrt(
                np.maximum(arrays["pca_explained_variance"], 1e-12)
            )
    if preprocessing["l2_normalize"]:
        vectors = normalize(vectors, norm="l2", axis=1)
    residual = np.ascontiguousarray(vectors, dtype=np.float32)
    codes = []
    for level in range(len(manifest["codebook_sizes"])):
        centroids = arrays[f"centroids_{level}"]
        labels = np.argmin(_squared_distances(residual, centroids), axis=1)
        codes.append(labels)
        residual = residual - centroids[labels]
    return np.column_stack(codes).astype(np.int64)
