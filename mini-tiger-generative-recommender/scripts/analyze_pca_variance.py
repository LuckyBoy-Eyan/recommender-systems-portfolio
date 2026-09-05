"""Measure how much Sentence-T5 variance is retained by candidate PCA sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 192, 256])
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.85, 0.90, 0.95]
    )
    args = parser.parse_args()

    vectors = np.load(args.embeddings).astype(np.float32)
    if vectors.ndim != 2 or len(vectors) < 2:
        raise ValueError("embeddings must be a non-empty [items, dimensions] matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings contain NaN or infinite values")

    maximum = min(vectors.shape)
    dimensions = sorted(set(args.dims))
    if not dimensions or any(dim < 1 or dim > maximum for dim in dimensions):
        raise ValueError(f"dims must be between 1 and {maximum}")
    if any(value <= 0 or value > 1 for value in args.thresholds):
        raise ValueError("thresholds must be in (0, 1]")

    # Full SVD gives an exact denominator for explained-variance ratios and is
    # fitted once, so all candidate dimensions are directly comparable.
    pca = PCA(n_components=maximum, svd_solver="full")
    pca.fit(vectors)
    cumulative = np.cumsum(pca.explained_variance_ratio_, dtype=np.float64)

    retained = {
        str(dim): {
            "explained_variance_ratio": float(cumulative[dim - 1]),
            "variance_lost_ratio": float(1.0 - cumulative[dim - 1]),
        }
        for dim in dimensions
    }
    minimum_dimensions = {
        f"{threshold:.2f}": int(np.searchsorted(cumulative, threshold) + 1)
        for threshold in args.thresholds
    }

    target = 0.85
    baseline_dim = min(128, maximum)
    if cumulative[baseline_dim - 1] >= target:
        recommendation = {
            "decision": "keep_pca_128",
            "reason": "PCA-128 retains at least 85% of total variance",
            "recommended_dim": baseline_dim,
        }
    else:
        meeting_candidates = [
            dim for dim in dimensions if cumulative[dim - 1] >= target
        ]
        recommended = (
            meeting_candidates[0]
            if meeting_candidates
            else int(np.searchsorted(cumulative, target) + 1)
        )
        recommendation = {
            "decision": "increase_pca_dimension",
            "reason": "PCA-128 retains less than 85% of total variance",
            "recommended_dim": recommended,
        }

    result = {
        "items": int(vectors.shape[0]),
        "input_dimension": int(vectors.shape[1]),
        "method": "exact full-SVD PCA on the same normalized Sentence-T5 vectors used by RQ-KMeans",
        "candidate_dimensions": retained,
        "minimum_dimensions_for_threshold": minimum_dimensions,
        "recommendation": recommendation,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
