"""将 Sentence-T5 内容向量量化为三级 RQ-KMeans 唯一 Semantic ID。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexing.rq_kmeans import build_rq_kmeans_codes, save_rq_kmeans_artifact
from src.indexing.semantic_ids import (
    append_collision_token,
    codebook_diagnostics,
    collision_rate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codebook-size", type=int, default=32)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument(
        "--codebook-sizes",
        type=int,
        nargs="+",
        help="非均匀码本，例如 --codebook-sizes 64 32 32",
    )
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    features = np.load(args.embeddings).astype(np.float32)
    sizes = args.codebook_sizes or [args.codebook_size] * args.levels
    if any(size < 2 for size in sizes):
        raise ValueError("all codebook sizes must be at least 2")
    result = build_rq_kmeans_codes(
        features,
        sizes,
        args.seed,
        backend="sklearn",
        pca_dim=min(args.pca_dim, features.shape[1]),
        whiten=False,
        l2_normalize=True,
        niter=25,
        nredo=3,
        use_gpu=False,
        max_balance_ratio=1.25,
        resolve_collisions=False,
        minibatch_threshold=5000,
    )
    prefix_codes = result.codes
    codes, tail_size = append_collision_token(prefix_codes)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_rq_kmeans_artifact(result, output / "rq_artifact")
    np.save(output / "semantic_codes.npy", codes)
    manifest = {
        "schema_version": "sentence-t5-rq-kmeans-v2",
        "codebook_sizes": sizes,
        "tail_size": int(tail_size),
        "prefix_collision_rate": collision_rate(prefix_codes),
        "complete_collision_rate": collision_rate(codes),
        "prefix_diagnostics": codebook_diagnostics(prefix_codes, sizes),
        "complete_codebook_sizes": [*sizes, int(tail_size)],
        "rq_kmeans": result.manifest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
