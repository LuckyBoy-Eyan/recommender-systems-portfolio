"""从最终 SASRec Item Embedding 构建行为感知工业 RQ-KMeans 索引。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexing.behavior_features import (
    build_behavior_aware_features,
    extract_sasrec_item_embeddings,
    save_behavior_feature_artifact,
    sha256_file,
)
from src.indexing.rq_kmeans import (
    build_rq_kmeans_codes,
    save_rq_kmeans_artifact,
)
from src.indexing.semantic_ids import (
    append_collision_token,
    codebook_diagnostics,
    collision_rate,
)
from src.models.generative import SemanticIDTransformer


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _model_parameters(codes, config) -> int:
    sizes = [int(codes[:, level].max()) + 1 for level in range(codes.shape[1])]
    model = SemanticIDTransformer(
        sizes,
        config["max_history"],
        config["hidden_dim"],
        config["num_heads"],
        config["num_layers"],
        config.get("feedforward_dim"),
    )
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kuairec_big_behavior_rq_cuda.json")
    parser.add_argument(
        "--sasrec-checkpoint",
        default="outputs/kuairec_big_cuda/sasrec_checkpoint.pt",
    )
    parser.add_argument(
        "--reference-codes",
        default="outputs/kuairec_big_cuda/semantic_codes.npy",
    )
    parser.add_argument(
        "--output", default="outputs/kuairec_big_behavior_rq/index"
    )
    args = parser.parse_args()

    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text())
    item_features_path = _resolve(config["item_features_path"])
    checkpoint_path = _resolve(args.sasrec_checkpoint)
    reference_codes_path = _resolve(args.reference_codes)
    output = _resolve(args.output)
    if output == _resolve("outputs/kuairec_big_cuda"):
        raise SystemExit("refusing to overwrite the baseline artifact directory")
    output.mkdir(parents=True, exist_ok=True)

    item_features = np.load(item_features_path).astype(np.float32)
    embeddings = extract_sasrec_item_embeddings(
        checkpoint_path, expected_num_items=len(item_features)
    )
    source = {
        "sasrec_checkpoint": str(checkpoint_path),
        "sasrec_checkpoint_sha256": sha256_file(checkpoint_path),
        "item_features": str(item_features_path),
        "item_features_sha256": sha256_file(item_features_path),
        "alignment_contract": (
            "SASRec item index and sorted prepared item feature row share the "
            "same zero-based catalog; row count verified"
        ),
    }
    item_ids_path = item_features_path.with_name("item_ids.npy")
    if item_ids_path.exists():
        source["item_ids_sha256"] = sha256_file(item_ids_path)
    fusion = build_behavior_aware_features(
        item_features,
        embeddings,
        behavior_weight=config.get("behavior_weight", 0.7),
        content_feature_dim=config.get("content_feature_dim", 64),
        behavior_pca_dim=config.get("behavior_pca_dim", 64),
        content_pca_dim=config.get("content_pca_dim", 32),
        seed=config["seed"],
        source_metadata=source,
    )
    save_behavior_feature_artifact(fusion, output)

    rq = build_rq_kmeans_codes(
        fusion.features,
        config["codebook_sizes"],
        config["seed"],
        backend=config.get("rq_backend", "auto"),
        # 分支内已经白化、归一化；再次 PCA/白化会抹掉70/30权重。
        pca_dim=None,
        whiten=False,
        l2_normalize=False,
        niter=config.get("rq_niter", 25),
        nredo=config.get("rq_nredo", 3),
        use_gpu=config.get("rq_use_gpu", False),
        max_balance_ratio=config.get("rq_max_balance_ratio", 1.25),
        resolve_collisions=config.get("rq_resolve_collisions", True),
        minibatch_threshold=config.get("minibatch_kmeans_threshold", 5000),
    )
    save_rq_kmeans_artifact(rq, output)
    codes, tail_size = append_collision_token(rq.codes)
    np.save(output / "semantic_codes.npy", codes)

    if rq.unresolved_collisions:
        raise RuntimeError(
            f"behavior RQ contains {rq.unresolved_collisions} unresolved collisions"
        )
    if collision_rate(codes) != 0.0:
        raise RuntimeError("resolved Semantic IDs are not unique")
    if any(level["used_tokens"] != size for level, size in zip(
        rq.level_metrics, config["codebook_sizes"]
    )):
        raise RuntimeError("at least one RQ level has unused tokens")

    reference_codes = np.load(reference_codes_path)
    if len(reference_codes) != len(codes):
        raise RuntimeError("reference and behavior-aware catalogs differ")
    reference_sizes = [
        int(reference_codes[:, level].max()) + 1
        for level in range(reference_codes.shape[1])
    ]
    behavior_sizes = [
        int(codes[:, level].max()) + 1 for level in range(codes.shape[1])
    ]
    reference_parameters = _model_parameters(reference_codes, config)
    behavior_parameters = _model_parameters(codes, config)
    if reference_sizes != behavior_sizes or reference_parameters != behavior_parameters:
        raise RuntimeError(
            "behavior-aware Semantic IDs do not preserve generator capacity: "
            f"{reference_sizes}/{reference_parameters} vs "
            f"{behavior_sizes}/{behavior_parameters}"
        )
    sasrec_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    sasrec_parameters = sum(
        value.numel() for value in sasrec_checkpoint["best_state"].values()
    )
    parameter_gap_ratio = abs(behavior_parameters - sasrec_parameters) / max(
        sasrec_parameters, 1
    )
    if parameter_gap_ratio > config.get("max_capacity_gap_ratio", 0.01):
        raise RuntimeError(
            "generator is not capacity matched to SASRec: "
            f"{behavior_parameters} vs {sasrec_parameters}"
        )

    average_bucket = len(codes) / np.asarray(config["codebook_sizes"])
    observed_balance = [
        level["largest_bucket"] / average
        for level, average in zip(rq.level_metrics, average_bucket)
    ]
    composite_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "rq_fingerprint": rq.fingerprint,
                "fusion": fusion.manifest,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    quality = {
        "status": "passed",
        "num_items": int(len(codes)),
        "fusion": fusion.manifest,
        "rq_fingerprint": rq.fingerprint,
        "behavior_index_fingerprint": composite_fingerprint,
        "raw_unique_codes": int(len(np.unique(rq.codes, axis=0))),
        "raw_unique_ratio": float(len(np.unique(rq.codes, axis=0)) / len(codes)),
        "raw_collision_rate": collision_rate(rq.codes),
        "resolved_unique_ratio": float(
            len(np.unique(codes, axis=0)) / len(codes)
        ),
        "tail_size": int(tail_size),
        "unresolved_collisions": int(rq.unresolved_collisions),
        "observed_max_balance_ratio": observed_balance,
        "codebook": codebook_diagnostics(rq.codes, config["codebook_sizes"]),
        "level_metrics": rq.level_metrics,
        "equal_capacity": {
            "reference_codebook_sizes": reference_sizes,
            "behavior_codebook_sizes": behavior_sizes,
            "reference_parameters": reference_parameters,
            "behavior_parameters": behavior_parameters,
            "sasrec_parameters": int(sasrec_parameters),
            "generator_vs_sasrec_gap_ratio": float(parameter_gap_ratio),
            "passed": True,
        },
    }
    (output / "behavior_rq_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2)
    )
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
