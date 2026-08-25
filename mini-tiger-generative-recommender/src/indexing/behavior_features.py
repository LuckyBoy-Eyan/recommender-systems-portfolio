"""使用 SASRec Item Embedding 构造可复现的行为感知 RQ 输入特征。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


FUSION_SCHEMA_VERSION = "behavior-content-fusion-v1"


@dataclass
class BehaviorFeatureResult:
    """行为/内容特征融合结果及其可发布变换参数。"""

    features: np.ndarray
    arrays: dict[str, np.ndarray]
    manifest: dict


def sha256_file(path: str | Path) -> str:
    """流式计算输入产物摘要，供索引版本追踪和回滚。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_sasrec_item_embeddings(
    checkpoint_path: str | Path,
    *,
    expected_num_items: int | None = None,
) -> np.ndarray:
    """从 SASRec 最佳 checkpoint 中提取去掉 padding 行的 Item Embedding。

    只接受 ``best_state``，避免误用最后一个 epoch。真实 item i 对应权重第 i+1
    行，第0行是 padding。
    """
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if "best_state" not in checkpoint:
        raise ValueError("SASRec checkpoint does not contain best_state")
    state = checkpoint["best_state"]
    if "item_embedding.weight" not in state:
        raise ValueError("SASRec checkpoint has no item_embedding.weight")
    weight = state["item_embedding.weight"].detach().cpu().float().numpy()
    if weight.ndim != 2 or len(weight) < 2:
        raise ValueError("invalid SASRec item embedding shape")
    embeddings = np.ascontiguousarray(weight[1:], dtype=np.float32)
    if expected_num_items is not None and len(embeddings) != expected_num_items:
        raise ValueError(
            "SASRec catalog size differs from item features: "
            f"{len(embeddings)} != {expected_num_items}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("SASRec item embeddings contain NaN or infinity")
    if not np.any(np.var(embeddings, axis=0) > 1e-12):
        raise ValueError("SASRec item embeddings have collapsed variance")
    return embeddings


def _fit_branch(
    values: np.ndarray,
    output_dim: int,
    *,
    seed: int,
    prefix: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """对单个特征分支独立 PCA、白化、L2 归一化。"""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError(f"{prefix} features must be a non-empty matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{prefix} features contain NaN or infinity")
    components = min(int(output_dim), values.shape[0], values.shape[1])
    if components <= 0:
        raise ValueError(f"{prefix} PCA dimension must be positive")
    pca = PCA(
        n_components=components,
        whiten=True,
        random_state=seed,
        svd_solver="auto",
    )
    transformed = pca.fit_transform(values).astype(np.float32)
    transformed = normalize(transformed, norm="l2", axis=1).astype(np.float32)
    arrays = {
        f"{prefix}_pca_mean": np.asarray(pca.mean_, dtype=np.float32),
        f"{prefix}_pca_components": np.asarray(
            pca.components_, dtype=np.float32
        ),
        f"{prefix}_pca_explained_variance": np.asarray(
            pca.explained_variance_, dtype=np.float32
        ),
    }
    metadata = {
        "input_dim": int(values.shape[1]),
        "output_dim": int(components),
        "whiten": True,
        "l2_normalize": True,
        "explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
    }
    return transformed, arrays, metadata


def build_behavior_aware_features(
    item_features: np.ndarray,
    sasrec_embeddings: np.ndarray,
    *,
    behavior_weight: float = 0.7,
    content_feature_dim: int = 64,
    behavior_pca_dim: int = 64,
    content_pca_dim: int = 32,
    seed: int = 2026,
    source_metadata: dict | None = None,
) -> BehaviorFeatureResult:
    """融合行为和内容分支，输出保持权重语义的定长向量。

    两个分支先独立白化和单位化，再分别乘 ``sqrt(weight)``。拼接向量的平方
    L2 距离因而近似等于两个分支距离的加权和。融合后不应再次白化，否则会
    抹掉显式设置的70/30权重。
    """
    content = np.asarray(item_features, dtype=np.float32)
    behavior = np.asarray(sasrec_embeddings, dtype=np.float32)
    if content.ndim != 2 or behavior.ndim != 2:
        raise ValueError("item and SASRec features must be matrices")
    if len(content) != len(behavior):
        raise ValueError("item and SASRec feature rows must align")
    if not 0.0 <= behavior_weight <= 1.0:
        raise ValueError("behavior_weight must be between zero and one")
    if not 0 < content_feature_dim <= content.shape[1]:
        raise ValueError("content_feature_dim exceeds available item features")
    content = content[:, :content_feature_dim]
    behavior_vectors, behavior_arrays, behavior_meta = _fit_branch(
        behavior, behavior_pca_dim, seed=seed, prefix="behavior"
    )
    content_vectors, content_arrays, content_meta = _fit_branch(
        content, content_pca_dim, seed=seed + 1, prefix="content"
    )
    content_weight = 1.0 - behavior_weight
    fused = np.concatenate(
        [
            np.sqrt(behavior_weight) * behavior_vectors,
            np.sqrt(content_weight) * content_vectors,
        ],
        axis=1,
    ).astype(np.float32)
    norms = np.linalg.norm(fused, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("fused features are not unit normalized")
    manifest = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "num_items": int(len(fused)),
        "output_dim": int(fused.shape[1]),
        "behavior_weight": float(behavior_weight),
        "content_weight": float(content_weight),
        "content_columns": [0, int(content_feature_dim)],
        "behavior": behavior_meta,
        "content": content_meta,
        "source": source_metadata or {},
    }
    return BehaviorFeatureResult(
        features=np.ascontiguousarray(fused),
        arrays={**behavior_arrays, **content_arrays},
        manifest=manifest,
    )


def save_behavior_feature_artifact(
    result: BehaviorFeatureResult, output_dir: str | Path
) -> None:
    """保存融合特征、PCA状态与来源清单。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "behavior_aware_features.npy", result.features)
    np.savez_compressed(output / "behavior_feature_transform.npz", **result.arrays)
    (output / "behavior_feature_manifest.json").write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2)
    )
