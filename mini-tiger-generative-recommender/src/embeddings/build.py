"""利用稠密响应、序列转移和内容信息构建 SID 输入 embedding。"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def _svd_embedding(
    matrix: csr_matrix,
    output_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    components = min(output_dim, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if components < 1:
        raise ValueError("matrix is too small for item embedding")
    model = TruncatedSVD(n_components=components, random_state=seed)
    embedding = model.fit_transform(matrix).astype(np.float32)
    if components < output_dim:
        embedding = np.pad(embedding, ((0, 0), (0, output_dim - components)))
    embedding = normalize(embedding, axis=1).astype(np.float32)
    return embedding, {
        "output_dim": int(output_dim),
        "effective_components": int(components),
        "explained_variance_ratio_sum": float(
            model.explained_variance_ratio_.sum()
        ),
        "nonzero_entries": int(matrix.nnz),
    }


def build_weighted_response_embedding(
    item_rows: np.ndarray,
    user_columns: np.ndarray,
    response_scores: np.ndarray,
    *,
    num_items: int,
    num_users: int,
    output_dim: int = 64,
    seed: int = 2026,
) -> tuple[np.ndarray, dict]:
    """对连续观看响应按 user-item 聚合、用户内中心化后执行 SVD。"""
    item_rows = np.asarray(item_rows, dtype=np.int64)
    user_columns = np.asarray(user_columns, dtype=np.int64)
    response_scores = np.asarray(response_scores, dtype=np.float32)
    if not (len(item_rows) == len(user_columns) == len(response_scores)):
        raise ValueError("response coordinate arrays must have equal lengths")
    if len(response_scores) == 0:
        raise ValueError("response coordinates must not be empty")
    shape = (num_items, num_users)
    sums = coo_matrix(
        (response_scores, (item_rows, user_columns)), shape=shape
    ).tocsr()
    counts = coo_matrix(
        (np.ones(len(item_rows), dtype=np.float32), (item_rows, user_columns)),
        shape=shape,
    ).tocsr()
    if not np.array_equal(sums.indices, counts.indices) or not np.array_equal(
        sums.indptr, counts.indptr
    ):
        raise RuntimeError("response sums and counts are misaligned")
    averages = sums.copy()
    averages.data /= counts.data

    centered = averages.tocsc(copy=True)
    for user in range(num_users):
        start, end = centered.indptr[user : user + 2]
        if end > start:
            centered.data[start:end] -= centered.data[start:end].mean()
    centered = centered.tocsr()
    centered.eliminate_zeros()
    embedding, diagnostics = _svd_embedding(centered, output_dim, seed)
    diagnostics.update(
        {
            "events": int(len(item_rows)),
            "unique_user_item_pairs": int(counts.nnz),
            "response_min": float(response_scores.min()),
            "response_max": float(response_scores.max()),
            "user_centered": True,
        }
    )
    return embedding, diagnostics


def build_transition_embedding(
    sequences: list[list[int]],
    num_items: int,
    *,
    output_dim: int = 64,
    window: int = 3,
    seed: int = 2026,
) -> tuple[np.ndarray, dict]:
    """由训练序列构建带位置衰减的有向 PPMI 转移 embedding。"""
    if window < 1:
        raise ValueError("transition window must be positive")
    row_parts, column_parts, value_parts = [], [], []
    transitions = 0
    for sequence in sequences:
        values = np.asarray(sequence, dtype=np.int64)
        for distance in range(1, min(window, len(values) - 1) + 1):
            source, target = values[:-distance], values[distance:]
            if len(source):
                row_parts.append(source)
                column_parts.append(target)
                value_parts.append(
                    np.full(len(source), 1.0 / distance, dtype=np.float32)
                )
                transitions += len(source)
    if not row_parts:
        raise ValueError("no transitions were produced")
    matrix = coo_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(num_items, num_items),
    ).tocsr()
    rows, columns = matrix.nonzero()
    values = matrix.data
    total = float(values.sum())
    row_mass = np.asarray(matrix.sum(axis=1)).ravel()
    column_mass = np.asarray(matrix.sum(axis=0)).ravel()
    denominator = row_mass[rows] * column_mass[columns]
    ppmi = np.maximum(
        np.log(np.maximum(values * total, 1e-12) / np.maximum(denominator, 1e-12)),
        0.0,
    ).astype(np.float32)
    positive = ppmi > 0
    ppmi_matrix = coo_matrix(
        (ppmi[positive], (rows[positive], columns[positive])),
        shape=matrix.shape,
    ).tocsr()
    embedding, diagnostics = _svd_embedding(ppmi_matrix, output_dim, seed)
    diagnostics.update(
        {
            "window": int(window),
            "raw_transitions": int(transitions),
            "directed_pairs": int(matrix.nnz),
            "ppmi_pairs": int(ppmi_matrix.nnz),
        }
    )
    return embedding, diagnostics


def fuse_item_embeddings(
    branches: dict[str, np.ndarray],
    weights: dict[str, float],
) -> tuple[np.ndarray, dict]:
    """不做白化，分支单位化后按平方距离权重拼接。"""
    if set(branches) != set(weights):
        raise ValueError("embedding branches and weights must have the same names")
    total_weight = float(sum(weights.values()))
    if total_weight <= 0 or any(value < 0 for value in weights.values()):
        raise ValueError("embedding weights must be non-negative with positive sum")
    row_counts = {len(values) for values in branches.values()}
    if len(row_counts) != 1:
        raise ValueError("embedding branches must contain the same items")
    normalized_weights = {
        name: float(weight) / total_weight for name, weight in weights.items()
    }
    parts = []
    dimensions = {}
    for name, values in branches.items():
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"invalid {name} embedding")
        centered = values - values.mean(axis=0, keepdims=True)
        centered = normalize(centered, axis=1).astype(np.float32)
        parts.append(np.sqrt(normalized_weights[name]) * centered)
        dimensions[name] = int(values.shape[1])
    fused = np.concatenate(parts, axis=1).astype(np.float32)
    fused = normalize(fused, axis=1).astype(np.float32)
    return fused, {
        "weights": normalized_weights,
        "branch_dimensions": dimensions,
        "output_dim": int(fused.shape[1]),
        "whiten": False,
        "l2_normalize": True,
    }
