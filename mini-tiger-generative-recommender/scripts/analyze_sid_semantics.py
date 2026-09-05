"""Diagnose RQ-KMeans utilization, collisions and category-prefix semantics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score


CATEGORY_FIELDS = (
    "first_level_category_name",
    "second_level_category_name",
    "third_level_category_name",
)


def _pairs(counts: np.ndarray) -> int:
    values = counts.astype(np.int64, copy=False)
    return int(np.sum(values * (values - 1) // 2))


def _same_pair_count(matrix: np.ndarray) -> int:
    if len(matrix) < 2:
        return 0
    _, counts = np.unique(matrix, axis=0, return_counts=True)
    return _pairs(counts)


def analyze_category(labels: np.ndarray, codes: np.ndarray) -> dict:
    """Compare exact within-category and cross-category SID pair statistics."""
    valid = np.asarray([bool(str(value).strip()) for value in labels])
    labels = labels[valid].astype(str)
    codes = codes[valid]
    category_names, category_ids = np.unique(labels, return_inverse=True)
    category_counts = np.bincount(category_ids)
    within_pairs = _pairs(category_counts)
    all_pairs = len(labels) * (len(labels) - 1) // 2
    cross_pairs = all_pairs - within_pairs
    if within_pairs == 0 or cross_pairs == 0:
        return {
            "items_with_category": int(len(labels)),
            "categories": int(len(category_names)),
            "status": "insufficient_pairs",
        }

    level_match = []
    for level in range(codes.shape[1]):
        token = codes[:, level : level + 1]
        within_match = _same_pair_count(
            np.column_stack([category_ids, token])
        )
        global_match = _same_pair_count(token)
        cross_match = global_match - within_match
        level_match.append(
            {
                "level": level + 1,
                "same_category_match_rate": within_match / within_pairs,
                "different_category_match_rate": cross_match / cross_pairs,
            }
        )

    prefix_match = []
    for depth in range(1, codes.shape[1] + 1):
        prefix = codes[:, :depth]
        within_match = _same_pair_count(
            np.column_stack([category_ids, prefix])
        )
        global_match = _same_pair_count(prefix)
        cross_match = global_match - within_match
        same_rate = within_match / within_pairs
        different_rate = cross_match / cross_pairs
        prefix_match.append(
            {
                "prefix_depth": depth,
                "same_category_match_rate": same_rate,
                "different_category_match_rate": different_rate,
                "lift": same_rate / max(different_rate, 1e-12),
            }
        )

    same_hamming = sum(
        1.0 - row["same_category_match_rate"] for row in level_match
    )
    different_hamming = sum(
        1.0 - row["different_category_match_rate"] for row in level_match
    )
    first_lift = prefix_match[0]["lift"]
    reduction = different_hamming - same_hamming
    if first_lift >= 1.20 and reduction >= 0.15:
        verdict = "strong"
    elif first_lift >= 1.05 and reduction >= 0.05:
        verdict = "moderate"
    else:
        verdict = "weak_or_random"
    return {
        "items_with_category": int(len(labels)),
        "categories": int(len(category_names)),
        "same_category_pairs": int(within_pairs),
        "different_category_pairs": int(cross_pairs),
        "mean_hamming_distance": {
            "same_category": same_hamming,
            "different_category": different_hamming,
            "reduction": reduction,
        },
        "per_level_token_match": level_match,
        "shared_prefix": prefix_match,
        "first_token_category_nmi": float(
            normalized_mutual_info_score(category_ids, codes[:, 0])
        ),
        "heuristic_semantic_signal": verdict,
    }


def _read_categories(path: Path, item_ids: np.ndarray) -> dict[str, np.ndarray]:
    patterns = {
        field: re.compile(rf"(?:^|;\s*){re.escape(field)}:\s*([^;]+)")
        for field in CATEGORY_FIELDS
    }
    text_by_item: dict[int, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text_by_item[int(row["item_id"])] = row.get("text", "")
    result = {}
    for field, pattern in patterns.items():
        values = []
        for item in item_ids.tolist():
            match = pattern.search(text_by_item.get(int(item), ""))
            values.append(match.group(1).strip() if match else "")
        result[field] = np.asarray(values, dtype=str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True)
    parser.add_argument("--item-ids", required=True)
    parser.add_argument("--item-texts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-levels", type=int, default=3)
    parser.add_argument("--codebook-sizes", type=int, nargs="+", required=True)
    args = parser.parse_args()

    codes = np.load(args.codes).astype(np.int64)
    item_ids = np.load(args.item_ids).astype(np.int64)
    levels = int(args.semantic_levels)
    sizes = [int(value) for value in args.codebook_sizes]
    if len(codes) != len(item_ids):
        raise ValueError("codes and item_ids must use the same catalog order")
    if levels != len(sizes) or codes.shape[1] < levels:
        raise ValueError("semantic levels and codebook sizes do not match codes")
    semantic_codes = codes[:, :levels]

    level_diagnostics = []
    for level, size in enumerate(sizes):
        if semantic_codes[:, level].min() < 0 or semantic_codes[:, level].max() >= size:
            raise ValueError(
                f"level {level + 1} contains tokens outside codebook size {size}"
            )
        counts = np.bincount(semantic_codes[:, level], minlength=size)
        used = int(np.count_nonzero(counts))
        probabilities = counts[counts > 0] / len(codes)
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        level_diagnostics.append(
            {
                "level": level + 1,
                "codebook_size": size,
                "used_tokens": used,
                "utilization": used / size,
                "normalized_entropy": entropy / math.log(size),
                "largest_bucket": int(counts.max()),
                "smallest_used_bucket": int(counts[counts > 0].min()),
            }
        )

    unique_prefixes = len(np.unique(semantic_codes, axis=0))
    unique_complete = len(np.unique(codes, axis=0))
    categories = _read_categories(Path(args.item_texts), item_ids)
    category_results = {
        field: analyze_category(labels, semantic_codes)
        for field, labels in categories.items()
    }
    result = {
        "items": int(len(codes)),
        "semantic_codebook_sizes": sizes,
        "complete_code_dimensions": int(codes.shape[1]),
        "per_level_codebook": level_diagnostics,
        "joint_prefix": {
            "theoretical_capacity": int(np.prod(sizes)),
            "unique_prefixes": int(unique_prefixes),
            "capacity_occupancy": unique_prefixes / int(np.prod(sizes)),
            "prefix_collision_rate": 1.0 - unique_prefixes / len(codes),
        },
        "complete_collision_rate": 1.0 - unique_complete / len(codes),
        "category_semantics": category_results,
        "interpretation": {
            "utilization_note": "Per-level 100% utilization is desirable; joint capacity cannot be fully occupied when capacity exceeds item count.",
            "semantic_note": "Useful SID semantics require lower within-category Hamming distance and higher shared-prefix rates than different-category pairs.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
