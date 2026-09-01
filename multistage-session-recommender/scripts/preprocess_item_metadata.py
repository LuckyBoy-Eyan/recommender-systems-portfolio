"""分块提取 RetailRocket 时序类目、可用状态，并构建类目路径。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_state_changes(
    frame: pd.DataFrame, value_column: str
) -> tuple[pd.DataFrame, dict]:
    """删除完全重复、同刻冲突和连续不变的状态记录。"""
    input_rows = len(frame)
    frame = frame.drop_duplicates(["timestamp", "itemid", value_column]).copy()
    exact_duplicates = input_rows - len(frame)
    conflict = frame.duplicated(["timestamp", "itemid"], keep=False)
    conflicting_rows = int(conflict.sum())
    frame = frame[~conflict].sort_values(["itemid", "timestamp"], kind="mergesort")
    repeated = frame["itemid"].eq(frame["itemid"].shift()) & frame[value_column].eq(
        frame[value_column].shift()
    )
    consecutive_repeats = int(repeated.sum())
    frame = frame[~repeated].reset_index(drop=True)
    return frame, {
        "input_rows": int(input_rows),
        "exact_duplicates": int(exact_duplicates),
        "conflicting_rows_removed": conflicting_rows,
        "consecutive_repeats_removed": consecutive_repeats,
        "output_rows": int(len(frame)),
        "items": int(frame["itemid"].nunique()),
    }


def extract_item_states(
    paths: list[Path], chunk_size: int = 1_000_000
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """一次扫描两个属性文件，仅保留 categoryid 和 available。"""
    categories = []
    availability = []
    property_counts: Counter[str] = Counter()
    total_rows = 0
    for path in paths:
        for chunk in pd.read_csv(
            path,
            usecols=["timestamp", "itemid", "property", "value"],
            chunksize=chunk_size,
            dtype={
                "timestamp": "int64",
                "itemid": "int64",
                "property": "string",
                "value": "string",
            },
        ):
            total_rows += len(chunk)
            property_counts.update(chunk["property"].value_counts().to_dict())
            category = chunk[chunk["property"].eq("categoryid")][
                ["timestamp", "itemid", "value"]
            ].copy()
            category["categoryid"] = pd.to_numeric(
                category.pop("value"), errors="coerce"
            )
            categories.append(category.dropna(subset=["categoryid"]))
            available = chunk[chunk["property"].eq("available")][
                ["timestamp", "itemid", "value"]
            ].copy()
            available["available"] = pd.to_numeric(
                available.pop("value"), errors="coerce"
            )
            availability.append(available[available["available"].isin([0, 1])])
    category_frame = pd.concat(categories, ignore_index=True)
    category_frame["categoryid"] = category_frame["categoryid"].astype("int64")
    availability_frame = pd.concat(availability, ignore_index=True)
    availability_frame["available"] = availability_frame["available"].astype("int8")
    category_frame, category_report = clean_state_changes(category_frame, "categoryid")
    availability_frame, availability_report = clean_state_changes(
        availability_frame, "available"
    )
    return category_frame, availability_frame, {
        "total_property_rows": int(total_rows),
        "unique_properties": int(len(property_counts)),
        "top_properties": {
            key: int(value) for key, value in property_counts.most_common(20)
        },
        "category": category_report,
        "availability": availability_report,
    }


def build_category_paths(
    tree: pd.DataFrame, observed_categories: set[int]
) -> tuple[pd.DataFrame, dict]:
    """为树中及商品实际出现的类目生成从根到叶的路径。"""
    if tree["categoryid"].duplicated().any():
        raise ValueError("类目树存在重复 categoryid")
    parent = {
        int(row.categoryid): None if pd.isna(row.parentid) else int(row.parentid)
        for row in tree.itertuples(index=False)
    }
    all_categories = set(parent) | observed_categories
    missing_parent_links = 0
    rows = []
    for category in sorted(all_categories):
        path = []
        seen = set()
        current: int | None = category
        while current is not None:
            if current in seen:
                raise ValueError(f"类目树存在循环: {category}")
            seen.add(current)
            path.append(current)
            if current not in parent:
                if current != category or category not in parent:
                    missing_parent_links += 1
                break
            current = parent[current]
        path.reverse()
        rows.append(
            {
                "categoryid": category,
                "root_categoryid": path[0],
                "category_depth": len(path) - 1,
                "category_path": path,
                "in_tree": category in parent,
            }
        )
    return pd.DataFrame(rows), {
        "categories": len(rows),
        "roots": len({row["root_categoryid"] for row in rows}),
        "max_depth": max(row["category_depth"] for row in rows),
        "observed_categories": len(observed_categories),
        "observed_categories_missing_from_tree": len(observed_categories - set(parent)),
        "missing_parent_links": missing_parent_links,
    }


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="提取 RetailRocket 商品时序元数据")
    parser.add_argument("--properties", nargs=2, required=True)
    parser.add_argument("--category-tree", required=True)
    parser.add_argument("--output", default="data/processed/retailrocket")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()
    started = time.perf_counter()
    property_paths = [Path(value) for value in args.properties]
    tree_path = Path(args.category_tree)
    output = Path(args.output)
    category, availability, property_report = extract_item_states(
        property_paths, args.chunk_size
    )
    raw_tree = pd.read_csv(tree_path)
    paths, tree_report = build_category_paths(
        raw_tree, set(category["categoryid"].astype(int))
    )
    write_parquet_atomic(category, output / "item_category_changes.parquet")
    write_parquet_atomic(availability, output / "item_availability_changes.parquet")
    write_parquet_atomic(paths, output / "category_paths.parquet")
    report = {
        "status": "completed",
        "sources": {
            str(path): sha256(path) for path in [*property_paths, tree_path]
        },
        "property_report": property_report,
        "category_tree_report": tree_report,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "item_metadata_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
