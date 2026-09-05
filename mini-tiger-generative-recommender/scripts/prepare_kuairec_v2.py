"""构建 MiniTIGER V2 的完整正负事件序列和 Sentence-T5 文本输入。"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_kuairec import (
    k_core_filter,
    limit_catalog,
    open_dataset_file,
    read_metadata,
    read_positive_events,
)


def _item_texts(source: Path, item_ids: np.ndarray) -> pd.DataFrame:
    captions = read_metadata(source, "kuairec_caption_category.csv")
    captions["video_id"] = pd.to_numeric(captions["video_id"], errors="coerce")
    captions = captions.dropna(subset=["video_id"])
    captions["video_id"] = captions["video_id"].astype(np.int64)
    captions = captions.drop_duplicates("video_id").set_index("video_id")
    categories = read_metadata(source, "item_categories.csv")
    category_map = categories.set_index("video_id")["feat"].to_dict()
    text_columns = [
        "manual_cover_text",
        "caption",
        "topic_tag",
        "first_level_category_name",
        "second_level_category_name",
        "third_level_category_name",
    ]
    rows = []
    for item in item_ids.tolist():
        parts = []
        if item in captions.index:
            row = captions.loc[item]
            for column in text_columns:
                value = row.get(column)
                if pd.notna(value) and str(value).upper() != "UNKNOWN":
                    parts.append(f"{column}: {value}")
        raw_tags = category_map.get(item)
        if pd.notna(raw_tags):
            try:
                parts.append("category_ids: " + " ".join(map(str, ast.literal_eval(str(raw_tags)))))
            except (ValueError, SyntaxError, TypeError):
                pass
        rows.append({"item_id": item, "text": " ; ".join(parts) or "无可用文本"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-watch-ratio", type=float, default=0.7)
    parser.add_argument("--min-play-seconds", type=float, default=5.0)
    parser.add_argument("--min-user-events", type=int, default=20)
    parser.add_argument("--min-item-events", type=int, default=5)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--chunk-size", type=int, default=500000)
    args = parser.parse_args()

    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    positive = read_positive_events(
        source,
        min_watch_ratio=args.min_watch_ratio,
        min_play_seconds=args.min_play_seconds,
        chunk_size=args.chunk_size,
    )
    positive = k_core_filter(
        positive,
        min_user_events=args.min_user_events,
        min_item_events=args.min_item_events,
    )
    positive = limit_catalog(
        positive,
        max_users=args.max_users,
        max_items=args.max_items,
        min_user_events=args.min_user_events,
        min_item_events=args.min_item_events,
    )
    users = set(positive["user_id"].unique().tolist())
    item_ids = np.sort(positive["video_id"].unique()).astype(np.int64)
    items = set(item_ids.tolist())
    chunks, retained = [], 0
    with open_dataset_file(source, "big_matrix.csv") as handle:
        reader = pd.read_csv(
            handle,
            usecols=[
                "user_id", "video_id", "timestamp", "play_duration", "watch_ratio"
            ],
            dtype={
                "user_id": "int32",
                "video_id": "int32",
                "timestamp": "float64",
                "play_duration": "float32",
                "watch_ratio": "float32",
            },
            chunksize=args.chunk_size,
        )
        for number, chunk in enumerate(reader, 1):
            chunk = chunk[
                chunk["user_id"].isin(users) & chunk["video_id"].isin(items)
            ].copy()
            chunk["is_positive"] = (
                chunk["watch_ratio"].ge(args.min_watch_ratio)
                & chunk["play_duration"].ge(args.min_play_seconds * 1000.0)
            ).astype(np.int8)
            chunks.append(
                chunk[["user_id", "video_id", "timestamp", "is_positive"]]
            )
            retained += len(chunk)
            print(f"full_event_chunk={number} retained={retained:,}", flush=True)
    events = pd.concat(chunks, ignore_index=True)
    events = events.sort_values(["user_id", "timestamp"]).rename(
        columns={"video_id": "item_id"}
    )
    events.to_csv(output / "interactions_full.csv", index=False)
    np.save(output / "item_ids.npy", item_ids)
    texts = _item_texts(source, item_ids)
    texts.to_csv(output / "item_texts.csv", index=False)
    per_user = events.groupby("user_id").agg(
        events=("item_id", "size"), positives=("is_positive", "sum")
    )
    stats = {
        "schema_version": "full-feedback-v2",
        "users": int(events["user_id"].nunique()),
        "items": int(len(item_ids)),
        "events": int(len(events)),
        "positive_events": int(events["is_positive"].sum()),
        "negative_events": int((events["is_positive"] == 0).sum()),
        "positive_definition": {
            "min_watch_ratio": args.min_watch_ratio,
            "min_play_seconds": args.min_play_seconds,
        },
        "per_user_events_mean": float(per_user["events"].mean()),
        "per_user_positive_mean": float(per_user["positives"].mean()),
    }
    (output / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
