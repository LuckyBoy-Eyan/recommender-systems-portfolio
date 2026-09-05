"""从 KuaiRec 连续响应、训练序列和内容特征构建独立物品 embedding。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_kuairec import open_dataset_file
from src.data.load import load_interactions
from src.data.split import temporal_leave_two_out
from src.embeddings import (
    build_transition_embedding,
    build_weighted_response_embedding,
    fuse_item_embeddings,
)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _training_cutoffs(interactions: pd.DataFrame) -> dict[int, float]:
    ordered = interactions.sort_values(["user_id", "timestamp"])
    cutoffs = {}
    for user, group in ordered.groupby("user_id", sort=False):
        if len(group) >= 3:
            cutoffs[int(user)] = float(group.iloc[-3]["timestamp"])
    return cutoffs


def _read_response_coordinates(
    source: Path,
    interactions_path: Path,
    item_ids: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    prepared = pd.read_csv(
        interactions_path,
        usecols=["user_id", "item_id", "timestamp"],
        dtype={"user_id": "int64", "item_id": "int64", "timestamp": "float64"},
    )
    cutoffs = _training_cutoffs(prepared)
    users = np.asarray(sorted(cutoffs), dtype=np.int64)
    user_to_column = {int(user): index for index, user in enumerate(users)}
    item_to_row = {int(item): index for index, item in enumerate(item_ids.tolist())}
    rows, columns, scores = [], [], []
    observed_rows = 0
    with open_dataset_file(source, "big_matrix.csv") as handle:
        chunks = pd.read_csv(
            handle,
            usecols=[
                "user_id",
                "video_id",
                "timestamp",
                "play_duration",
                "watch_ratio",
            ],
            dtype={
                "user_id": "int64",
                "video_id": "int64",
                "timestamp": "float64",
                "play_duration": "float32",
                "watch_ratio": "float32",
            },
            chunksize=chunk_size,
        )
        for chunk_number, chunk in enumerate(chunks, start=1):
            user_columns = chunk["user_id"].map(user_to_column)
            item_rows = chunk["video_id"].map(item_to_row)
            valid = user_columns.notna() & item_rows.notna()
            filtered = chunk.loc[valid]
            current_users = user_columns.loc[valid].astype(np.int64).to_numpy()
            current_items = item_rows.loc[valid].astype(np.int64).to_numpy()
            cutoff_values = np.asarray(
                [cutoffs[int(user)] for user in filtered["user_id"].to_numpy()],
                dtype=np.float64,
            )
            before_cutoff = filtered["timestamp"].to_numpy() <= cutoff_values
            if before_cutoff.any():
                watch = np.clip(
                    filtered["watch_ratio"].to_numpy(dtype=np.float32)[before_cutoff],
                    0.0,
                    3.0,
                )
                play_seconds = np.maximum(
                    filtered["play_duration"].to_numpy(dtype=np.float32)[before_cutoff]
                    / 1000.0,
                    0.0,
                )
                response = np.log1p(watch) * np.sqrt(
                    np.clip(play_seconds / 5.0, 0.2, 5.0)
                )
                rows.append(current_items[before_cutoff].astype(np.int32))
                columns.append(current_users[before_cutoff].astype(np.int32))
                scores.append(response.astype(np.float32))
                observed_rows += int(before_cutoff.sum())
            print(
                f"response_chunk={chunk_number} retained={observed_rows:,}",
                flush=True,
            )
    if not rows:
        raise ValueError("no raw responses aligned with the prepared catalog")
    return (
        np.concatenate(rows),
        np.concatenate(columns),
        np.concatenate(scores),
        {
            "users": int(len(users)),
            "items": int(len(item_ids)),
            "retained_raw_events": int(observed_rows),
            "cutoff": "per-user timestamp of the last positive training event",
            "response": (
                "log1p(clip(watch_ratio,0,3)) * "
                "sqrt(clip(play_seconds/5,0.2,5))"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="官方 KuaiRec ZIP 或目录")
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--item-features", required=True)
    parser.add_argument("--item-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--response-dim", type=int, default=64)
    parser.add_argument("--transition-dim", type=int, default=64)
    parser.add_argument("--content-dim", type=int, default=64)
    parser.add_argument("--transition-window", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--response-weight", type=float, default=0.45)
    parser.add_argument("--transition-weight", type=float, default=0.45)
    parser.add_argument("--content-weight", type=float, default=0.10)
    parser.add_argument("--chunk-size", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source = _resolve(args.source)
    interactions_path = _resolve(args.interactions)
    item_features_path = _resolve(args.item_features)
    item_ids = np.load(_resolve(args.item_ids)).astype(np.int64)
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows, columns, scores, response_source = _read_response_coordinates(
        source,
        interactions_path,
        item_ids,
        chunk_size=args.chunk_size,
    )
    response, response_diagnostics = build_weighted_response_embedding(
        rows,
        columns,
        scores,
        num_items=len(item_ids),
        num_users=response_source["users"],
        output_dim=args.response_dim,
        seed=args.seed,
    )

    _, sequences = load_interactions(
        interactions_path,
        item_features_path,
        min_sequence_length=5,
        max_sequence_length=args.max_sequence_length,
    )
    train_sequences, _, _ = temporal_leave_two_out(sequences, 3)
    transition, transition_diagnostics = build_transition_embedding(
        train_sequences,
        len(item_ids),
        output_dim=args.transition_dim,
        window=args.transition_window,
        seed=args.seed + 1,
    )

    prepared_features = np.load(item_features_path).astype(np.float32)
    content_path = item_features_path.with_name("content_features.npy")
    content = (
        np.load(content_path).astype(np.float32)
        if content_path.exists()
        else prepared_features[:, : args.content_dim]
    )
    content = content[:, : args.content_dim]
    fused, fusion_diagnostics = fuse_item_embeddings(
        {
            "response": response,
            "transition": transition,
            "content": content,
        },
        {
            "response": args.response_weight,
            "transition": args.transition_weight,
            "content": args.content_weight,
        },
    )

    np.save(output / "item_embeddings.npy", fused)
    np.save(output / "response_embeddings.npy", response)
    np.save(output / "transition_embeddings.npy", transition)
    np.save(output / "content_embeddings.npy", content)
    manifest = {
        "schema_version": "independent-item-embedding-v1",
        "sasrec_used": False,
        "source": response_source,
        "response": response_diagnostics,
        "transition": transition_diagnostics,
        "content": {"output_dim": int(content.shape[1])},
        "fusion": fusion_diagnostics,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
