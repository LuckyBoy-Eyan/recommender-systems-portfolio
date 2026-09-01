"""Score every candidate with all MMoE/PLE towers; no training or candidate truncation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ranking.neural import MMoE, PLE
from scripts.train_multitask_ranker import CATEGORICAL, encode_categories


# 累积塔的边际价值，对应行为总价值 clicks=.1/carts=.3/orders=.6。
FUSION_WEIGHTS = np.array([0.05, 0.30, 0.40], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["mmoe", "ple"], default="mmoe")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-rows", type=int, default=200000)
    parser.add_argument("--min-sample-id", type=int, default=None)
    parser.add_argument("--include-fusion-context", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    columns = checkpoint["feature_columns"]
    category_vocab, type_vocab = checkpoint["category_vocab"], checkpoint["type_vocab"]
    model_args = {"category_vocab_size": len(category_vocab) + 1, "type_vocab_size": len(type_vocab) + 1}
    model = MMoE(len(columns), **model_args) if args.model == "mmoe" else PLE(len(columns), **model_args)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    mean, std = checkpoint["mean"], checkpoint["std"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    context_columns = [
        "history_length", "history_unique_items", "session_span_minutes", "last_type_id",
        "source_score_hybrid_popular", "source_score_category", "candidate_age_days",
        "candidate_in_history",
    ] if args.include_fusion_context else []
    try:
        with torch.no_grad():
            for record in pq.ParquetFile(args.features).iter_batches(
                batch_size=args.batch_rows,
                columns=columns + CATEGORICAL + ["sample_id", "aid", "target_type", "label"] + [c for c in context_columns if c not in columns and c not in CATEGORICAL],
            ):
                frame = record.to_pandas()
                if args.min_sample_id is not None:
                    frame = frame[frame["sample_id"] >= args.min_sample_id]
                if frame.empty:
                    continue
                values = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
                tensor = torch.from_numpy((values - mean) / std).to(args.device)
                category, event_type = encode_categories(frame, category_vocab, type_vocab)
                category = torch.from_numpy(category).to(args.device); event_type = torch.from_numpy(event_type).to(args.device)
                probabilities = torch.sigmoid(model(tensor, category, event_type)).cpu().numpy()
                result = frame[["sample_id", "aid", "target_type", "label"] + context_columns].copy()
                result[["score_clicks", "score_carts", "score_orders"]] = probabilities
                result["final_score"] = probabilities @ FUSION_WEIGHTS
                table = pa.Table.from_pandas(result, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    print({"output": str(output), "test_evaluated": False, "training_performed": False})


if __name__ == "__main__":
    main()
