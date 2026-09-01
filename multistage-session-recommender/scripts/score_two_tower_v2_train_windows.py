"""Use the frozen final Two-Tower V2 checkpoint to score all ranker training samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_two_tower_v2_windows import build_catalog
from src.recall.two_tower_v2 import TwoTowerV2, TwoTowerV2Dataset, collate_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--checkpoint", default="outputs/windows_two_tower_v2_validation/two_tower_v2.pt")
    parser.add_argument("--output", default="outputs/windows_two_tower_v2_train_candidates/two_tower_candidates.parquet")
    parser.add_argument("--samples-file", default="train_samples.parquet")
    parser.add_argument("--group-by", choices=["sample_id", "session"], default="sample_id")
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    processed = ROOT / args.processed
    checkpoint = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    item_ids = checkpoint["item_ids"].astype(np.int64)
    labels = pd.read_parquet(processed / "labels.parquet")
    cutoff = int(labels.loc[labels.split.eq("validation"), "target_ts"].min())
    item_to_row, features, _, _ = build_catalog(processed, cutoff, item_ids)
    pretrained = np.zeros((len(item_ids) + 1, 128), np.float32)
    model = TwoTowerV2(features, torch.from_numpy(pretrained))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()

    train = pd.read_parquet(processed / args.samples_file).sort_values(
        ["target_ts", "session"], kind="mergesort"
    ).reset_index(drop=True)
    train["sample_id"] = np.arange(len(train), dtype=np.int64)
    train["original_session"] = train["session"]
    if args.group_by == "sample_id": train["session"] = train["sample_id"]
    placeholder = np.tile(np.arange(1, 97, dtype=np.int32), (len(train), 1))
    dataset = TwoTowerV2Dataset(train, item_to_row, placeholder)

    catalog = []
    with torch.no_grad():
        for start in range(1, len(item_ids) + 1, 8192):
            indices = torch.arange(start, min(start + 8192, len(item_ids) + 1), device=args.device)
            catalog.append(model.encode_items(indices).cpu())
    vectors = torch.cat(catalog).numpy().astype("float32")
    import faiss
    faiss.omp_set_num_threads(1)
    ann = faiss.IndexHNSWFlat(64, 32, faiss.METRIC_INNER_PRODUCT)
    ann.hnsw.efConstruction = 120; ann.hnsw.efSearch = 128; ann.add(vectors)

    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    writer = None; rows = 0
    try:
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_v2, num_workers=0)
        with torch.no_grad():
            for batch_index, batch in enumerate(loader, 1):
                users = model.encode_user(
                    batch["history_items"].to(args.device), batch["history_types"].to(args.device),
                    batch["history_times"].to(args.device), batch["lengths"].to(args.device),
                ).cpu().numpy().astype("float32")
                scores, found = ann.search(np.ascontiguousarray(users), args.topk)
                sessions = np.repeat(batch["session"].numpy().astype(np.int64), args.topk)
                table = pa.table({
                    "session": sessions,
                    "aid": item_ids[found.reshape(-1)].astype(np.int64),
                    "source": pa.array(["two_tower"] * len(sessions)),
                    "source_rank": np.tile(np.arange(1, args.topk + 1, dtype=np.int16), len(users)),
                    "source_score": scores.reshape(-1).astype(np.float32),
                })
                if writer is None: writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                writer.write_table(table); rows += len(sessions)
                if batch_index % 50 == 0: print(f"scored={min(batch_index * args.batch_size, len(dataset))}/{len(dataset)}", flush=True)
    finally:
        if writer is not None: writer.close()
    print({"rows": rows, "samples": len(train), "topk": args.topk, "test_evaluated": False})


if __name__ == "__main__":
    main()
