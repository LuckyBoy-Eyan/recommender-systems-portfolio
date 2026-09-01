"""Streaming MMoE/PLE trainer with one deployable, action-agnostic Top-20 list."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ranking.neural import MMoE, PLE

ACTIONS = {"clicks": 0, "carts": 1, "orders": 2}
TASK_LABELS = ["label_clicks", "label_carts", "label_orders"]
CATEGORICAL = ["last_categoryid", "last_type_id"]
# 累积概率使用边际价值；评估仍按真实行为价值加权。
FUSION_WEIGHTS = np.array([0.05, 0.30, 0.40], dtype=np.float32)
ACTION_WEIGHTS = np.array([0.1, 0.3, 0.6], dtype=np.float32)
EXCLUDED = {
    "sample_id", "session", "aid", "target_aid", "last_aid", "categoryid",
    "root_categoryid", "target_type", "label", "label_clicks", "label_carts",
    "label_orders", "fold_id", "snapshot_ts", "target_ts", "category_state_ts",
    "availability_state_ts", "first_seen_ts",
}


def feature_columns(path: Path) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow
    return [field.name for field in schema if field.name not in EXCLUDED and field.name not in CATEGORICAL and not str(field.type).startswith("string")]


def fit_vocabularies(paths: list[Path], max_sample_id: int | None = None) -> tuple[dict[int, int], dict[int, int]]:
    values = {column: set() for column in CATEGORICAL}
    for path in paths:
        read_columns = CATEGORICAL + (["sample_id"] if max_sample_id is not None else [])
        for batch in pq.ParquetFile(path).iter_batches(batch_size=200000, columns=read_columns):
            frame = batch.to_pandas()
            if max_sample_id is not None: frame = frame[frame.sample_id < max_sample_id]
            for column in CATEGORICAL:
                values[column].update(frame[column].dropna().astype(np.int64).tolist())
    mappings = [{value: index + 1 for index, value in enumerate(sorted(values[column]))} for column in CATEGORICAL]
    return mappings[0], mappings[1]


def encode_categories(frame, category_vocab, type_vocab):
    category = frame["last_categoryid"].fillna(-1).astype(np.int64).map(category_vocab).fillna(0).to_numpy(np.int64)
    event_type = frame["last_type_id"].fillna(-1).astype(np.int64).map(type_vocab).fillna(0).to_numpy(np.int64)
    return category, event_type


def fit_statistics(paths: list[Path], columns: list[str], batch_rows: int, max_sample_id: int | None = None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    total = np.zeros(len(columns), dtype=np.float64)
    squares = np.zeros(len(columns), dtype=np.float64)
    count = 0
    positives = np.zeros(3, dtype=np.int64)
    observed = np.zeros(3, dtype=np.int64)
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=batch_rows, columns=columns + TASK_LABELS + (["sample_id"] if max_sample_id is not None else [])
        ):
            frame = batch.to_pandas()
            if max_sample_id is not None: frame = frame[frame.sample_id < max_sample_id]
            values = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float64)
            total += values.sum(0); squares += np.square(values).sum(0); count += len(values)
            for task in range(3):
                label = frame[TASK_LABELS[task]].to_numpy()
                observed[task] += len(label)
                positives[task] += int(label.sum())
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - np.square(mean), 1e-8))
    imbalance = (observed - positives) / np.maximum(positives, 1)
    weights = np.minimum(50.0, np.sqrt(imbalance)).astype(float).tolist()
    return mean.astype(np.float32), std.astype(np.float32), weights


def iter_batches(paths, columns, mean, std, category_vocab, type_vocab, batch_rows, batch_size, seed, max_sample_id=None, negative_keep_probability=1.0):
    rng = np.random.default_rng(seed)
    for path in paths:
        for record in pq.ParquetFile(path).iter_batches(
            batch_size=batch_rows, columns=columns + CATEGORICAL + TASK_LABELS + ["sample_id", "aid"]
        ):
            frame = record.to_pandas()
            if max_sample_id is not None: frame = frame[frame.sample_id < max_sample_id]
            if negative_keep_probability < 1.0:
                positive = frame[TASK_LABELS].any(axis=1).to_numpy()
                # Epoch-varying deterministic sampling from the complete pool; independent of RRF.
                mixed = (
                    frame["sample_id"].to_numpy(np.uint64) * np.uint64(11400714819323198485)
                    + frame["aid"].to_numpy(np.uint64) * np.uint64(14029467366897019727)
                    + np.uint64(seed) * np.uint64(1609587929392839161)
                )
                uniform = (mixed >> np.uint64(11)).astype(np.float64) / float(1 << 53)
                frame = frame[positive | (uniform < negative_keep_probability)]
            x = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
            x = (x - mean) / std
            labels = frame[TASK_LABELS].to_numpy(np.float32)
            category, event_type = encode_categories(frame, category_vocab, type_vocab)
            order = rng.permutation(len(frame))
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                yield x[idx], category[idx], event_type[idx], labels[idx]


def evaluate(model, path, columns, mean, std, device, category_vocab, type_vocab, batch_rows=200000, min_sample_id=None) -> dict:
    parts = []
    model.eval()
    with torch.no_grad():
        for record in pq.ParquetFile(path).iter_batches(
            batch_size=batch_rows,
            columns=columns + CATEGORICAL + ["sample_id", "aid", "target_type", "label"],
        ):
            frame = record.to_pandas()
            if min_sample_id is not None: frame = frame[frame.sample_id >= min_sample_id]
            if frame.empty: continue
            x = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
            x = torch.from_numpy((x - mean) / std).to(device)
            category, event_type = encode_categories(frame, category_vocab, type_vocab)
            category = torch.from_numpy(category).to(device); event_type = torch.from_numpy(event_type).to(device)
            probabilities = torch.sigmoid(model(x, category, event_type)).cpu().numpy()
            frame = frame[["sample_id", "aid", "target_type", "label"]]
            frame["final_score"] = probabilities @ FUSION_WEIGHTS
            parts.append(frame)
    scored = pd.concat(parts, ignore_index=True).sort_values(
        ["sample_id", "final_score", "aid"], ascending=[True, False, True]
    )
    top = scored.groupby("sample_id", sort=False).head(20)
    hits = top.groupby("sample_id")["label"].max()
    all_samples = scored[["sample_id", "target_type"]].drop_duplicates("sample_id")
    hit_map = all_samples["sample_id"].map(hits).fillna(0).to_numpy()
    result = {"recall_at_20": float(hit_map.mean())}
    scored["rank"] = scored.groupby("sample_id").cumcount() + 1
    positives = scored[scored["label"].eq(1)][["sample_id", "rank", "final_score"]]
    positive_rank = dict(zip(positives["sample_id"], positives["rank"]))
    ranks = all_samples["sample_id"].map(positive_rank).fillna(np.inf).to_numpy()
    result["mrr_at_20"] = float(np.where(ranks <= 20, 1.0 / ranks, 0.0).mean())
    result["ndcg_at_20"] = float(
        np.where(ranks <= 20, 1.0 / np.log2(ranks + 1.0), 0.0).mean()
    )
    for action in ACTIONS:
        mask = all_samples["target_type"].eq(action).to_numpy()
        result[f"{action}_recall_at_20"] = float(hit_map[mask].mean())
    result["weighted_recall_at_20"] = float(sum(
        weight * result[f"{action}_recall_at_20"]
        for weight, action in zip(ACTION_WEIGHTS, ACTIONS)
    ))
    positive_score = dict(zip(positives["sample_id"], positives["final_score"]))
    negatives = scored[scored["label"].eq(0)].copy()
    negatives["positive_score"] = negatives["sample_id"].map(positive_score)
    eligible = negatives["positive_score"].notna()
    comparisons = (
        (negatives.loc[eligible, "positive_score"] > negatives.loc[eligible, "final_score"]).astype(float)
        + 0.5 * (negatives.loc[eligible, "positive_score"] == negatives.loc[eligible, "final_score"]).astype(float)
    )
    negatives.loc[eligible, "pair_auc"] = comparisons
    session_auc = negatives.loc[eligible].groupby("sample_id")["pair_auc"].mean()
    result["candidate_auc"] = float(comparisons.mean()) if len(comparisons) else None
    result["session_gauc"] = float(session_auc.mean()) if len(session_auc) else None
    result["auc_eligible_sessions"] = int(len(session_auc))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="outputs/ranker_datasets/features")
    parser.add_argument("--output", default="outputs/ple_validation")
    parser.add_argument("--model", choices=["mmoe", "ple"], default="ple")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0002)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--batch-rows", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--negatives-per-epoch", type=int, default=160)
    parser.add_argument("--average-candidates", type=float, default=783.2108685)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    negative_keep_probability = min(1.0, args.negatives_per_epoch / max(args.average_candidates - 1.0, 1.0))
    root = ROOT / args.features; output = ROOT / args.output; output.mkdir(parents=True, exist_ok=True)
    direct_train = root / "train.parquet"
    if direct_train.exists():
        train_paths = [direct_train]; early_path = direct_train
        max_sample_id = max(
            pq.ParquetFile(direct_train).metadata.row_group(i).column(
                pq.ParquetFile(direct_train).schema_arrow.get_field_index("sample_id")
            ).statistics.max
            for i in range(pq.ParquetFile(direct_train).metadata.num_row_groups)
        )
        early_split = int((int(max_sample_id) + 1) * 0.9)
    else:
        train_paths = [root / f"train_fold_{fold}.parquet" for fold in range(3)]
        early_path = root / "train_fold_3.parquet"; early_split = None
    validation_path = root / "validation.parquet"
    columns = feature_columns(train_paths[0])
    category_vocab, type_vocab = fit_vocabularies(train_paths, early_split)
    mean, std, pos_weights = fit_statistics(train_paths, columns, args.batch_rows, early_split)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto": device = "cpu"
    torch.manual_seed(args.seed)
    model_args = {"category_vocab_size": len(category_vocab) + 1, "type_vocab_size": len(type_vocab) + 1}
    model = (PLE(len(columns), **model_args) if args.model == "ple" else MMoE(len(columns), **model_args)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    weights = torch.tensor(pos_weights, device=device)
    best, patience_best, best_epoch = -1.0, -1.0, 0
    best_state, stale, history = None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, category, event_type, labels in iter_batches(
            train_paths, columns, mean, std, category_vocab, type_vocab, args.batch_rows, args.batch_size, args.seed + epoch, early_split, negative_keep_probability
        ):
            x = torch.from_numpy(x).to(device); category = torch.from_numpy(category).to(device); event_type = torch.from_numpy(event_type).to(device); labels = torch.from_numpy(labels).to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(x, category, event_type)
                element_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                element_weight = torch.where(labels > 0, weights.unsqueeze(0), torch.ones_like(labels))
                loss = (element_loss * element_weight).mean()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, early_path, columns, mean, std, device, category_vocab, type_vocab, args.batch_rows, early_split)
        metrics.update({"epoch": epoch, "loss": float(np.mean(losses))}); history.append(metrics); print(metrics, flush=True)
        score = metrics["weighted_recall_at_20"]
        # 始终保存真实最高分；min_delta 只控制耐心计数，不能让较优模型被丢弃。
        if score > best:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if score > patience_best + args.min_delta:
            patience_best, stale = score, 0
        else:
            stale += 1
        if epoch >= 4 and stale >= args.patience: break
    model.load_state_dict(best_state)
    final = evaluate(model, validation_path, columns, mean, std, device, category_vocab, type_vocab, args.batch_rows)
    report = {"model": args.model, "feature_columns": columns, "positive_weights": pos_weights,
              "task_supervision": "joint_cumulative_ordinal_all_towers_observed",
              "fusion_weights": FUSION_WEIGHTS.tolist(),
              "action_metric_weights": ACTION_WEIGHTS.tolist(),
              "early_stopping_sample_id_split": early_split,
              "negative_sampling": {"method": "epoch_varying_uniform_hash_full_pool", "rrf_used": False,
                                    "target_negatives_per_sample": args.negatives_per_epoch,
                                    "keep_probability": negative_keep_probability},
              "history": history, "best_epoch": best_epoch, "stopped_epoch": history[-1]["epoch"],
              "best_internal_weighted_recall": best, "validation": final,
              "test_evaluated": False}
    torch.save({"state_dict": model.cpu().state_dict(), "mean": mean, "std": std,
                "feature_columns": columns, "category_vocab": category_vocab, "type_vocab": type_vocab,
                "category_embedding_dim": 16, "type_embedding_dim": 8,
                "config": vars(args)}, output / f"{args.model}.pt")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(final, ensure_ascii=False))

if __name__ == "__main__": main()
