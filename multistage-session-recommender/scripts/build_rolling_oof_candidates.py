"""Build expanding-window, point-in-time OOF candidates for ranker training."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import candidate_recall
from src.recall.full_catalog import build_category_popularity, build_directional_transitions, build_frozen_indexes, build_hybrid_popularity, latest_category_map, recall_from_frozen_indexes
from src.recall.item2vec_ann import Item2VecANN, Item2VecEmbeddings, train_item2vec_embeddings
from src.recall.two_tower import (
    SequenceDataset, TwoTowerModel, TwoTowerVocabulary, attach_target_categories,
    collate_sequences, train_two_tower,
)


def temporal_folds(samples: pd.DataFrame, folds: int, warmup_fraction: float) -> list[tuple[int, int]]:
    if folds < 1 or not 0.0 < warmup_fraction < 1.0:
        raise ValueError("folds must be positive and warmup_fraction must be between 0 and 1")
    quantiles = np.linspace(warmup_fraction, 1.0, folds + 1)
    boundaries = [int(samples["target_ts"].quantile(float(q))) for q in quantiles]
    boundaries[-1] += 1
    return list(zip(boundaries[:-1], boundaries[1:]))


def build_vocabulary(
    first_seen: pd.DataFrame, events: pd.DataFrame, cutoff: int
) -> TwoTowerVocabulary:
    item_ids = np.sort(
        first_seen.loc[first_seen["first_seen_ts"] < cutoff, "aid"].astype(int).unique()
    )
    visible = events[events["ts"] < cutoff]
    return TwoTowerVocabulary(
        item_ids,
        np.sort(visible["categoryid"].astype(int).unique()),
        np.sort(visible["root_categoryid"].astype(int).unique()),
    )


def warm_start(model: TwoTowerModel, vocabulary: TwoTowerVocabulary, embeddings: Item2VecEmbeddings) -> int:
    if embeddings.vectors.shape[1] != model.item_embedding.embedding_dim:
        return 0
    count = 0
    with torch.no_grad():
        for aid, vector in zip(embeddings.item_ids, embeddings.vectors):
            index = vocabulary.item_to_index.get(int(aid))
            if index is not None:
                model.item_embedding.weight[index].copy_(torch.from_numpy(vector))
                count += 1
    return count


def prepare_tower_ann(
    model: TwoTowerModel,
    vocabulary: TwoTowerVocabulary,
    events: pd.DataFrame,
    category_changes: pd.DataFrame,
    cutoff: int,
    device: str,
) -> tuple[np.ndarray, object]:
    item_ids = vocabulary.item_ids
    category_map = latest_category_map(category_changes, cutoff)
    latest = events[events["ts"] < cutoff].sort_values(["aid", "ts"]).groupby("aid").tail(1)
    root_map = dict(zip(latest["aid"].astype(int), latest["root_categoryid"].astype(int)))
    item_indices = torch.tensor(
        [vocabulary.item_to_index[int(aid)] for aid in item_ids], dtype=torch.long, device=device
    )
    categories = torch.tensor(
        [vocabulary.category_to_index.get(category_map.get(int(aid), -1), 0) for aid in item_ids],
        dtype=torch.long, device=device,
    )
    roots = torch.tensor(
        [vocabulary.root_to_index.get(root_map.get(int(aid), -1), 0) for aid in item_ids],
        dtype=torch.long, device=device,
    )
    model.eval()
    with torch.no_grad():
        catalog_vectors = model.encode_item(item_indices, categories, roots).cpu().numpy().astype("float32")
    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(catalog_vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 80
    index.add(np.ascontiguousarray(catalog_vectors))
    return item_ids, index


def tower_recall(
    model: TwoTowerModel,
    samples: pd.DataFrame,
    vocabulary: TwoTowerVocabulary,
    item_ids: np.ndarray,
    index: object,
    topk: int,
    device: str,
) -> pd.DataFrame:
    dataset = SequenceDataset(samples, vocabulary)
    loader = DataLoader(dataset, batch_size=1024, collate_fn=collate_sequences)
    rows = []
    with torch.no_grad():
        for batch in loader:
            users = model.encode_user(
                batch["history_items"].to(device), batch["history_types"].to(device),
                batch["history_times"].to(device), batch["lengths"].to(device),
            ).cpu().numpy().astype("float32")
            scores, indices = index.search(np.ascontiguousarray(users), topk)
            for session, found, values in zip(batch["session"].numpy(), indices, scores):
                rows.extend(
                    (int(session), int(item_ids[i]), "two_tower", rank, float(score))
                    for rank, (i, score) in enumerate(zip(found, values), 1) if i >= 0
                )
    return pd.DataFrame(rows, columns=["session", "aid", "source", "source_rank", "source_score"])


def compact_candidates(
    candidates: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    max_negatives: int,
    topk: int,
    inject_missing_positives: bool = True,
    prioritize_positives: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Deduplicate routes, inject positives, and retain bounded hard negatives."""
    target_map = dict(zip(samples["session"].astype(int), samples["target_aid"].astype(int)))
    target_pairs = {(session, aid) for session, aid in target_map.items()}
    raw_pairs = set(zip(candidates["session"].astype(int), candidates["aid"].astype(int)))
    recalled_hits = len(target_pairs & raw_pairs)
    candidates = candidates.copy()
    candidates["rrf_score"] = 1.0 / (60.0 + candidates["source_rank"].astype(float))
    summary = candidates.groupby(["session", "aid"], as_index=False).agg(
        rrf_score=("rrf_score", "sum"),
        source_count=("source", "nunique"),
        best_source_rank=("source_rank", "min"),
    )
    ranks = candidates.pivot_table(
        index=["session", "aid"], columns="source", values="source_rank", aggfunc="min"
    ).add_prefix("source_rank_").reset_index()
    scores = candidates.pivot_table(
        index=["session", "aid"], columns="source", values="source_score", aggfunc="max"
    ).add_prefix("source_score_").reset_index()
    compact = summary.merge(ranks, on=["session", "aid"], how="left").merge(
        scores, on=["session", "aid"], how="left"
    )
    compact["label"] = [int(target_map[int(s)] == int(a)) for s, a in zip(compact.session, compact.aid)]
    present_positive_sessions = set(compact.loc[compact["label"].eq(1), "session"].astype(int))
    missing = [
        {"session": int(session), "aid": int(aid), "rrf_score": 0.0,
         "source_count": 0, "best_source_rank": topk + 1, "label": 1}
        for session, aid in target_map.items() if session not in present_positive_sessions
    ]
    if missing and inject_missing_positives:
        compact = pd.concat([compact, pd.DataFrame(missing)], ignore_index=True)
    if prioritize_positives:
        compact = compact.sort_values(
            ["session", "label", "rrf_score", "source_count", "aid"],
            ascending=[True, False, False, False, True], kind="mergesort",
        )
        if max_negatives > 0:
            compact["negative_rank"] = compact["label"].eq(0).groupby(compact["session"]).cumsum()
            compact = compact[
                compact["label"].eq(1) | compact["negative_rank"].le(max_negatives)
            ].drop(columns="negative_rank").reset_index(drop=True)
    else:
        compact = compact.sort_values(
            ["session", "rrf_score", "source_count", "aid"],
            ascending=[True, False, False, True], kind="mergesort",
        )
        if max_negatives > 0:
            compact = compact.groupby("session", sort=False).head(max_negatives).reset_index(drop=True)
    compact["target_type"] = compact["session"].map(
        dict(zip(samples["session"].astype(int), samples["target_type"].astype(str)))
    )
    return compact, recalled_hits


def main() -> None:
    import joblib

    parser = argparse.ArgumentParser(description="生成训练期滚动 Point-in-Time OOF 候选")
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--output", default="outputs/rolling_oof_candidates")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--item2vec-topk", type=int, default=250)
    parser.add_argument("--tower-topk", type=int, default=300)
    parser.add_argument("--item2vec-epochs", type=int, default=10)
    parser.add_argument("--tower-epochs", type=int, default=2)
    parser.add_argument("--skip-two-tower", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--item2vec-dim", type=int, default=128)
    parser.add_argument("--item2vec-negative-samples", type=int, default=15)
    parser.add_argument("--item2vec-min-count", type=int, default=5)
    parser.add_argument("--item2vec-subsample", type=float, default=3e-5)
    parser.add_argument("--max-fit-samples", type=int, default=0)
    parser.add_argument("--max-score-samples", type=int, default=0)
    parser.add_argument("--score-chunk-size", type=int, default=5000)
    parser.add_argument("--max-negatives", type=int, default=60)
    parser.add_argument("--popularity-window", choices=["1d", "7d", "30d", "all"], default="30d")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    started = time.perf_counter()
    processed = ROOT / args.processed
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "fold_cache"
    cache.mkdir(exist_ok=True)

    samples = pd.read_parquet(processed / "train_samples.parquet")
    events = pd.read_parquet(processed / "events_enriched.parquet")
    category_changes = pd.read_parquet(processed / "item_category_changes.parquet")
    first_seen = pd.read_parquet(processed / "item_first_seen.parquet")
    samples = attach_target_categories(samples, events)
    # Sliding prefixes create multiple labels per original Session. All candidate
    # generation/evaluation therefore uses a stable sample_id as the ranking group.
    samples = samples.sort_values(["target_ts", "session"], kind="mergesort").reset_index(drop=True)
    samples["sample_id"] = np.arange(len(samples), dtype=np.int64)
    folds = temporal_folds(samples, args.folds, args.warmup_fraction)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法访问 CUDA GPU")
    fold_reports = []
    total_scored = 0
    total_recalled_hits = 0

    for fold_id, (cutoff, end) in enumerate(folds):
        print(f"[fold {fold_id + 1}/{len(folds)}] cutoff={cutoff} end={end}", flush=True)
        fold_started = time.perf_counter()
        fold_path = output / f"fold_{fold_id}_candidates.parquet"
        report_path = output / f"fold_{fold_id}_report.json"
        if fold_path.exists() and report_path.exists():
            cached_report = json.loads(report_path.read_text())
            fold_reports.append(cached_report)
            total_scored += int(cached_report["score_samples"])
            total_recalled_hits += int(cached_report["recalled_hits_before_positive_injection"])
            continue
        fit = samples[samples["target_ts"] < cutoff]
        score = samples[(samples["target_ts"] >= cutoff) & (samples["target_ts"] < end)]
        if args.max_fit_samples and len(fit) > args.max_fit_samples:
            fit = fit.sample(args.max_fit_samples, random_state=args.seed + fold_id).sort_values("target_ts")
        if args.max_score_samples and len(score) > args.max_score_samples:
            score = score.sample(args.max_score_samples, random_state=args.seed + fold_id).sort_values("target_ts")
        vocabulary = build_vocabulary(first_seen, events, cutoff)
        fit = fit[fit["target_aid"].isin(vocabulary.item_to_index)].reset_index(drop=True)
        score = score.reset_index(drop=True)
        score_for_recall = score.copy()
        original_session = dict(zip(score["sample_id"].astype(int), score["session"].astype(int)))
        score_for_recall["session"] = score_for_recall["sample_id"]
        reference = events[events["ts"] < cutoff]

        index_path = cache / f"fold_{fold_id}_heuristic.joblib"
        if index_path.exists():
            indexes = joblib.load(index_path)
            if indexes.get("index_version", 0) < 2:
                indexes = build_frozen_indexes(events, events, category_changes, cutoff)
                joblib.dump(indexes, index_path, compress=3)
            elif indexes.get("index_version", 0) < 3:
                indexes["category_popularity"] = build_category_popularity(
                    events[events["ts"] < cutoff], cutoff
                )
                indexes["index_version"] = 3
                joblib.dump(indexes, index_path, compress=3)
            if indexes.get("index_version", 0) < 4:
                indexes["transition"] = build_directional_transitions(events[events["ts"] < cutoff])
                indexes["index_version"] = 4
                joblib.dump(indexes, index_path, compress=3)
            if indexes.get("index_version", 0) < 5:
                indexes["hybrid_popularity"] = build_hybrid_popularity(events[events["ts"] < cutoff], cutoff)
                indexes["index_version"] = 5
                joblib.dump(indexes, index_path, compress=3)
        else:
            indexes = build_frozen_indexes(events, events, category_changes, cutoff)
            joblib.dump(indexes, index_path, compress=3)
        recall_indexes = dict(indexes)
        embedding_path = cache / f"fold_{fold_id}_item2vec_v2.npz"
        if embedding_path.exists():
            embeddings = Item2VecEmbeddings.load(embedding_path)
        else:
            embeddings = train_item2vec_embeddings(
                reference, dimensions=args.item2vec_dim, window=10,
                negative_samples=args.item2vec_negative_samples,
                epochs=args.item2vec_epochs, batch_size=4096,
                min_count=args.item2vec_min_count,
                subsample=args.item2vec_subsample, adaptive_window=True,
                seed=args.seed + fold_id,
            )
            embeddings.save(embedding_path)
        item2vec_ann = Item2VecANN(embeddings)

        model = tower_item_ids = tower_index = None
        initialized, losses = 0, []
        if not args.skip_two_tower:
            torch.manual_seed(args.seed + fold_id)
            model = TwoTowerModel(
                len(vocabulary.item_ids), len(vocabulary.category_ids),
                len(vocabulary.root_ids), args.embedding_dim,
            )
            initialized = warm_start(model, vocabulary, embeddings)
            losses = train_two_tower(
                model, SequenceDataset(fit, vocabulary), epochs=args.tower_epochs,
                batch_size=args.batch_size, seed=args.seed + fold_id, device=device,
            )
            tower_item_ids, tower_index = prepare_tower_ann(
                model, vocabulary, events, category_changes, cutoff, device
            )
        compact_parts = []
        recalled_hits = 0
        for chunk_start in range(0, len(score_for_recall), args.score_chunk_size):
            chunk = score_for_recall.iloc[chunk_start : chunk_start + args.score_chunk_size]
            routes = [
                recall_from_frozen_indexes(chunk, recall_indexes),
                item2vec_ann.recall(chunk, args.item2vec_topk),
            ]
            if not args.skip_two_tower:
                routes.append(tower_recall(
                        model, chunk, vocabulary, tower_item_ids, tower_index,
                        args.tower_topk, device,
                    ))
            raw = pd.concat(routes, ignore_index=True)
            compact, chunk_hits = compact_candidates(
                raw, chunk, max_negatives=args.max_negatives,
                topk=max(args.item2vec_topk, args.tower_topk, 200),
            )
            compact_parts.append(compact)
            recalled_hits += chunk_hits
            del raw
            if chunk_start == 0 or chunk_start + args.score_chunk_size >= len(score_for_recall):
                print(
                    f"[fold {fold_id + 1}] scored {min(chunk_start + args.score_chunk_size, len(score_for_recall))}/{len(score_for_recall)}",
                    flush=True,
                )
        candidates = pd.concat(compact_parts, ignore_index=True)
        candidates["fold_id"] = fold_id
        candidates["snapshot_ts"] = cutoff
        candidates = candidates.rename(columns={"session": "sample_id"})
        candidates["session"] = candidates["sample_id"].map(original_session).astype(int)
        candidates.to_parquet(fold_path, index=False)
        report = {
            "fold_id": fold_id, "cutoff_ts": cutoff, "end_ts": end,
            "fit_samples": len(fit), "score_samples": len(score),
            "reference_events": len(reference), "catalog_items": len(vocabulary.item_ids),
            "max_fit_target_ts": int(fit["target_ts"].max()),
            "min_score_target_ts": int(score["target_ts"].min()),
            "causal_audit_passed": bool(
                fit["target_ts"].max() < cutoff <= score["target_ts"].min()
                and reference["ts"].max() < cutoff
            ),
            "tower_losses": losses, "item2vec_warm_started_items": initialized,
            "recalled_hits_before_positive_injection": recalled_hits,
            "candidate_recall_before_positive_injection": recalled_hits / len(score),
            "compacted_rows": len(candidates),
            "max_negatives_per_sample": args.max_negatives,
            "runtime_seconds": time.perf_counter() - fold_started,
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(
            f"[fold {fold_id + 1}] recall={report['candidate_recall_before_positive_injection']:.6f} rows={len(candidates)}",
            flush=True,
        )
        fold_reports.append(report)
        total_scored += len(score)
        total_recalled_hits += recalled_hits
        del indexes, embeddings, model, candidates
        gc.collect()

    manifest = {
        "protocol": "expanding_window_point_in_time_oof",
        "folds": fold_reports,
        "candidate_files": [f"fold_{fold_id}_candidates.parquet" for fold_id in range(len(folds))],
        "covered_samples": total_scored,
        "coverage_of_train_samples": total_scored / len(samples),
        "candidate_recall_before_positive_injection": total_recalled_hits / total_scored,
        "test_evaluated": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
