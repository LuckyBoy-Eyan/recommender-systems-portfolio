"""Train a causal two-tower retriever and evaluate on the frozen validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import candidate_recall
from src.recall.full_catalog import latest_category_map
from src.recall.two_tower import (
    SequenceDataset, TwoTowerModel, TwoTowerVocabulary, attach_target_categories,
    collate_sequences, train_two_tower,
)
from src.recall.item2vec_ann import Item2VecEmbeddings
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="训练并验证全量目录双塔召回")
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--output", default="outputs/two_tower_recall_validation")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-item2vec-init", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.0002)
    parser.add_argument(
        "--baseline-candidates",
        default="outputs/full_catalog_recall_validation_item2vec/recall_candidates.parquet",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    processed = ROOT / args.processed
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(processed / "train_samples.parquet")
    validation = pd.read_parquet(processed / "validation_samples.parquet")
    events = pd.read_parquet(processed / "events_enriched.parquet")
    first_seen = pd.read_parquet(processed / "item_first_seen.parquet")
    category_changes = pd.read_parquet(processed / "item_category_changes.parquet")
    labels = pd.read_parquet(processed / "labels.parquet")
    cutoff = int(labels.loc[labels["split"].eq("validation"), "target_ts"].min())
    if args.train_limit:
        train = train.sample(min(args.train_limit, len(train)), random_state=args.seed).sort_values("target_ts")
    if args.validation_limit:
        validation = validation.sample(
            min(args.validation_limit, len(validation)), random_state=args.seed
        ).sort_values("target_ts")
    train = attach_target_categories(train, events)
    validation = attach_target_categories(validation, events)

    item_ids = np.sort(first_seen.loc[first_seen["first_seen_ts"] < cutoff, "aid"].astype(int).unique())
    category_ids = np.sort(events.loc[events["ts"] < cutoff, "categoryid"].astype(int).unique())
    root_ids = np.sort(events.loc[events["ts"] < cutoff, "root_categoryid"].astype(int).unique())
    vocabulary = TwoTowerVocabulary(item_ids, category_ids, root_ids)
    train = train[train["target_aid"].isin(vocabulary.item_to_index)].reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    train_dataset = SequenceDataset(train, vocabulary)
    validation_dataset = SequenceDataset(validation, vocabulary)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法访问 CUDA GPU")
    model = TwoTowerModel(
        len(item_ids), len(category_ids), len(root_ids), args.embedding_dim
    )
    warm_started_items = 0
    item2vec_path = processed / "frozen_item2vec_embeddings.npz"
    if not args.no_item2vec_init and item2vec_path.exists():
        pretrained = Item2VecEmbeddings.load(item2vec_path)
        if pretrained.vectors.shape[1] == args.embedding_dim:
            with torch.no_grad():
                for aid, vector in zip(pretrained.item_ids, pretrained.vectors):
                    item_index = vocabulary.item_to_index.get(int(aid))
                    if item_index is not None:
                        model.item_embedding.weight[item_index].copy_(torch.from_numpy(vector))
                        warm_started_items += 1

    validation_callback = None
    if args.early_stopping:
        category_map_es = latest_category_map(category_changes, cutoff)
        latest_es = events[events["ts"] < cutoff].sort_values(["aid", "ts"]).groupby("aid").tail(1)
        root_map_es = dict(zip(latest_es["aid"].astype(int), latest_es["root_categoryid"].astype(int)))
        catalog_items_es = torch.tensor(
            [vocabulary.item_to_index[int(aid)] for aid in item_ids], dtype=torch.long, device=device
        )
        catalog_categories_es = torch.tensor(
            [vocabulary.category_to_index.get(category_map_es.get(int(aid), -1), 0) for aid in item_ids],
            dtype=torch.long, device=device,
        )
        catalog_roots_es = torch.tensor(
            [vocabulary.root_to_index.get(root_map_es.get(int(aid), -1), 0) for aid in item_ids],
            dtype=torch.long, device=device,
        )
        labels_es = validation[["session", "target_aid"]]
        targets_es = set(zip(labels_es["session"].astype(int), labels_es["target_aid"].astype(int)))
        history_es = dict(zip(validation["session"].astype(int), validation["history_aids"]))
        novel_es = {
            (int(row.session), int(row.target_aid)) for row in labels_es.itertuples(index=False)
            if int(row.target_aid) not in set(map(int, history_es[int(row.session)]))
        }
        baseline_es = set()
        baseline_path_es = ROOT / args.baseline_candidates
        if baseline_path_es.exists():
            frame_es = pd.read_parquet(baseline_path_es, columns=["session", "aid"])
            baseline_es = set(zip(frame_es["session"].astype(int), frame_es["aid"].astype(int)))

        def validation_callback(current_model, epoch):
            import faiss
            faiss.omp_set_num_threads(1)
            current_model.eval()
            with torch.no_grad():
                vectors = current_model.encode_item(
                    catalog_items_es, catalog_categories_es, catalog_roots_es
                ).cpu().numpy().astype("float32")
            ann = faiss.IndexHNSWFlat(args.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            ann.hnsw.efConstruction = 100
            ann.hnsw.efSearch = 80
            ann.add(np.ascontiguousarray(vectors))
            pairs = set()
            loader_es = DataLoader(validation_dataset, batch_size=1024, collate_fn=collate_sequences)
            with torch.no_grad():
                for batch in loader_es:
                    users = current_model.encode_user(
                        batch["history_items"].to(device), batch["history_types"].to(device),
                        batch["history_times"].to(device), batch["lengths"].to(device),
                    ).cpu().numpy().astype("float32")
                    _, found = ann.search(np.ascontiguousarray(users), args.topk)
                    for session, indices in zip(batch["session"].numpy(), found):
                        pairs.update((int(session), int(item_ids[i])) for i in indices if i >= 0)
            union = baseline_es | pairs
            union_recall = len(targets_es & union) / len(targets_es)
            novel_union_recall = len(novel_es & union) / len(novel_es)
            report = {
                "union_recall": union_recall,
                "novel_union_recall": novel_union_recall,
                "selection_score": 0.7 * union_recall + 0.3 * novel_union_recall,
            }
            print({"epoch": epoch, **report}, flush=True)
            return report

    losses = train_two_tower(
        model, train_dataset, epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, device=device,
        validation_callback=validation_callback,
        min_epochs=args.min_epochs, patience=args.patience, min_delta=args.min_delta,
    )

    # Build catalog features strictly as of the validation cutoff.
    category_map = latest_category_map(category_changes, cutoff)
    latest_visible = events[events["ts"] < cutoff].sort_values(["aid", "ts"]).groupby("aid").tail(1)
    root_map = dict(zip(latest_visible["aid"].astype(int), latest_visible["root_categoryid"].astype(int)))
    catalog_items = torch.tensor(
        [vocabulary.item_to_index[int(aid)] for aid in item_ids], dtype=torch.long, device=device
    )
    catalog_categories = torch.tensor(
        [vocabulary.category_to_index.get(category_map.get(int(aid), -1), 0) for aid in item_ids],
        dtype=torch.long, device=device,
    )
    catalog_roots = torch.tensor(
        [vocabulary.root_to_index.get(root_map.get(int(aid), -1), 0) for aid in item_ids],
        dtype=torch.long, device=device,
    )
    model.eval()
    with torch.no_grad():
        catalog_vectors = model.encode_item(catalog_items, catalog_categories, catalog_roots).cpu().numpy().astype("float32")

    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(args.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 120
    index.hnsw.efSearch = 96
    index.add(np.ascontiguousarray(catalog_vectors))
    loader = DataLoader(validation_dataset, batch_size=1024, collate_fn=collate_sequences)
    rows = []
    sampled_wins = 0.0
    sampled_pairs = 0
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        for batch in loader:
            users = model.encode_user(
                batch["history_items"].to(device), batch["history_types"].to(device),
                batch["history_times"].to(device), batch["lengths"].to(device),
            ).cpu().numpy().astype("float32")
            scores, indices = index.search(np.ascontiguousarray(users), args.topk)
            for session, found, values in zip(batch["session"].numpy(), indices, scores):
                rows.extend(
                    (int(session), int(item_ids[i]), "two_tower", rank, float(score))
                    for rank, (i, score) in enumerate(zip(found, values), 1) if i >= 0
                )
            for row_index, target_aid in enumerate(batch["target_aid"].numpy()):
                target = vocabulary.item_to_index.get(int(target_aid))
                if target is None:
                    continue
                target_catalog_index = int(np.searchsorted(item_ids, target_aid))
                negatives = rng.integers(0, len(item_ids), size=100)
                positive = float(users[row_index] @ catalog_vectors[target_catalog_index])
                negative_scores = catalog_vectors[negatives] @ users[row_index]
                sampled_wins += float((positive > negative_scores).sum()) + 0.5 * float((positive == negative_scores).sum())
                sampled_pairs += len(negatives)
    recalled = pd.DataFrame(rows, columns=["session", "aid", "source", "source_rank", "source_score"])
    labels_eval = validation[["session", "target_aid", "target_type"]]
    target_pairs = set(zip(labels_eval["session"].astype(int), labels_eval["target_aid"].astype(int)))
    tower_pairs = set(zip(recalled["session"].astype(int), recalled["aid"].astype(int)))
    history_map = dict(zip(validation["session"].astype(int), validation["history_aids"]))
    novel = labels_eval[[int(r.target_aid) not in set(map(int, history_map[int(r.session)])) for r in labels_eval.itertuples()]]
    novel_pairs = set(zip(novel["session"].astype(int), novel["target_aid"].astype(int)))
    metrics = {
        "protocol": {"cutoff_ts": cutoff, "test_evaluated": False, "device": device},
        "train_samples": len(train), "validation_samples": len(validation),
        "catalog_items": len(item_ids), "losses": losses,
        "early_stopping": model.early_stopping_report,
        "item2vec_warm_started_items": warm_started_items,
        "two_tower_recall_at_k": len(target_pairs & tower_pairs) / len(labels_eval),
        "novel_recall_at_k": len(novel_pairs & tower_pairs) / len(novel) if len(novel) else None,
        "sampled_auc_gauc": sampled_wins / sampled_pairs if sampled_pairs else None,
        "runtime_seconds": time.perf_counter() - started,
    }
    baseline_path = ROOT / args.baseline_candidates
    if baseline_path.exists():
        baseline = pd.read_parquet(baseline_path, columns=["session", "aid"])
        selected_sessions = set(validation["session"].astype(int))
        baseline = baseline[baseline["session"].isin(selected_sessions)]
        baseline_pairs = set(zip(baseline["session"].astype(int), baseline["aid"].astype(int)))
        union_pairs = baseline_pairs | tower_pairs
        union_action_recall = {}
        for action, group in labels_eval.groupby("target_type"):
            action_pairs = set(zip(group["session"].astype(int), group["target_aid"].astype(int)))
            union_action_recall[str(action)] = len(action_pairs & union_pairs) / len(group)
        metrics["union_with_baseline"] = {
            "baseline_recall": len(target_pairs & baseline_pairs) / len(labels_eval),
            "union_recall": len(target_pairs & union_pairs) / len(labels_eval),
            "two_tower_exclusive_hits": len((target_pairs & tower_pairs) - baseline_pairs),
            "baseline_novel_recall": len(novel_pairs & baseline_pairs) / len(novel),
            "union_novel_recall": len(novel_pairs & union_pairs) / len(novel),
            "two_tower_novel_exclusive_hits": len((novel_pairs & tower_pairs) - baseline_pairs),
            "union_recall_by_action": union_action_recall,
            "weighted_union_recall": (
                0.1 * union_action_recall.get("clicks", 0.0)
                + 0.3 * union_action_recall.get("carts", 0.0)
                + 0.6 * union_action_recall.get("orders", 0.0)
            ),
        }
    recalled.to_parquet(output / "two_tower_candidates.parquet", index=False)
    torch.save({"state_dict": model.cpu().state_dict(), "item_ids": item_ids, "category_ids": category_ids, "root_ids": root_ids, "config": vars(args)}, output / "two_tower.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
