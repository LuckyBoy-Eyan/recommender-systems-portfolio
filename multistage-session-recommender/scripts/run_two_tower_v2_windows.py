"""Build hard negatives, train Two-Tower V2, and export validation Top-300."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recall.item2vec_ann import Item2VecANN, Item2VecEmbeddings
from src.recall.item2vec_ann import train_item2vec_embeddings
from src.recall.full_catalog import build_frozen_indexes
from src.recall.two_tower_v2 import TwoTowerV2, TwoTowerV2Dataset, causal_mask, collate_v2


def aggregate(recent, index, limit=80):
    scores = Counter()
    for rank, aid in enumerate(recent[:10]):
        for candidate, score in index.get(int(aid), [])[:80]:
            scores[int(candidate)] += float(score) * math.exp(-0.1 * rank)
    return [aid for aid, _ in scores.most_common(limit)]


def build_catalog(processed: Path, cutoff: int, item_ids: np.ndarray):
    item_to_row = {int(a): i + 1 for i, a in enumerate(item_ids)}
    paths = pd.read_parquet(processed / "category_paths.parquet")
    path_map = {int(r.categoryid): tuple(map(int, r.category_path)) for r in paths.itertuples(index=False) if bool(r.in_tree)}
    changes = pd.read_parquet(processed / "item_category_changes.parquet")
    latest = changes[changes.timestamp < cutoff].sort_values(["itemid", "timestamp"]).groupby("itemid").tail(1)
    category_ids = sorted(set(latest.categoryid.astype(int)) | {-1}); cat_index = {v: i + 1 for i, v in enumerate(category_ids)}
    parent_values = sorted({path[-2] for path in path_map.values() if len(path) >= 2} | {-1}); parent_index = {v: i + 1 for i, v in enumerate(parent_values)}
    root_values = sorted(set(paths.root_categoryid.astype(int)) | {-1}); root_index = {v: i + 1 for i, v in enumerate(root_values)}
    category = np.zeros(len(item_ids) + 1, np.int64); parent = category.copy(); root = category.copy(); depth = category.copy()
    for row in latest.itertuples(index=False):
        idx = item_to_row.get(int(row.itemid)); path = path_map.get(int(row.categoryid), ())
        if idx is None: continue
        category[idx] = cat_index[int(row.categoryid)]
        parent[idx] = parent_index.get(path[-2] if len(path) >= 2 else -1, 0)
        root[idx] = root_index.get(path[0] if path else -1, 0); depth[idx] = min(len(path), 7)

    availability = pd.read_parquet(processed / "item_availability_changes.parquet")
    availability = availability[availability.timestamp < cutoff].sort_values(["itemid", "timestamp"]).groupby("itemid").tail(1)
    available = np.zeros(len(item_ids) + 1, np.int64)
    for row in availability.itertuples(index=False):
        idx = item_to_row.get(int(row.itemid))
        if idx is not None: available[idx] = 2 if int(row.available) == 1 else 1

    events = pd.read_parquet(processed / "events_all.parquet")
    reference = events[events.ts < cutoff]
    columns = []
    for days in (1, 7, 30):
        window = reference[reference.ts >= cutoff - days * 86_400_000]
        for action in ("clicks", "carts", "orders"):
            counts = window.loc[window.type.eq(action), "aid"].value_counts()
            columns.append(np.array([counts.get(int(a), 0) for a in item_ids], np.float32))
    sessions30 = reference[reference.ts >= cutoff - 30 * 86_400_000].groupby("aid")["session"].nunique()
    first = reference.groupby("aid")["ts"].min(); last = reference.groupby("aid")["ts"].max()
    columns += [
        np.array([sessions30.get(int(a), 0) for a in item_ids], np.float32),
        np.array([(cutoff - first.get(int(a), cutoff)) / 86_400_000 for a in item_ids], np.float32),
        np.array([(cutoff - last.get(int(a), cutoff)) / 86_400_000 for a in item_ids], np.float32),
    ]
    stats = np.stack(columns, axis=1); stats = np.log1p(np.maximum(stats, 0))
    stats = (stats - stats.mean(0)) / np.maximum(stats.std(0), 1e-6)
    stats = np.vstack([np.zeros((1, stats.shape[1]), np.float32), stats.astype(np.float32)])
    features = {
        "category": torch.from_numpy(category), "parent": torch.from_numpy(parent),
        "root": torch.from_numpy(root), "depth": torch.from_numpy(depth),
        "availability": torch.from_numpy(available), "stats": torch.from_numpy(stats),
    }
    return item_to_row, features, reference, cat_index


def build_hard_negatives(samples, item_ids, item_to_row, indexes, embeddings, reference, output, seed=2026):
    if output.exists():
        cached = np.load(output)
        if cached.shape == (len(samples), 96): return cached
    ann = Item2VecANN(embeddings)
    scoring = samples.copy(); vectors, valid = ann.session_vectors(scoring, max_history=10)
    found = np.full((len(samples), 80), -1, np.int64)
    if valid.any():
        _, result = ann.index.search(np.ascontiguousarray(vectors[valid]), 80); found[valid] = result
    counts = reference.aid.value_counts(); probability = np.array([counts.get(int(a), 0) ** 0.75 for a in item_ids], np.float64); probability /= probability.sum()
    session_events = {
        int(s): (g.ts.to_numpy(), g.aid.astype(int).to_numpy())
        for s, g in reference.sort_values(["session", "ts"]).groupby("session", sort=False)
    }
    rng = np.random.default_rng(seed); hard = np.zeros((len(samples), 96), np.int32)
    for row_index, sample in enumerate(samples.itertuples(index=False)):
        recent = list(dict.fromkeys(map(int, list(sample.history_aids)[::-1])))
        i2v = [int(ann.item_ids[i]) for i in found[row_index, 10:] if i >= 0][:32]
        behavior = []
        left = aggregate(recent, indexes["itemcf"], 40); right = aggregate(recent, indexes["transition"], 40)
        for a, b in zip(left, right): behavior.extend([a, b])
        behavior = behavior[:24]
        category_candidates = []
        for aid in recent[:5]:
            category = indexes["item_category"].get(aid)
            category_candidates.extend(a for a, _ in indexes["category_popularity"].get(category, [])[:40])
        category_candidates = category_candidates[:16]
        stage = "order" if 2 in set(sample.history_type_ids) else ("cart" if 1 in set(sample.history_type_ids) else "click")
        popular = indexes["hybrid_popularity"][stage][:12]
        random_items = item_ids[rng.choice(len(item_ids), 12, replace=True, p=probability)].astype(int).tolist()
        ts_values, aids = session_events.get(int(sample.session), (np.array([]), np.array([])))
        future = set(map(int, aids[ts_values >= int(sample.target_ts)]))
        blocked = future | {int(sample.target_aid)}; pools = [i2v, behavior, category_candidates, popular, random_items]
        selected, seen = [], set()
        while any(pools) and len(selected) < 96:
            for pool in pools:
                if not pool: continue
                aid = int(pool.pop(0))
                if aid not in seen and aid not in blocked and aid in item_to_row:
                    selected.append(aid); seen.add(aid)
        while len(selected) < 96:
            aid = int(item_ids[rng.choice(len(item_ids), p=probability)])
            if aid not in seen and aid not in blocked:
                selected.append(aid); seen.add(aid)
        hard[row_index] = [item_to_row[a] for a in selected]
        if (row_index + 1) % 50000 == 0: print(f"hard negatives {row_index + 1}/{len(samples)}", flush=True)
    np.save(output, hard); return hard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/retailrocket")
    parser.add_argument("--output", default="outputs/windows_two_tower_v2_validation")
    parser.add_argument("--epochs", type=int, default=15); parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=300); parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0); parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--oof-fold", type=int, default=-1, help="0..3生成严格时间OOF；-1保持正式验证模式")
    parser.add_argument("--oof-folds", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    args = parser.parse_args(); started = time.perf_counter(); processed = ROOT / args.processed; output = ROOT / args.output; output.mkdir(parents=True, exist_ok=True)
    all_train = pd.read_parquet(processed / "train_samples.parquet").sort_values(["target_ts", "session"], kind="mergesort").reset_index(drop=True)
    all_train["sample_id"] = np.arange(len(all_train), dtype=np.int64)
    if args.oof_fold >= 0:
        if args.oof_fold >= args.oof_folds: raise ValueError("oof-fold超出范围")
        quantiles = np.linspace(args.warmup_fraction, 1.0, args.oof_folds + 1)
        boundaries = [int(all_train.target_ts.quantile(float(q))) for q in quantiles]; boundaries[-1] += 1
        cutoff, score_end = boundaries[args.oof_fold], boundaries[args.oof_fold + 1]
    else:
        labels = pd.read_parquet(processed / "labels.parquet"); cutoff = int(labels.loc[labels.split.eq("validation"), "target_ts"].min()); score_end = None
    first_seen = pd.read_parquet(processed / "item_first_seen.parquet"); item_ids = np.sort(first_seen.loc[first_seen.first_seen_ts < cutoff, "aid"].astype(int).unique())
    item_to_row, features, reference, _ = build_catalog(processed, cutoff, item_ids)
    if args.oof_fold >= 0:
        embedding_path = output / "item2vec_v2.npz"
        if embedding_path.exists(): embeddings = Item2VecEmbeddings.load(embedding_path)
        else:
            embeddings = train_item2vec_embeddings(reference, dimensions=128, window=10, negative_samples=15, epochs=10, batch_size=4096, min_count=5, subsample=3e-5, adaptive_window=True, seed=args.seed + args.oof_fold)
            embeddings.save(embedding_path)
    else:
        embeddings = Item2VecEmbeddings.load(processed / "frozen_item2vec_embeddings_v2.npz")
    pretrained = np.zeros((len(item_ids) + 1, embeddings.vectors.shape[1]), np.float32)
    for aid, vector in zip(embeddings.item_ids, embeddings.vectors):
        idx = item_to_row.get(int(aid))
        if idx is not None: pretrained[idx] = vector
    if args.oof_fold >= 0:
        train = all_train[all_train.target_ts < cutoff].copy()
        validation = all_train[(all_train.target_ts >= cutoff) & (all_train.target_ts < score_end)].copy()
        validation["original_session"] = validation["session"]
        validation["session"] = validation["sample_id"]
    else:
        train = all_train; validation = pd.read_parquet(processed / "validation_samples.parquet")
    train = train[train.target_aid.isin(item_to_row)].reset_index(drop=True)
    if args.train_limit: train = train.sample(min(args.train_limit, len(train)), random_state=args.seed).sort_values("target_ts").reset_index(drop=True)
    if args.validation_limit: validation = validation.sample(min(args.validation_limit, len(validation)), random_state=args.seed).sort_values("target_ts").reset_index(drop=True)
    indexes = build_frozen_indexes(reference, reference, pd.read_parquet(processed / "item_category_changes.parquet"), cutoff) if args.oof_fold >= 0 else joblib.load(processed / "frozen_recall_indexes.joblib")
    hard_path = output / ("hard_negatives.npy" if not args.train_limit else f"hard_negatives_{len(train)}.npy")
    hard = build_hard_negatives(train, item_ids, item_to_row, indexes, embeddings, reference, hard_path, args.seed)
    dataset = TwoTowerV2Dataset(train, item_to_row, hard); loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_v2, num_workers=0)
    validation_hard = np.tile(np.arange(1, 97, dtype=np.int32), (len(validation), 1)); validation_dataset = TwoTowerV2Dataset(validation, item_to_row, validation_hard)
    device = args.device
    if device == "cuda": assert torch.cuda.is_available(), "CUDA unavailable"
    model = TwoTowerV2(features, torch.from_numpy(pretrained)).to(device); model.item2vec_embedding.weight.requires_grad_(False)
    base_lr = 1.5e-3
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not n.startswith("item2vec_embedding")], "lr": base_lr},
        {"params": model.item2vec_embedding.parameters(), "lr": base_lr * 0.1},
    ], weight_decay=1e-5)
    total_steps = len(loader) * args.epochs; warmup = max(1, int(total_steps * .05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp); action_weights = torch.tensor([1., math.sqrt(3), math.sqrt(6)], device=device)
    best_score = -1.; best_state = None; stale = 0; history = []
    baseline_path = ROOT / "outputs/full_catalog_recall_validation_hybrid/recall_candidates.parquet"
    if args.oof_fold >= 0: baseline_pairs = set()
    else:
        baseline = pd.read_parquet(baseline_path, columns=["session", "aid"]); baseline_pairs = set(zip(baseline.session.astype(int), baseline.aid.astype(int)))
    targets = set(zip(validation.session.astype(int), validation.target_aid.astype(int)))
    for epoch in range(1, args.epochs + 1):
        model.train(); model.item2vec_embedding.weight.requires_grad_(epoch >= 3); total_loss = 0
        hard_alpha = 0. if epoch <= 2 else (.25 if epoch <= 5 else .5)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                users = model.encode_user(batch["history_items"], batch["history_types"], batch["history_times"], batch["lengths"])
                positives = model.encode_items(batch["target_item"]); logits = users @ positives.T / .07
                logits = logits.masked_fill(~causal_mask(batch["target_aid"], batch["target_ts"]), torch.finfo(logits.dtype).min)
                row_inbatch = F.cross_entropy(logits, torch.arange(len(users), device=device), reduction="none")
                negatives = model.encode_items(batch["hard_negatives"]); positive_score = (users * positives).sum(-1, keepdim=True)
                hard_logits = torch.cat([positive_score, torch.einsum("bd,bhd->bh", users, negatives)], 1) / .07
                row_hard = F.cross_entropy(hard_logits, torch.zeros(len(users), dtype=torch.long, device=device), reduction="none")
                loss = ((1-hard_alpha)*row_inbatch + hard_alpha*row_hard) * action_weights[batch["target_type"]]; loss = loss.mean()
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update(); scheduler.step(); total_loss += float(loss)*len(users)
        model.eval(); catalog=[]
        with torch.no_grad():
            for start in range(1, len(item_ids)+1, 8192): catalog.append(model.encode_items(torch.arange(start, min(start+8192, len(item_ids)+1), device=device)).cpu())
        catalog_vectors = torch.cat(catalog).numpy().astype("float32")
        import faiss; faiss.omp_set_num_threads(1); ann=faiss.IndexHNSWFlat(64,32,faiss.METRIC_INNER_PRODUCT); ann.hnsw.efConstruction=120; ann.hnsw.efSearch=128; ann.add(catalog_vectors)
        pairs=set(); val_loader=DataLoader(validation_dataset,batch_size=1024,collate_fn=collate_v2)
        with torch.no_grad():
            for batch in val_loader:
                users=model.encode_user(batch["history_items"].to(device),batch["history_types"].to(device),batch["history_times"].to(device),batch["lengths"].to(device)).cpu().numpy().astype("float32"); scores,found=ann.search(users,args.topk)
                for session,indices in zip(batch["session"].numpy(),found): pairs.update((int(session),int(item_ids[i])) for i in indices if i>=0)
        union=baseline_pairs|pairs; by={}
        for action,g in validation.groupby("target_type"): q=set(zip(g.session.astype(int),g.target_aid.astype(int))); by[str(action)]=len(q&union)/len(g)
        weighted=.1*by.get("clicks",0)+.3*by.get("carts",0)+.6*by.get("orders",0); union_recall=len(targets&union)/len(targets); exclusive=len((targets&pairs)-baseline_pairs)/len(targets); score=.6*weighted+.3*union_recall+.1*exclusive
        report={"epoch":epoch,"loss":total_loss/len(dataset),"union_recall":union_recall,"weighted_union_recall":weighted,"exclusive_rate":exclusive,"selection_score":score}; history.append(report); print(report,flush=True)
        if args.oof_fold >= 0:
            best_score=score; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; stale=0
        elif score>best_score+.0002: best_score=score; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if args.oof_fold < 0 and epoch>=5 and stale>=4: break
    model.load_state_dict(best_state); model.to(device).eval(); rows=[]
    catalog=[]
    with torch.no_grad():
        for start in range(1,len(item_ids)+1,8192): catalog.append(model.encode_items(torch.arange(start,min(start+8192,len(item_ids)+1),device=device)).cpu())
    vectors=torch.cat(catalog).numpy().astype("float32"); import faiss; ann=faiss.IndexHNSWFlat(64,32,faiss.METRIC_INNER_PRODUCT); ann.hnsw.efConstruction=120; ann.hnsw.efSearch=128; ann.add(vectors)
    with torch.no_grad():
        for batch in DataLoader(validation_dataset,batch_size=1024,collate_fn=collate_v2):
            users=model.encode_user(batch["history_items"].to(device),batch["history_types"].to(device),batch["history_times"].to(device),batch["lengths"].to(device)).cpu().numpy().astype("float32"); scores,found=ann.search(users,args.topk)
            for session,ii,ss in zip(batch["session"].numpy(),found,scores): rows.extend((int(session),int(item_ids[i]),"two_tower",rank,float(sc)) for rank,(i,sc) in enumerate(zip(ii,ss),1) if i>=0)
    pd.DataFrame(rows,columns=["session","aid","source","source_rank","source_score"]).to_parquet(output/"two_tower_candidates.parquet",index=False)
    torch.save({"state_dict":model.cpu().state_dict(),"item_ids":item_ids,"config":vars(args)},output/"two_tower_v2.pt")
    metrics={"history":history,"best_selection_score":best_score,"runtime_seconds":time.perf_counter()-started,"test_evaluated":False,"hard_negative_shape":list(hard.shape),"oof_fold":args.oof_fold,"cutoff_ts":cutoff,"score_end_ts":score_end,"fixed_epoch_oof":args.oof_fold>=0}; (output/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False)); print(json.dumps(metrics,ensure_ascii=False))


if __name__ == "__main__": main()
