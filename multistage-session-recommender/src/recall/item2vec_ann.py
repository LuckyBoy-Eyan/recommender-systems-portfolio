"""Scalable Item2Vec training and FAISS ANN retrieval for the frozen catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.recall.sources import _build_skipgram_pairs


DEFAULT_ACTION_WEIGHTS = {"clicks": 1.0, "carts": 3.0, "orders": 6.0}


@dataclass
class Item2VecEmbeddings:
    item_ids: np.ndarray
    vectors: np.ndarray
    metadata: dict

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            item_ids=self.item_ids.astype(np.int64),
            vectors=self.vectors.astype(np.float32),
            metadata=np.array([self.metadata], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Item2VecEmbeddings":
        with np.load(path, allow_pickle=True) as data:
            return cls(
                item_ids=data["item_ids"].astype(np.int64),
                vectors=data["vectors"].astype(np.float32),
                metadata=dict(data["metadata"][0]),
            )


def train_item2vec_embeddings(
    events: pd.DataFrame,
    *,
    dimensions: int = 64,
    window: int = 3,
    negative_samples: int = 5,
    epochs: int = 3,
    batch_size: int = 8192,
    learning_rate: float = 0.01,
    min_count: int = 2,
    action_weights: dict[str, float] | None = None,
    seed: int = 2026,
    num_threads: int | None = None,
    subsample: float | None = None,
    adaptive_window: bool = False,
    checkpoint_epochs: tuple[int, ...] = (),
    checkpoint_callback=None,
) -> Item2VecEmbeddings:
    """Train SGNS with sparse gradients; memory grows linearly with items and pairs."""
    if events.empty:
        return Item2VecEmbeddings(np.array([], dtype=np.int64), np.empty((0, dimensions)), {})
    if min(dimensions, window, negative_samples, epochs, batch_size, min_count) < 1:
        raise ValueError("Item2Vec integer parameters must be positive")
    weights = action_weights or DEFAULT_ACTION_WEIGHTS
    counts = events["aid"].astype(int).value_counts()
    item_ids = np.array(
        sorted(int(aid) for aid, count in counts.items() if int(count) >= min_count),
        dtype=np.int64,
    )
    if len(item_ids) < 2:
        return Item2VecEmbeddings(item_ids, np.empty((len(item_ids), dimensions)), {})
    item_to_index = {int(aid): index for index, aid in enumerate(item_ids)}
    training_events = events
    if subsample is not None:
        frequencies = events["aid"].map(counts).to_numpy(dtype=np.float64) / len(events)
        keep_probability = np.minimum(
            1.0, np.sqrt(float(subsample) / frequencies) + float(subsample) / frequencies
        )
        rng = np.random.default_rng(seed)
        training_events = events.loc[rng.random(len(events)) < keep_probability].copy()
    if adaptive_window:
        centers_list, contexts_list, weights_list = [], [], []
        ordered = training_events.sort_values(["session", "ts"], kind="stable")
        for _, group in ordered.groupby("session", sort=False):
            sequence = [
                (item_to_index[int(row.aid)], float(weights[row.type]))
                for row in group.itertuples(index=False) if int(row.aid) in item_to_index
            ]
            radius = min(10, max(5, int(round(len(sequence) * 0.5))))
            for position, (center, center_weight) in enumerate(sequence):
                for context_position in range(max(0, position - radius), min(len(sequence), position + radius + 1)):
                    if context_position == position:
                        continue
                    context, context_weight = sequence[context_position]
                    distance = abs(context_position - position)
                    distance_weight = 1.0 if distance <= 3 else math.exp(-0.2 * (distance - 3))
                    centers_list.append(center)
                    contexts_list.append(context)
                    weights_list.append(math.sqrt(center_weight * context_weight) * distance_weight)
        centers = torch.tensor(centers_list, dtype=torch.long)
        contexts = torch.tensor(contexts_list, dtype=torch.long)
        pair_weights = torch.tensor(weights_list, dtype=torch.float32)
    else:
        centers, contexts, pair_weights = _build_skipgram_pairs(
            training_events, item_to_index, window, weights
        )
    if centers.numel() == 0:
        return Item2VecEmbeddings(item_ids, np.empty((len(item_ids), dimensions)), {})
    pair_weights /= pair_weights.mean()

    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    if num_threads is not None:
        torch.set_num_threads(max(1, int(num_threads)))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    input_embeddings = nn.Embedding(len(item_ids), dimensions, sparse=True)
    output_embeddings = nn.Embedding(len(item_ids), dimensions, sparse=True)
    bound = 0.5 / dimensions
    with torch.no_grad():
        input_embeddings.weight.uniform_(-bound, bound, generator=generator)
        output_embeddings.weight.zero_()
    optimizer = torch.optim.SparseAdam(
        list(input_embeddings.parameters()) + list(output_embeddings.parameters()),
        lr=learning_rate,
    )
    negative_distribution = torch.tensor(
        [float(counts.get(int(aid), 0)) ** 0.75 for aid in item_ids],
        dtype=torch.float32,
    )
    negative_distribution /= negative_distribution.sum()
    try:
        for epoch in range(1, epochs + 1):
            learning_rate_epoch = learning_rate if epoch <= 5 else (learning_rate * 0.5 if epoch <= 10 else learning_rate * 0.2)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_epoch
            permutation = torch.randperm(len(centers), generator=generator)
            for start in range(0, len(centers), batch_size):
                selected = permutation[start : start + batch_size]
                center_batch = centers[selected]
                context_batch = contexts[selected]
                weight_batch = pair_weights[selected]
                negatives = torch.multinomial(
                    negative_distribution,
                    len(selected) * negative_samples,
                    replacement=True,
                    generator=generator,
                ).view(len(selected), negative_samples)
                # Reject only accidental self negatives. Full positive-context masks are
                # quadratic in vocabulary size and are unsuitable for the full catalog.
                invalid = negatives.eq(center_batch[:, None])
                while invalid.any():
                    negatives[invalid] = torch.multinomial(
                        negative_distribution,
                        int(invalid.sum()),
                        replacement=True,
                        generator=generator,
                    )
                    invalid = negatives.eq(center_batch[:, None])
                center_vectors = input_embeddings(center_batch)
                positive_logits = (center_vectors * output_embeddings(context_batch)).sum(1)
                negative_logits = torch.bmm(
                    output_embeddings(negatives), center_vectors.unsqueeze(2)
                ).squeeze(2)
                loss = -(
                    F.logsigmoid(positive_logits)
                    + F.logsigmoid(-negative_logits).sum(1)
                )
                loss = (loss * weight_batch).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch in checkpoint_epochs and checkpoint_callback is not None:
                vectors_epoch = input_embeddings.weight.detach() + output_embeddings.weight.detach()
                vectors_epoch = F.normalize(vectors_epoch, p=2, dim=1).cpu().numpy().astype(np.float32)
                checkpoint_callback(epoch, Item2VecEmbeddings(item_ids, vectors_epoch, {
                    "epoch": epoch, "dimensions": dimensions, "min_count": min_count,
                    "negative_samples": negative_samples, "subsample": subsample,
                }))
    finally:
        torch.set_num_threads(previous_threads)

    # Input and output spaces contain complementary signal in SGNS.
    vectors = input_embeddings.weight.detach() + output_embeddings.weight.detach()
    vectors = F.normalize(vectors, p=2, dim=1).cpu().numpy().astype(np.float32)
    metadata = {
        "vocabulary_items": int(len(item_ids)),
        "positive_pairs": int(len(centers)),
        "dimensions": int(dimensions),
        "window": int(window),
        "negative_samples": int(negative_samples),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "min_count": int(min_count),
        "subsample": subsample,
        "adaptive_window": bool(adaptive_window),
        "retained_events": int(len(training_events)),
        "seed": int(seed),
        "training_seconds": time.perf_counter() - started,
    }
    return Item2VecEmbeddings(item_ids, vectors, metadata)


class Item2VecANN:
    """FAISS HNSW inner-product index over normalized Item2Vec vectors."""

    def __init__(
        self,
        embeddings: Item2VecEmbeddings,
        *,
        hnsw_m: int = 32,
        ef_construction: int = 120,
        ef_search: int = 96,
    ) -> None:
        try:
            import faiss
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install faiss-cpu to use full-catalog ANN retrieval") from error
        # Avoid libomp contention with PyTorch/Scikit-learn in the same process.
        faiss.omp_set_num_threads(1)
        self.item_ids = embeddings.item_ids
        self.vectors = np.ascontiguousarray(embeddings.vectors, dtype=np.float32)
        self.item_to_index = {int(aid): i for i, aid in enumerate(self.item_ids)}
        # Some FAISS HNSW builds are unstable when the graph is smaller than M.
        # Exact search is trivial at this size and keeps tests/small dev runs robust.
        if len(self.item_ids) < max(100, hnsw_m * 2):
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        else:
            self.index = faiss.IndexHNSWFlat(
                self.vectors.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT
            )
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        self.index.add(self.vectors)

    def session_vectors(self, samples: pd.DataFrame, max_history: int = 5) -> tuple[np.ndarray, np.ndarray]:
        vectors = np.zeros((len(samples), self.vectors.shape[1]), dtype=np.float32)
        valid = np.zeros(len(samples), dtype=bool)
        for row_index, sample in enumerate(samples.itertuples(index=False)):
            history = list(sample.history_aids)[-max_history:][::-1]
            weighted = []
            vector_weights = []
            for recency, aid in enumerate(history):
                index = self.item_to_index.get(int(aid))
                if index is not None:
                    weighted.append(self.vectors[index])
                    vector_weights.append(1.0 / math.sqrt(recency + 1))
            if weighted:
                vector = np.average(np.stack(weighted), axis=0, weights=vector_weights)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vectors[row_index] = vector / norm
                    valid[row_index] = True
        return vectors, valid

    def recall(self, samples: pd.DataFrame, topk: int = 50) -> pd.DataFrame:
        vectors, valid = self.session_vectors(samples)
        rows = []
        if not valid.any():
            return pd.DataFrame(columns=["session", "aid", "source", "source_rank", "source_score"])
        scores, indices = self.index.search(np.ascontiguousarray(vectors[valid]), topk)
        sessions = samples.iloc[np.flatnonzero(valid)]["session"].astype(int).to_numpy()
        for session, item_indices, item_scores in zip(sessions, indices, scores):
            for rank, (item_index, score) in enumerate(zip(item_indices, item_scores), 1):
                if item_index >= 0:
                    rows.append((int(session), int(self.item_ids[item_index]), "item2vec", rank, float(score)))
        return pd.DataFrame(rows, columns=["session", "aid", "source", "source_rank", "source_score"])

    def sampled_auc_gauc(
        self,
        samples: pd.DataFrame,
        negatives_per_session: int = 100,
        seed: int = 2026,
    ) -> dict:
        vectors, valid = self.session_vectors(samples)
        rng = np.random.default_rng(seed)
        wins = 0.0
        pairs = 0
        session_aucs = []
        target_covered = 0
        for row_index, sample in enumerate(samples.itertuples(index=False)):
            target_index = self.item_to_index.get(int(sample.target_aid))
            if not valid[row_index] or target_index is None:
                continue
            target_covered += 1
            negative_indices = rng.integers(0, len(self.item_ids), size=negatives_per_session)
            invalid = negative_indices == target_index
            while invalid.any():
                negative_indices[invalid] = rng.integers(0, len(self.item_ids), size=int(invalid.sum()))
                invalid = negative_indices == target_index
            positive_score = float(vectors[row_index] @ self.vectors[target_index])
            negative_scores = self.vectors[negative_indices] @ vectors[row_index]
            local_wins = float((positive_score > negative_scores).sum())
            local_wins += 0.5 * float((positive_score == negative_scores).sum())
            session_aucs.append(local_wins / negatives_per_session)
            wins += local_wins
            pairs += negatives_per_session
        return {
            "sampled_auc": wins / pairs if pairs else None,
            "session_gauc": float(np.mean(session_aucs)) if session_aucs else None,
            "eligible_sessions": len(session_aucs),
            "target_embedding_coverage": target_covered / len(samples) if len(samples) else 0.0,
            "negatives_per_session": int(negatives_per_session),
            "seed": int(seed),
        }
