"""Behavior-aware two-tower retrieval with causal in-batch negatives."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ACTION_WEIGHTS = torch.tensor([1.0, math.sqrt(3.0), math.sqrt(6.0)])


@dataclass
class TwoTowerVocabulary:
    item_ids: np.ndarray
    category_ids: np.ndarray
    root_ids: np.ndarray

    def __post_init__(self) -> None:
        self.item_to_index = {int(aid): index + 1 for index, aid in enumerate(self.item_ids)}
        self.category_to_index = {
            int(category): index + 1 for index, category in enumerate(self.category_ids)
        }
        self.root_to_index = {int(root): index + 1 for index, root in enumerate(self.root_ids)}


class SequenceDataset(Dataset):
    def __init__(self, samples: pd.DataFrame, vocabulary: TwoTowerVocabulary, max_history: int = 30):
        self.samples = samples.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.max_history = max_history

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        row = self.samples.iloc[index]
        aids = list(row.history_aids)[-self.max_history :]
        types = list(row.history_type_ids)[-self.max_history :]
        deltas = list(row.history_time_deltas_ms)[-self.max_history :]
        return {
            "history_items": [self.vocabulary.item_to_index.get(int(aid), 0) for aid in aids],
            "history_types": [int(value) + 1 for value in types],
            "history_times": [min(31, int(math.log2(max(int(value) // 1000, 0) + 1))) + 1 for value in deltas],
            "target_item": self.vocabulary.item_to_index.get(int(row.target_aid), 0),
            "target_aid": int(row.target_aid),
            "target_type": int(row.target_type_id),
            "target_ts": int(row.target_ts),
            "target_category": self.vocabulary.category_to_index.get(int(row.target_categoryid), 0),
            "target_root": self.vocabulary.root_to_index.get(int(row.target_root_categoryid), 0),
            "session": int(row.session),
        }


def collate_sequences(rows: list[dict]) -> dict[str, torch.Tensor]:
    length = max(len(row["history_items"]) for row in rows)
    output = {
        "history_items": torch.zeros((len(rows), length), dtype=torch.long),
        "history_types": torch.zeros((len(rows), length), dtype=torch.long),
        "history_times": torch.zeros((len(rows), length), dtype=torch.long),
        "lengths": torch.tensor([len(row["history_items"]) for row in rows]),
    }
    for index, row in enumerate(rows):
        size = len(row["history_items"])
        output["history_items"][index, :size] = torch.tensor(row["history_items"])
        output["history_types"][index, :size] = torch.tensor(row["history_types"])
        output["history_times"][index, :size] = torch.tensor(row["history_times"])
    for key in (
        "target_item", "target_aid", "target_type", "target_ts",
        "target_category", "target_root", "session",
    ):
        output[key] = torch.tensor([row[key] for row in rows], dtype=torch.long)
    return output


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        item_count: int,
        category_count: int,
        root_count: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(item_count + 1, embedding_dim, padding_idx=0)
        self.history_type_embedding = nn.Embedding(4, embedding_dim, padding_idx=0)
        self.time_embedding = nn.Embedding(34, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.category_embedding = nn.Embedding(category_count + 1, embedding_dim, padding_idx=0)
        self.root_embedding = nn.Embedding(root_count + 1, embedding_dim, padding_idx=0)
        self.user_projection = nn.Linear(embedding_dim, embedding_dim)
        self.item_projection = nn.Linear(embedding_dim, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (
            self.item_embedding, self.history_type_embedding, self.time_embedding,
            self.category_embedding, self.root_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        nn.init.eye_(self.user_projection.weight)
        nn.init.eye_(self.item_projection.weight)
        nn.init.zeros_(self.user_projection.bias)
        nn.init.zeros_(self.item_projection.bias)

    def encode_user(
        self,
        history_items: torch.Tensor,
        history_types: torch.Tensor,
        history_times: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        sequence = (
            self.item_embedding(history_items)
            + self.history_type_embedding(history_types)
            + self.time_embedding(history_times)
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        item_vectors = self.item_embedding(history_items)
        mask = history_items.ne(0).unsqueeze(2)
        mean_history = (item_vectors * mask).sum(1) / lengths.clamp_min(1).unsqueeze(1)
        return F.normalize(self.user_projection(hidden[-1] + mean_history), dim=1)

    def encode_item(
        self, items: torch.Tensor, categories: torch.Tensor, roots: torch.Tensor
    ) -> torch.Tensor:
        vector = (
            self.item_embedding(items)
            + self.category_embedding(categories)
            + self.root_embedding(roots)
        )
        return F.normalize(self.item_projection(vector), dim=1)


def causal_inbatch_mask(target_aids: torch.Tensor, target_ts: torch.Tensor) -> torch.Tensor:
    """Mask future and duplicate in-batch items while retaining each diagonal positive."""
    visible = target_ts[None, :] <= target_ts[:, None]
    duplicates = target_aids[None, :] == target_aids[:, None]
    mask = visible & ~duplicates
    mask.fill_diagonal_(True)
    return mask


def train_two_tower(
    model: TwoTowerModel,
    dataset: SequenceDataset,
    *,
    epochs: int = 3,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    temperature: float = 0.07,
    seed: int = 2026,
    device: str = "cpu",
    validation_callback=None,
    min_epochs: int = 3,
    patience: int = 2,
    min_delta: float = 0.0002,
) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences,
        generator=generator, num_workers=0,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses = []
    validation_history = []
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    action_weights = ACTION_WEIGHTS.to(device)
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                users = model.encode_user(
                    batch["history_items"], batch["history_types"], batch["history_times"], batch["lengths"]
                )
                items = model.encode_item(
                    batch["target_item"], batch["target_category"], batch["target_root"]
                )
                logits = users @ items.T / temperature
                mask = causal_inbatch_mask(batch["target_aid"], batch["target_ts"])
                logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                labels = torch.arange(len(logits), device=device)
                row_loss = F.cross_entropy(logits, labels, reduction="none")
                loss = (row_loss * action_weights[batch["target_type"]]).mean()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(logits)
            total_rows += len(logits)
        losses.append(total_loss / max(total_rows, 1))
        if validation_callback is not None:
            report = dict(validation_callback(model, len(losses)))
            report["epoch"] = len(losses)
            validation_history.append(report)
            score = float(report["selection_score"])
            if score > best_score + min_delta:
                best_score = score
                best_epoch = len(losses)
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if len(losses) >= min_epochs and stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.early_stopping_report = {
        "enabled": validation_callback is not None,
        "best_epoch": best_epoch if validation_callback is not None else len(losses),
        "best_selection_score": best_score if validation_callback is not None else None,
        "validation_history": validation_history,
        "stopped_epoch": len(losses),
    }
    return losses


def attach_target_categories(samples: pd.DataFrame, events_enriched: pd.DataFrame) -> pd.DataFrame:
    target_features = events_enriched[
        ["session", "aid", "ts", "categoryid", "root_categoryid"]
    ].rename(
        columns={
            "aid": "target_aid", "ts": "target_ts", "categoryid": "target_categoryid",
            "root_categoryid": "target_root_categoryid",
        }
    )
    target_features = target_features.drop_duplicates(["session", "target_aid", "target_ts"])
    output = samples.merge(
        target_features, on=["session", "target_aid", "target_ts"], how="left", validate="many_to_one"
    )
    output[["target_categoryid", "target_root_categoryid"]] = output[
        ["target_categoryid", "target_root_categoryid"]
    ].fillna(-1).astype(int)
    return output
