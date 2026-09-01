"""Feature-rich two-tower retriever with deduplicated hard negatives."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class TwoTowerV2Dataset(Dataset):
    def __init__(self, samples: pd.DataFrame, item_to_index: dict[int, int], hard_negatives: np.ndarray):
        self.samples = samples.reset_index(drop=True)
        self.item_to_index = item_to_index
        self.hard_negatives = hard_negatives

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples.iloc[index]
        aids = list(row.history_aids)[-30:]
        types = list(row.history_type_ids)[-30:]
        deltas = list(row.history_time_deltas_ms)[-30:]
        return {
            "history_items": [self.item_to_index.get(int(a), 0) for a in aids],
            "history_types": [int(t) + 1 for t in types],
            "history_times": [min(31, int(math.log2(max(int(t) // 1000, 0) + 1))) + 1 for t in deltas],
            "target_item": self.item_to_index.get(int(row.target_aid), 0),
            "target_aid": int(row.target_aid), "target_type": int(row.target_type_id),
            "target_ts": int(row.target_ts), "session": int(row.session),
            "hard_negatives": self.hard_negatives[index].astype(np.int64),
        }


def collate_v2(rows):
    length = max(len(r["history_items"]) for r in rows)
    out = {
        "history_items": torch.zeros((len(rows), length), dtype=torch.long),
        "history_types": torch.zeros((len(rows), length), dtype=torch.long),
        "history_times": torch.zeros((len(rows), length), dtype=torch.long),
        "lengths": torch.tensor([len(r["history_items"]) for r in rows]),
    }
    for i, row in enumerate(rows):
        n = len(row["history_items"])
        out["history_items"][i, :n] = torch.tensor(row["history_items"])
        out["history_types"][i, :n] = torch.tensor(row["history_types"])
        out["history_times"][i, :n] = torch.tensor(row["history_times"])
    for key in ("target_item", "target_aid", "target_type", "target_ts", "session"):
        out[key] = torch.tensor([r[key] for r in rows], dtype=torch.long)
    out["hard_negatives"] = torch.tensor(np.stack([r["hard_negatives"] for r in rows]), dtype=torch.long)
    return out


class TwoTowerV2(nn.Module):
    def __init__(self, features: dict[str, torch.Tensor], item2vec_vectors: torch.Tensor):
        super().__init__()
        item_count = len(features["category"]) - 1
        self.item_embedding = nn.Embedding(item_count + 1, 64, padding_idx=0)
        self.type_embedding = nn.Embedding(4, 16, padding_idx=0)
        self.time_embedding = nn.Embedding(34, 16, padding_idx=0)
        self.event_projection = nn.Linear(96, 128)
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.mean_projection = nn.Linear(64, 128)
        self.user_projection = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64))

        self.item2vec_embedding = nn.Embedding.from_pretrained(item2vec_vectors, freeze=False, padding_idx=0)
        self.item2vec_projection = nn.Linear(item2vec_vectors.shape[1], 64)
        self.leaf_embedding = nn.Embedding(int(features["category"].max()) + 1, 32, padding_idx=0)
        self.parent_embedding = nn.Embedding(int(features["parent"].max()) + 1, 32, padding_idx=0)
        self.root_embedding = nn.Embedding(int(features["root"].max()) + 1, 32, padding_idx=0)
        self.depth_embedding = nn.Embedding(int(features["depth"].max()) + 1, 8, padding_idx=0)
        self.category_projection = nn.Linear(104, 64)
        self.availability_embedding = nn.Embedding(3, 8)
        self.stats_projection = nn.Sequential(
            nn.Linear(features["stats"].shape[1] + 8, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.item_mlp = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64)
        )
        for name, value in features.items():
            self.register_buffer(f"feature_{name}", value)

    def encode_user(self, items, types, times, lengths):
        item_vectors = self.item_embedding(items)
        sequence = torch.cat([item_vectors, self.type_embedding(types), self.time_embedding(times)], dim=-1)
        sequence = self.event_projection(sequence)
        packed = nn.utils.rnn.pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        mask = items.ne(0).unsqueeze(-1)
        mean = (item_vectors * mask).sum(1) / lengths.clamp_min(1).unsqueeze(1)
        return F.normalize(self.user_projection(hidden[-1] + self.mean_projection(mean)), dim=-1)

    def encode_items(self, indices):
        shape = indices.shape
        flat = indices.reshape(-1)
        item = self.item_embedding(flat)
        pretrained = self.item2vec_projection(self.item2vec_embedding(flat))
        category = self.category_projection(torch.cat([
            self.leaf_embedding(self.feature_category[flat]),
            self.parent_embedding(self.feature_parent[flat]),
            self.root_embedding(self.feature_root[flat]),
            self.depth_embedding(self.feature_depth[flat]),
        ], dim=-1))
        stats = self.stats_projection(torch.cat([
            self.feature_stats[flat], self.availability_embedding(self.feature_availability[flat])
        ], dim=-1))
        vector = self.item_mlp(torch.cat([item, pretrained, category, stats], dim=-1)) + item
        return F.normalize(vector, dim=-1).reshape(*shape, 64)


def causal_mask(target_aids, target_ts):
    visible = target_ts[None, :] <= target_ts[:, None]
    duplicate = target_aids[None, :] == target_aids[:, None]
    mask = visible & ~duplicate
    mask.fill_diagonal_(True)
    return mask
