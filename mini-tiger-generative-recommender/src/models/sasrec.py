"""独立的 SASRec Item-ID 序列推荐基线。"""

from __future__ import annotations

import math

import torch
from torch import nn


class SASRec(nn.Module):
    """使用因果自注意力编码行为历史，并用共享 Item Embedding 打分。"""

    def __init__(
        self,
        num_items: int,
        max_history: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        self.num_items = num_items
        self.max_history = max_history
        self.num_heads = num_heads
        self.item_embedding = nn.Embedding(
            num_items + 1, hidden_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(max_history, hidden_dim)
        # 0=padding, 1=negative event, 2=positive event.
        self.feedback_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_bias = nn.Parameter(torch.zeros(num_items))

    def encode_sequence(
        self,
        history_tokens: torch.Tensor,
        feedback_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """编码每个历史位置，显式区分正、负反馈与 padding。"""
        _, sequence_length = history_tokens.shape
        if sequence_length > self.max_history:
            raise ValueError("history is longer than configured max_history")
        positions = torch.arange(sequence_length, device=history_tokens.device)
        hidden = self.item_embedding(history_tokens) * math.sqrt(
            self.item_embedding.embedding_dim
        )
        if feedback_types is None:
            # 兼容旧 positive-only 数据；真实事件均视为正反馈。
            feedback_types = history_tokens.ne(0).long() * 2
        if feedback_types.shape != history_tokens.shape:
            raise ValueError("feedback types must match history tokens")
        hidden = self.input_dropout(
            hidden
            + self.position_embedding(positions)
            + self.feedback_embedding(feedback_types)
        )
        padding_mask = history_tokens.eq(0)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=history_tokens.device,
            ),
            diagonal=1,
        )
        if padding_mask.any():
            # 左 padding 与 causal mask 直接叠加时，最左侧 padding query 可能没有
            # 任何合法 key，某些 PyTorch/CUDA 内核会因此产生整行 NaN。构造合并
            # mask，并仅让这种 padding query 关注自己；真实 token 仍完全看不到
            # padding，因此不改变有效历史的注意力语义。
            batch_size = history_tokens.shape[0]
            combined = causal_mask[None].expand(
                batch_size, sequence_length, sequence_length
            ).clone()
            combined |= padding_mask[:, None, :]
            fully_masked = combined.all(dim=2)
            batch_rows, query_rows = fully_masked.nonzero(as_tuple=True)
            combined[batch_rows, query_rows, query_rows] = False
            attention_mask = combined.repeat_interleave(self.num_heads, dim=0)
            encoded = self.encoder(hidden, mask=attention_mask)
        else:
            encoded = self.encoder(hidden, mask=causal_mask)
        return self.output_norm(encoded)

    def encode_history(
        self,
        history_tokens: torch.Tensor,
        feedback_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_sequence(history_tokens, feedback_types)[:, -1]

    def score_all_items(self, state: torch.Tensor) -> torch.Tensor:
        """精确打分完整目录；支持 [B,D] 或 [B,L,D]。"""
        return (
            state @ self.item_embedding.weight[1:].transpose(0, 1)
            + self.output_bias
        )

    def score_candidates(
        self, state: torch.Tensor, candidate_items: torch.Tensor
    ) -> torch.Tensor:
        """只打分每个用户的候选 Item，避免全目录矩阵乘法。

        参数:
            state: [batch, hidden_dim] 用户状态。
            candidate_items: [batch, candidates]，使用从 0 开始的 item index。
        """
        embeddings = self.item_embedding(candidate_items + 1)
        scores = torch.einsum("bd,bkd->bk", state, embeddings)
        return scores + self.output_bias[candidate_items]

    def forward(
        self,
        history_tokens: torch.Tensor,
        feedback_types: torch.Tensor | None = None,
        *,
        all_positions: bool = False,
    ) -> torch.Tensor:
        states = self.encode_sequence(history_tokens, feedback_types)
        return self.score_all_items(states if all_positions else states[:, -1])
