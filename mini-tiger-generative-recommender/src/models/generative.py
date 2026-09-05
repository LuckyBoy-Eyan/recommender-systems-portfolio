"""TIGER 风格的 Semantic ID Transformer Encoder-Decoder。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EncodedHistory:
    """可被候选打分和 Beam Search 复用的历史编码。"""

    memory: torch.Tensor
    padding_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.memory.shape[0])

    def repeat_interleave(self, repeats: int) -> "EncodedHistory":
        return EncodedHistory(
            self.memory.repeat_interleave(repeats, dim=0),
            self.padding_mask.repeat_interleave(repeats, dim=0),
        )

    def select(self, index: int) -> "EncodedHistory":
        return EncodedHistory(
            self.memory[index : index + 1],
            self.padding_mask[index : index + 1],
        )


class SemanticIDTransformer(nn.Module):
    """编码用户历史，并用 Transformer Decoder 自回归生成 Semantic ID。"""

    def __init__(
        self,
        codebook_sizes: list[int],
        max_history: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int | None = None,
        decoder_layers: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not codebook_sizes:
            raise ValueError("codebook_sizes must not be empty")
        self.codebook_sizes = tuple(int(size) for size in codebook_sizes)
        self.num_code_levels = len(self.codebook_sizes)
        self.hidden_dim = int(hidden_dim)

        self.code_embeddings = nn.ModuleList(
            [
                nn.Embedding(size + 1, hidden_dim, padding_idx=0)
                for size in self.codebook_sizes
            ]
        )
        self.history_position = nn.Embedding(max_history, hidden_dim)
        # 0=padding, 1=negative event, 2=positive event.
        self.feedback_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            feedforward_dim or hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
            enable_nested_tensor=False,
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        self.target_embeddings = nn.ModuleList(
            [nn.Embedding(size, hidden_dim) for size in self.codebook_sizes]
        )
        self.decoder_level_embedding = nn.Embedding(
            self.num_code_levels, hidden_dim
        )
        self.start_token = nn.Parameter(torch.zeros(hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim,
            num_heads,
            feedforward_dim or hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            decoder_layers if decoder_layers is not None else num_layers,
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, size) for size in self.codebook_sizes]
        )

    def encode_history(
        self,
        history_codes: torch.Tensor,
        feedback_types: torch.Tensor | None = None,
    ) -> EncodedHistory:
        """把 ``[B, history, levels]`` 编码为 Decoder 可交叉注意的 memory。"""
        _, sequence_length, _ = history_codes.shape
        hidden = sum(
            embedding(history_codes[:, :, level])
            for level, embedding in enumerate(self.code_embeddings)
        )
        positions = torch.arange(sequence_length, device=history_codes.device)
        padding_mask = history_codes.sum(dim=-1).eq(0)
        if feedback_types is None:
            feedback_types = (~padding_mask).long() * 2
        if feedback_types.shape != padding_mask.shape:
            raise ValueError("feedback types must match history sequence")
        hidden = (
            hidden
            + self.history_position(positions)
            + self.feedback_embedding(feedback_types)
        )
        memory = self.encoder(
            hidden,
            src_key_padding_mask=padding_mask if padding_mask.any() else None,
        )
        return EncodedHistory(self.encoder_norm(memory), padding_mask)

    def _decoder_inputs(
        self,
        batch_size: int,
        prefix_codes: torch.Tensor | None,
        target_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        rows = [self.start_token.expand(batch_size, -1)]
        prefix_length = 0 if prefix_codes is None else prefix_codes.shape[1]
        for position in range(1, target_length):
            if position - 1 >= prefix_length:
                raise ValueError("prefix does not contain required previous token")
            rows.append(
                self.target_embeddings[position - 1](
                    prefix_codes[:, position - 1]
                )
            )
        inputs = torch.stack(rows, dim=1)
        levels = torch.arange(target_length, device=device)
        return inputs + self.decoder_level_embedding(levels)[None, :, :]

    def _decode_hidden(
        self,
        encoded: EncodedHistory,
        prefix_codes: torch.Tensor | None,
        target_length: int,
    ) -> torch.Tensor:
        inputs = self._decoder_inputs(
            encoded.batch_size,
            prefix_codes,
            target_length,
            encoded.memory.device,
        )
        causal_mask = torch.triu(
            torch.ones(
                target_length,
                target_length,
                dtype=torch.bool,
                device=encoded.memory.device,
            ),
            diagonal=1,
        )
        hidden = self.decoder(
            inputs,
            encoded.memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=(
                encoded.padding_mask if encoded.padding_mask.any() else None
            ),
        )
        return self.decoder_norm(hidden)

    def next_token_logits(
        self,
        encoded: EncodedHistory,
        prefix_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """根据已生成前缀返回下一层 Token logits。"""
        level = 0 if prefix_codes is None else int(prefix_codes.shape[1])
        if level >= self.num_code_levels:
            raise ValueError("prefix already contains a complete Semantic ID")
        if prefix_codes is not None and prefix_codes.shape[0] != encoded.batch_size:
            raise ValueError("prefix and encoded history batch sizes differ")
        hidden = self._decode_hidden(encoded, prefix_codes, level + 1)
        return self.heads[level](hidden[:, level])

    def decode(
        self,
        encoded: EncodedHistory,
        target_codes: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Teacher forcing 训练，或在未提供目标时执行贪心自回归生成。"""
        if target_codes is not None:
            expected = (encoded.batch_size, self.num_code_levels)
            if target_codes.shape != expected:
                raise ValueError("target code shape differs from model code levels")
            hidden = self._decode_hidden(
                encoded,
                target_codes,
                self.num_code_levels,
            )
            return [
                head(hidden[:, level]) for level, head in enumerate(self.heads)
            ]

        logits: list[torch.Tensor] = []
        generated: list[torch.Tensor] = []
        for _ in range(self.num_code_levels):
            prefix = torch.stack(generated, dim=1) if generated else None
            current = self.next_token_logits(encoded, prefix)
            logits.append(current)
            generated.append(current.argmax(dim=-1))
        return logits

    def forward(
        self,
        history_codes: torch.Tensor,
        target_codes: torch.Tensor | None = None,
        feedback_types: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return self.decode(
            self.encode_history(history_codes, feedback_types), target_codes
        )
