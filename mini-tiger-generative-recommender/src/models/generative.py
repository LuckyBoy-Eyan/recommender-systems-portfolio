"""MiniTIGER 的历史编码器与自回归 Semantic ID 解码器。

调用关系：
    scripts/run_demo.py 实例化 SemanticIDTransformer
        ├─ training/train.py 调用 forward(history, target_codes) 做 teacher forcing
        └─ training/evaluate.py 调用 forward(history, candidate_codes)
           计算目录中每条合法完整编码的自回归概率

模型不是生成标题文本，而是逐级生成类似 ``[粗类, 细类, tail]`` 的物品编码。
"""

from __future__ import annotations

import torch
from torch import nn


class SemanticIDTransformer(nn.Module):
    """
    基于 Semantic ID 的生成式序列推荐模型。

    模型先用 Transformer Encoder 编码用户历史行为序列，再用 GRUCell 作为
    自回归解码器，按层生成下一个物品的 Semantic ID token。
    """

    def __init__(
        self,
        codebook_sizes: list[int],
        max_history: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int | None = None,
    ):
        """
        初始化历史编码器和自回归 Semantic ID 解码器。

        参数:
            codebook_sizes: 每一级 Semantic ID 的词表大小，包含最后的 tail token 层。
            max_history: 输入历史序列的最大长度。
            hidden_dim: token embedding 和模型隐状态维度。
            num_heads: Transformer Encoder 的注意力头数。
            num_layers: Transformer Encoder 层数。
            feedforward_dim: Transformer 前馈层宽度；None 时使用
                ``hidden_dim * 4``。等容量实验可独立调整它，使生成模型与
                SASRec 参数量接近而仍保持 Tensor Core 友好的 hidden_dim。

        创建的子模块:
            code_embeddings: 把历史中每一级 ID token 映射成 hidden_dim 维向量。
            position: 历史序列位置 embedding。
            encoder: 编码用户历史的 Transformer Encoder。
            heads: 每一级输出词表各自对应一个分类头。
            target_embeddings: 把已生成 token 变成下一解码步的输入。
            decoder_cell: 在各 ID 层级之间传递状态的 GRUCell。
            start_token: 第一个解码步使用的可学习起始输入。

        调用:
            scripts/run_demo.py 分别创建 Semantic ID 模型和 Random ID 模型。
        """
        super().__init__()
        # 历史 item 的每一级 code 都有自己的 embedding。
        # size + 1 是为了给 padding 预留 0，真实 code 在 dataset 中会整体 +1。
        self.code_embeddings = nn.ModuleList(
            [nn.Embedding(size + 1, hidden_dim) for size in codebook_sizes]
        )
        # 位置 embedding 让 Transformer 知道行为发生在历史序列中的相对位置。
        self.position = nn.Embedding(max_history, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            feedforward_dim or hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        # Transformer Encoder 用来从用户历史 Semantic ID 序列中提取兴趣表示。
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        # 每一级 Semantic ID token 都对应一个分类头。
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, size) for size in codebook_sizes])
        # 解码时，上一层已经生成的 token 会被映射成 embedding，作为下一步输入。
        self.target_embeddings = nn.ModuleList(
            [nn.Embedding(size, hidden_dim) for size in codebook_sizes]
        )
        # GRUCell 负责自回归解码：每生成一级 token，就更新一次解码状态。
        self.decoder_cell = nn.GRUCell(hidden_dim, hidden_dim)
        # 第一级 token 没有“上一层 token”，因此使用一个可学习的起始向量。
        self.start_token = nn.Parameter(torch.zeros(hidden_dim))

    def encode_history(self, history_codes):
        """只运行一次 Transformer，得到用户当前兴趣状态。

        参数:
            history_codes: LongTensor，形状
                [batch_size, max_history, num_levels]。

        返回:
            state: FloatTensor，形状 [batch_size, hidden_dim]。

        评估万级 KuaiRec 目录时，同一用户需要给许多候选物品打分。把历史编码
        与候选解码拆开后，Transformer 不必为每个候选重复计算。
        """
        _, seq_len, _ = history_codes.shape
        # 同一个历史 item 有多级 code；把各级 code embedding 相加得到 item 表示。
        hidden = sum(
            embedding(history_codes[:, :, level])
            for level, embedding in enumerate(self.code_embeddings)
        )
        # positions 形状是 [seq_len]，广播到 batch 中所有用户。
        positions = torch.arange(seq_len, device=history_codes.device)
        hidden = hidden + self.position(positions)
        # padding 位置的所有 code 都是 0；mask 后 Transformer 不会关注空历史。
        padding_mask = history_codes.sum(dim=-1).eq(0)
        # encoded 形状保持为 [batch_size, seq_len, hidden_dim]。
        # KuaiRec 大部分训练样本已经占满 max_history。全 False mask 没有语义作用，
        # 却会触发 Transformer 的 nested-tensor 检查与转换；此时直接省略它。
        encoded = self.encoder(
            hidden,
            src_key_padding_mask=padding_mask if padding_mask.any() else None,
        )
        # 历史采用左 padding，因此最后一个位置一定是最新的真实交互。
        return encoded[:, -1]

    def decode(self, state, target_codes=None):
        """从已经编码好的用户状态逐级生成 Semantic ID logits。

        参数:
            state: encode_history 的输出，形状 [batch_size, hidden_dim]。
            target_codes: 可选的完整目标/候选编码，形状
                [batch_size, num_levels]。传入时使用 teacher forcing；
                不传时使用每一级的 argmax token。

        返回:
            各层 logits 列表。
        """
        batch_size = state.shape[0]
        previous = self.start_token.expand(batch_size, -1)
        logits = []
        for level, head in enumerate(self.heads):
            state = self.decoder_cell(previous, state)
            level_logits = head(state)
            logits.append(level_logits)
            token = (
                target_codes[:, level]
                if target_codes is not None
                else level_logits.argmax(dim=-1)
            )
            previous = self.target_embeddings[level](token)
        return logits

    def forward(self, history_codes, target_codes=None):
        """根据用户历史生成下一个物品的 Semantic ID 各级 logits。

        参数:
            history_codes: 历史物品 code，形状为 [batch_size, seq_len, num_levels]。
                padding 位置全为 0，真实历史 code 已经整体 +1。
            target_codes: 训练时传入的真实目标 code，形状为 [batch_size, num_levels]。
                如果传入，则使用 teacher forcing；如果不传入，则从每一级
                logits 中贪心选择 argmax token。

        返回:
            logits: 一个列表，每个元素是某一级 token 的分类 logits，
                形状为 [batch_size, level_vocab_size]。

        调用:
            train_model 传入真实 target_codes，计算训练损失；
            evaluate_model 传入目录候选编码，精确计算每个候选物品的条件概率。

        自回归依赖:
            第 0 级预测 P(c0 | history)；
            第 1 级预测 P(c1 | history, c0)；
            ...
            因为上一层 token embedding 会作为下一次 GRUCell 的输入。
        """
        state = self.encode_history(history_codes)
        return self.decode(state, target_codes)
