"""候选级学习式精排器。"""

from __future__ import annotations

import torch
from torch import nn


class CandidateRanker(nn.Module):
    """用共享 MLP 为同一用户的每个候选物品计算精排分数。

    输入形状为 ``[batch, candidates, feature_dim]``。模型只学习候选特征之间
    的非线性组合，不重新编码用户历史，因此参数少、训练快，也方便在工业链路中
    独立更新。
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        base_alpha: float = 0.25,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if feature_dim < 2:
            raise ValueError("feature_dim must include generation and SASRec scores")
        if not 0.0 <= base_alpha <= 1.0:
            raise ValueError("base_alpha must be between zero and one")
        self.base_alpha = base_alpha
        self.residual_scale = residual_scale
        layers: list[nn.Module] = []
        current = feature_dim
        for hidden in hidden_dims:
            if hidden <= 0:
                raise ValueError("hidden dimensions must be positive")
            layers.extend(
                [nn.Linear(current, hidden), nn.GELU(), nn.Dropout(dropout)]
            )
            current = hidden
        output = nn.Linear(current, 1)
        # 初始输出严格等于已经验证过的固定权重融合；学习器只拟合残差。
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """返回每个候选的分数；padding 位置被设为负无穷。"""
        base = (
            self.base_alpha * features[..., 0]
            + (1.0 - self.base_alpha) * features[..., 1]
        )
        scores = base + self.residual_scale * self.network(features).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -torch.inf)
        return scores
