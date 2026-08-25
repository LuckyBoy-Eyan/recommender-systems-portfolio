"""实现Shared-Bottom多目标排序器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F


ACTIONS = ("clicks", "carts", "orders")
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}


def _needs_log1p(column: str) -> bool:
    """判断一个非负长尾特征是否需要在标准化前执行 ``log1p``。"""
    return column in {
        "item_events",
        "item_sessions",
        "item_clicks",
        "item_carts",
        "item_orders",
        "session_length",
        "session_unique_items",
        "in_session_count",
        "seconds_since_seen",
    }


@dataclass
class NeuralFeaturePreprocessor:
    """保存神经排序器训练阶段拟合的特征变换状态。"""

    columns: list[str]
    log_columns: list[str]
    scaler: StandardScaler

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "NeuralFeaturePreprocessor":
        """仅使用训练候选拟合 ``log1p`` 列和StandardScaler。"""
        log_columns = [column for column in columns if _needs_log1p(column)]
        values = frame.reindex(columns=columns, fill_value=0).astype(np.float32).copy()
        for column in log_columns:
            values[column] = np.log1p(values[column].clip(lower=0))
        scaler = StandardScaler()
        scaler.fit(values)
        return cls(columns=columns, log_columns=log_columns, scaler=scaler)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """按照训练时完全相同的列顺序和统计量转换候选特征。"""
        values = frame.reindex(columns=self.columns, fill_value=0).astype(np.float32).copy()
        for column in self.log_columns:
            values[column] = np.log1p(values[column].clip(lower=0))
        return self.scaler.transform(values).astype(np.float32)


class SharedBottom(nn.Module):
    """共享底层表示、任务塔独立的Shared-Bottom模型。"""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, int]):
        """创建共享底层和三个任务塔。

        参数：
            input_dim:
                候选特征维度。
            hidden_dims:
                共享底层两层隐藏维度。第二层输出同时送入点击、加购、购买三个线性塔。

        调用关系：
            由 ``_build_neural_model`` 在 ``method=shared_bottom`` 时实例化；这是当前
            ``configs/retailrocket.json`` 选择的正式排序结构。
        """
        super().__init__()
        first, second = hidden_dims
        self.bottom = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
        )
        self.towers = nn.ModuleList([nn.Linear(second, 1) for _ in ACTIONS])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """对共享表示分别输出三个任务Logit。"""
        shared = self.bottom(features)
        return torch.cat([tower(shared) for tower in self.towers], dim=1)


@dataclass
class NeuralRankerBundle:
    """封装神经模型、特征预处理器及训练审计信息。"""

    method: str
    model: nn.Module
    preprocessor: NeuralFeaturePreprocessor
    positive_weights: list[float]
    observed_rows: list[int]
    positive_rows: list[int]


def build_task_targets(labeled: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """把单任务观测行转换为三任务标签与Mask。

    每行只对 ``target_type`` 对应任务设置 ``mask=1``；其他任务属于未观测，
    不能错误地写成负标签。
    """
    labels = torch.zeros((len(labeled), len(ACTIONS)), dtype=torch.float32)
    masks = torch.zeros_like(labels, dtype=torch.bool)
    action_indices = labeled["target_type"].map(ACTION_TO_INDEX)
    if action_indices.isna().any():
        unknown = sorted(labeled.loc[action_indices.isna(), "target_type"].unique())
        raise ValueError(f"存在未知目标行为类型: {unknown}")
    rows = torch.arange(len(labeled))
    columns = torch.tensor(action_indices.to_numpy(), dtype=torch.long)
    labels[rows, columns] = torch.tensor(
        labeled["label"].to_numpy(), dtype=torch.float32
    )
    masks[rows, columns] = True
    return labels, masks


def _build_neural_model(
    method: str,
    input_dim: int,
    hidden_dims: tuple[int, int],
) -> nn.Module:
    """创建正式Shared-Bottom排序器。

    参数：
        method:
            必须为 ``shared_bottom``。
        input_dim:
            预处理后的候选特征维度。
        hidden_dims:
            两层隐藏维度。
    返回：
        尚未训练的PyTorch模型。

    调用关系：
        仅由 ``train_neural_ranker`` 调用。
    """
    if method == "shared_bottom":
        return SharedBottom(input_dim, hidden_dims)
    raise ValueError("排序方法只支持 shared_bottom")


def train_neural_ranker(
    labeled: pd.DataFrame,
    feature_columns: list[str],
    method: str,
    seed: int,
    hidden_dims: tuple[int, int] = (64, 32),
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
) -> NeuralRankerBundle:
    """用任务Mask和任务内类别权重训练Shared-Bottom。

    每个任务的BCE只在对应 ``mask=1`` 的候选上计算，并使用该任务训练负例数/正例数
    作为 ``pos_weight``。
    """
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("神经排序器训练参数非法")
    if len(hidden_dims) != 2 or min(hidden_dims) < 1:
        raise ValueError("hidden_dims必须包含两个正整数")

    preprocessor = NeuralFeaturePreprocessor.fit(labeled, feature_columns)
    features = torch.tensor(preprocessor.transform(labeled), dtype=torch.float32)
    labels, masks = build_task_targets(labeled)
    positive_weights = []
    observed_rows = []
    positive_rows = []
    for task_index in range(len(ACTIONS)):
        task_labels = labels[masks[:, task_index], task_index]
        positives = int(task_labels.sum().item())
        negatives = int(len(task_labels) - positives)
        positive_weights.append(float(negatives / max(positives, 1)))
        observed_rows.append(int(len(task_labels)))
        positive_rows.append(positives)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    model = _build_neural_model(method, features.shape[1], hidden_dims)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    try:
        model.train()
        for _ in range(epochs):
            permutation = torch.randperm(len(features), generator=generator)
            for start in range(0, len(features), batch_size):
                indices = permutation[start : start + batch_size]
                batch_logits = model(features[indices])
                batch_labels = labels[indices]
                batch_masks = masks[indices]
                task_losses = []
                for task_index in range(len(ACTIONS)):
                    task_mask = batch_masks[:, task_index]
                    if not task_mask.any():
                        continue
                    task_losses.append(
                        F.binary_cross_entropy_with_logits(
                            batch_logits[task_mask, task_index],
                            batch_labels[task_mask, task_index],
                            pos_weight=torch.tensor(positive_weights[task_index]),
                        )
                    )
                loss = torch.stack(task_losses).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    finally:
        torch.set_num_threads(previous_threads)
    model.eval()
    return NeuralRankerBundle(
        method=method,
        model=model,
        preprocessor=preprocessor,
        positive_weights=positive_weights,
        observed_rows=observed_rows,
        positive_rows=positive_rows,
    )


def score_neural_candidates(
    bundle: NeuralRankerBundle, features: pd.DataFrame, action: str
) -> pd.DataFrame:
    """使用指定任务输出候选概率并在Session内降序排列。"""
    if action not in ACTION_TO_INDEX:
        raise ValueError(f"未知任务: {action}")
    transformed = torch.tensor(
        bundle.preprocessor.transform(features), dtype=torch.float32
    )
    with torch.no_grad():
        probabilities = torch.sigmoid(bundle.model(transformed))[
            :, ACTION_TO_INDEX[action]
        ].numpy()
    scored = features[["session", "aid"]].copy()
    scored["score"] = probabilities
    return scored.sort_values(
        ["session", "score", "aid"], ascending=[True, False, True]
    )
