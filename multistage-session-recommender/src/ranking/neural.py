"""实现Shared-Bottom多目标排序器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
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
    scaler: object

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "NeuralFeaturePreprocessor":
        """仅使用训练候选拟合 ``log1p`` 列和StandardScaler。"""
        from sklearn.preprocessing import StandardScaler

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


def _expert(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())


class MMoE(nn.Module):
    """Multi-gate Mixture-of-Experts with one gate and tower per action."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, experts: int = 8,
        category_vocab_size: int = 0, type_vocab_size: int = 0,
        category_embedding_dim: int = 16, type_embedding_dim: int = 8,
    ):
        super().__init__()
        self.category_embedding = nn.Embedding(category_vocab_size, category_embedding_dim, padding_idx=0) if category_vocab_size else None
        self.type_embedding = nn.Embedding(type_vocab_size, type_embedding_dim, padding_idx=0) if type_vocab_size else None
        combined_dim = input_dim + (category_embedding_dim if category_vocab_size else 0) + (type_embedding_dim if type_vocab_size else 0)
        self.experts = nn.ModuleList([_expert(combined_dim, hidden_dim) for _ in range(experts)])
        self.gates = nn.ModuleList([nn.Linear(combined_dim, experts) for _ in ACTIONS])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in ACTIONS
        ])

    def forward(self, features: torch.Tensor, category=None, event_type=None) -> torch.Tensor:
        if self.category_embedding is not None:
            features = torch.cat([features, self.category_embedding(category)], dim=1)
        if self.type_embedding is not None:
            features = torch.cat([features, self.type_embedding(event_type)], dim=1)
        expert_values = torch.stack([expert(features) for expert in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(features), dim=1).unsqueeze(2)
            outputs.append(tower((expert_values * weights).sum(1)))
        return torch.cat(outputs, dim=1)


class PLE(nn.Module):
    """Single-level PLE with shared and task-specific experts."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 64,
        shared_experts: int = 2, task_experts: int = 2,
        category_vocab_size: int = 0, type_vocab_size: int = 0,
        category_embedding_dim: int = 16, type_embedding_dim: int = 8,
    ):
        super().__init__()
        self.category_embedding = nn.Embedding(category_vocab_size, category_embedding_dim, padding_idx=0) if category_vocab_size else None
        self.type_embedding = nn.Embedding(type_vocab_size, type_embedding_dim, padding_idx=0) if type_vocab_size else None
        combined_dim = input_dim + (category_embedding_dim if category_vocab_size else 0) + (type_embedding_dim if type_vocab_size else 0)
        self.shared = nn.ModuleList([_expert(combined_dim, hidden_dim) for _ in range(shared_experts)])
        self.specific = nn.ModuleList([
            nn.ModuleList([_expert(combined_dim, hidden_dim) for _ in range(task_experts)])
            for _ in ACTIONS
        ])
        gate_width = shared_experts + task_experts
        self.gates = nn.ModuleList([nn.Linear(combined_dim, gate_width) for _ in ACTIONS])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in ACTIONS
        ])

    def forward(self, features: torch.Tensor, category=None, event_type=None) -> torch.Tensor:
        if self.category_embedding is not None:
            features = torch.cat([features, self.category_embedding(category)], dim=1)
        if self.type_embedding is not None:
            features = torch.cat([features, self.type_embedding(event_type)], dim=1)
        shared_values = [expert(features) for expert in self.shared]
        outputs = []
        for task, (gate, tower) in enumerate(zip(self.gates, self.towers)):
            values = torch.stack(shared_values + [expert(features) for expert in self.specific[task]], dim=1)
            weights = torch.softmax(gate(features), dim=1).unsqueeze(2)
            outputs.append(tower((values * weights).sum(1)))
        return torch.cat(outputs, dim=1)


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
    """把下一次物品-行为观测转换为三个联合监督标签。

    正例候选仅在真实下一行为对应任务上为1；其余任务以及所有非目标候选为0。
    三个任务均参与训练，使三个塔学习同一候选集合上的下一物品-行为联合分布。
    """
    labels = torch.zeros((len(labeled), len(ACTIONS)), dtype=torch.float32)
    masks = torch.ones_like(labels, dtype=torch.bool)
    action_indices = labeled["target_type"].map(ACTION_TO_INDEX)
    if action_indices.isna().any():
        unknown = sorted(labeled.loc[action_indices.isna(), "target_type"].unique())
        raise ValueError(f"存在未知目标行为类型: {unknown}")
    rows = torch.arange(len(labeled))
    columns = torch.tensor(action_indices.to_numpy(), dtype=torch.long)
    positive = torch.tensor(labeled["label"].to_numpy(), dtype=torch.float32)
    # clicks塔覆盖全部正交互；carts塔覆盖加购和购买；orders塔仅覆盖购买。
    labels[:, 0] = positive
    labels[:, 1] = positive * (columns >= ACTION_TO_INDEX["carts"]).float()
    labels[:, 2] = positive * (columns >= ACTION_TO_INDEX["orders"]).float()
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
    if method == "mmoe":
        return MMoE(input_dim, hidden_dims[0])
    if method == "ple":
        return PLE(input_dim, hidden_dims[0])
    raise ValueError("排序方法只支持 shared_bottom/mmoe/ple")


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
    """用三任务联合标签和任务内类别权重训练神经排序器。

    每行同时监督三个任务，并使用各任务训练负例数/正例数作为 ``pos_weight``。
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
