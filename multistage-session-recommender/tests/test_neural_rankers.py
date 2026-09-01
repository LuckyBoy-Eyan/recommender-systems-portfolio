"""验证神经排序器的三任务联合监督、训练及统一输出。"""

import numpy as np
import pandas as pd
import torch

from src.ranking.neural import (
    ACTIONS,
    SharedBottom,
    build_task_targets,
    score_neural_candidates,
    train_neural_ranker,
)


def _make_labeled_candidates() -> pd.DataFrame:
    """构造每个任务各含一个正例和三个负例的最小训练表。"""
    rows = []
    for task_index, action in enumerate(ACTIONS):
        for candidate_index in range(4):
            rows.append(
                {
                    "session": task_index,
                    "aid": task_index * 10 + candidate_index,
                    "feature_a": float(candidate_index),
                    "feature_b": float(task_index + candidate_index),
                    "target_type": action,
                    "label": int(candidate_index == 0),
                }
            )
    return pd.DataFrame(rows)


def test_multitask_targets_jointly_supervise_all_three_tasks():
    """每行应同时监督三塔，正例只落在真实下一行为对应的塔。"""
    labeled = _make_labeled_candidates()
    labels, masks = build_task_targets(labeled)
    assert labels.shape == masks.shape == (12, 3)
    assert masks.sum(dim=1).tolist() == [3] * 12
    assert labels.sum(dim=0).tolist() == [3.0, 2.0, 1.0]
    assert masks.sum(dim=0).tolist() == [12, 12, 12]


def test_shared_bottom_returns_three_task_logits():
    """Shared-Bottom必须输出点击、加购、购买三个Logit。"""
    features = torch.zeros((5, 7))
    model = SharedBottom(7, (8, 4))
    assert model(features).shape == (5, 3)


def test_shared_bottom_training_and_scoring_are_deterministic():
    """相同样本、参数和种子应产生完全一致的Shared-Bottom候选分数。"""
    labeled = _make_labeled_candidates()
    parameters = {
        "labeled": labeled,
        "feature_columns": ["feature_a", "feature_b"],
        "method": "shared_bottom",
        "seed": 11,
        "hidden_dims": (8, 4),
        "epochs": 2,
        "batch_size": 6,
        "learning_rate": 0.01,
    }
    first = train_neural_ranker(**parameters)
    second = train_neural_ranker(**parameters)
    first_scores = score_neural_candidates(first, labeled, "clicks")["score"].to_numpy()
    second_scores = score_neural_candidates(second, labeled, "clicks")["score"].to_numpy()
    assert np.array_equal(first_scores, second_scores)
