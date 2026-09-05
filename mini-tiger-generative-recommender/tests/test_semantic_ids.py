"""MiniTIGER 最关键的正确性测试。

pytest 自动发现所有 ``test_*`` 函数并逐个调用。这里不追求覆盖所有训练效果，
而是保护最容易悄悄出错的约束：编码形状、唯一性、自回归依赖和评估样本切分。
"""

import os

# 测试时限制底层线程数，减少小机器上因为并行库冲突导致的不稳定。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import torch

from src.indexing.semantic_ids import append_collision_token, build_hierarchical_codes, collision_rate
from src.data.dataset import NextItemDataset
from src.data.split import temporal_leave_two_out
from src.models.generative import SemanticIDTransformer
from src.training.evaluate import evaluate_model_beam, evaluate_model_exact
from src.training.train import train_model


def test_semantic_id_shape_and_collision_range():
    """build_hierarchical_codes 应返回正确形状，碰撞率应是合法比例。"""
    # 随机造 20 个物品特征，验证层次 KMeans 输出形状正确。
    features = np.random.default_rng(0).normal(size=(20, 4))
    codes = build_hierarchical_codes(features, [4, 4], 0)
    assert codes.shape == (20, 2)
    # 碰撞率应该是 0 到 1 之间的比例值。
    assert 0 <= collision_rate(codes) <= 1


def test_collision_token_makes_catalog_ids_unique():
    """append_collision_token 应把重复前缀扩展成唯一完整编码。"""
    # 前缀 [0, 0] 出现了三次，原始 Semantic ID 存在碰撞。
    codes = np.asarray([[0, 0], [0, 0], [1, 0], [0, 0]])
    resolved, tail_size = append_collision_token(codes)
    # 追加 tail token 后，每个完整 code 都应该唯一。
    assert collision_rate(resolved) == 0
    # [0, 0] 这组碰撞需要 tail=0、1、2，因此 tail 词表大小是 3。
    assert tail_size == 3


def test_decoder_is_conditioned_on_previous_target_tokens():
    """forward 的后一级 logits 必须依赖前一级目标 token。"""
    # 这个测试验证模型确实是自回归的：
    # 第 2 级 token 的预测会受到第 1 级真实 token 的影响。
    torch.manual_seed(0)
    model = SemanticIDTransformer([3, 3], 4, 8, 2, 1).eval()
    history = torch.ones((1, 4, 2), dtype=torch.long)
    left = torch.tensor([[0, 1]])
    right = torch.tensor([[2, 1]])
    with torch.no_grad():
        left_logits = model(history, left)
        right_logits = model(history, right)
    # 第 1 级预测只依赖用户历史和 start token，所以两次应该相同。
    assert torch.allclose(left_logits[0], right_logits[0])
    # 第 2 级预测会接收第 1 级 token embedding，因此两次应该不同。
    assert not torch.allclose(left_logits[1], right_logits[1])


def test_last_only_evaluation_has_one_sample_per_sequence():
    """NextItemDataset(last_only=True) 应为每个用户只保留最后一题。"""
    # last_only=True 时，每个用户序列只产生最后一个 next-item 评估样本。
    codes = np.asarray([[0], [1], [2], [3]])
    dataset = NextItemDataset([[0, 1, 2], [1, 2, 3]], codes, 3, last_only=True)
    assert len(dataset) == 2
    # 两条序列的最后一个物品分别是 2 和 3。
    assert [int(dataset[index][2]) for index in range(2)] == [2, 3]


def test_temporal_split_uses_future_only_for_validation_and_test():
    """每个用户的倒数第二、最后一个行为应分别成为验证、测试目标。"""
    train, validation, test = temporal_leave_two_out([[0, 1, 2, 3, 4]])
    assert train == [[0, 1, 2]]
    assert validation == [[0, 1, 2, 3]]
    assert test == [[0, 1, 2, 3, 4]]


def test_exact_and_full_beam_agree_on_topk_metrics():
    """Beam 足够宽、覆盖所有合法叶子时，应与精确目录打分一致。"""
    torch.manual_seed(3)
    codes = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    dataset = NextItemDataset([[0, 1, 2, 3]], codes, 3, last_only=True)
    model = SemanticIDTransformer([2, 2], 3, 8, 2, 1).eval()
    exact = evaluate_model_exact(
        model, dataset, codes, [1, 4], batch_size=1, catalog_chunk_size=2
    )
    beam = evaluate_model_beam(
        model, dataset, codes, [1, 4], batch_size=1, beam_size=4
    )
    assert {key: value for key, value in exact.items() if key not in {"auc", "uauc"}} == beam
    assert 0.0 <= exact["auc"] <= 1.0
    assert 0.0 <= exact["uauc"] <= 1.0


def test_training_checkpoint_can_resume_at_next_epoch(tmp_path):
    """长训练中断后应恢复模型、优化器和历史，而不是从第 1 轮重跑。"""
    torch.manual_seed(12)
    codes = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    dataset = NextItemDataset([[0, 1, 2, 3]], codes, 3)
    checkpoint = tmp_path / "training.pt"
    first = SemanticIDTransformer([2, 2], 3, 8, 2, 1)
    train_model(
        first,
        dataset,
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        checkpoint_path=checkpoint,
    )
    resumed = SemanticIDTransformer([2, 2], 3, 8, 2, 1)
    train_model(
        resumed,
        dataset,
        epochs=2,
        batch_size=2,
        learning_rate=0.001,
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert [row["epoch"] for row in resumed.training_history] == [1, 2]
