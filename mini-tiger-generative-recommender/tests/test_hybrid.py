import numpy as np
import torch

from src.data.dataset import NextItemDataset, SASRecDataset
from src.models.generative import SemanticIDTransformer
from src.models.sasrec import SASRec
from src.training.hybrid import evaluate_generated_candidates_with_sasrec


def test_generated_full_catalog_candidates_can_be_reranked():
    """候选覆盖全目录时，候选 Recall 应为 1，融合结果应具有合法指标。"""
    torch.manual_seed(17)
    codes = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    sequences = [[0, 1, 2, 3], [1, 0, 3, 2]]
    semantic_data = NextItemDataset(
        sequences, codes, max_history=3, last_only=True
    )
    sasrec_data = SASRecDataset(sequences, max_history=3, last_only=True)
    semantic = SemanticIDTransformer([2, 2], 3, 8, 2, 1).eval()
    sasrec = SASRec(4, 3, 8, 2, 1, dropout=0.0).eval()

    result = evaluate_generated_candidates_with_sasrec(
        semantic,
        sasrec,
        semantic_data,
        sasrec_data,
        codes,
        candidate_k=4,
        beam_size=4,
        candidate_ks=(2, 4),
        final_ks=(1, 2),
        fusion_alphas=(0.0, 0.5, 1.0),
        batch_size=2,
        exclude_seen=False,
    )
    assert result["candidate_metrics"]["recall@4"] == 1.0
    assert result["average_generated_candidates"] == 4.0
    assert result["full_catalog_sasrec_metrics"]["recall@2"] >= 0.0
    assert set(result["rerank_metrics"]) == {
        "alpha=0",
        "alpha=0.5",
        "alpha=1",
    }


def test_hybrid_rejects_different_sample_orders():
    """两种模型目标不一致时必须失败，避免悄悄做错误对比。"""
    codes = np.asarray([[0], [1], [2]])
    semantic_data = NextItemDataset([[0, 1, 2]], codes, 2, last_only=True)
    sasrec_data = SASRecDataset([[0, 2, 1]], 2, last_only=True)
    semantic = SemanticIDTransformer([3], 2, 4, 2, 1).eval()
    sasrec = SASRec(3, 2, 4, 2, 1, dropout=0.0).eval()
    try:
        evaluate_generated_candidates_with_sasrec(
            semantic,
            sasrec,
            semantic_data,
            sasrec_data,
            codes,
            candidate_k=3,
            beam_size=3,
            candidate_ks=(3,),
            final_ks=(1,),
        )
    except ValueError as error:
        assert "target order" in str(error)
    else:
        raise AssertionError("mismatched datasets should fail")
