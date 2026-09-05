import pytest
import torch

from src.evaluation.metrics import single_positive_auc


def test_single_positive_auc_uses_only_eligible_items_and_half_ties():
    scores = torch.tensor(
        [
            [0.9, 0.5, 0.1, -torch.inf],
            [0.2, 0.2, 0.1, 0.0],
        ]
    )
    concordant, pairs, user_auc = single_positive_auc(
        scores, torch.tensor([0, 0])
    )
    assert pairs == 5
    assert concordant == pytest.approx(4.5)
    assert user_auc == pytest.approx([1.0, 5.0 / 6.0])
