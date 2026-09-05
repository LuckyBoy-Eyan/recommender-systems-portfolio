import numpy as np
import torch

from src.data.feedback import FeedbackSequence, positive_target_leave_two_out
from src.data.feedback_dataset import (
    MaskedSASRecDataset,
    PositiveTargetDataset,
    SelectedPositiveTargetSASRecDataset,
)
from src.models.generative import SemanticIDTransformer
from src.models.sasrec import SASRec


def test_split_uses_last_two_positive_events_and_keeps_negative_context():
    sequence = FeedbackSequence(
        items=(0, 1, 2, 3, 4, 5), positive=(1, 0, 0, 1, 0, 1)
    )
    train, validation, test = positive_target_leave_two_out(
        [sequence], min_train_positive=1
    )
    assert train[0].items == (0, 1, 2)
    assert validation[0].items == (0, 1, 2, 3)
    assert test[0].items == sequence.items


def test_positive_target_dataset_keeps_negative_feedback_in_history():
    sequence = FeedbackSequence(
        items=(0, 1, 2, 3), positive=(1, 0, 0, 1)
    )
    dataset = PositiveTargetDataset(
        [sequence], np.array([[0], [1], [2], [3]]), 4, last_only=True
    )
    _, feedback, target_code, target, _ = dataset[0]
    assert feedback.tolist() == [0, 2, 1, 1]
    assert target.item() == 3
    assert target_code.tolist() == [3]


def test_masked_sasrec_labels_only_positive_targets():
    sequence = FeedbackSequence(
        items=(0, 1, 2, 3), positive=(1, 0, 1, 0)
    )
    dataset = MaskedSASRecDataset([sequence], 4)
    _, feedback, labels = dataset[0]
    assert feedback.tolist() == [0, 2, 1, 2]
    assert labels.tolist() == [-100, -100, 2, -100]


def test_selected_sasrec_targets_match_generator_cap_and_keep_full_context():
    sequence = FeedbackSequence(
        items=(0, 1, 2, 3, 4, 5), positive=(1, 0, 1, 0, 1, 1)
    )
    dataset = SelectedPositiveTargetSASRecDataset(
        [sequence], 4, max_targets_per_sequence=2
    )
    assert len(dataset) == 2
    items, feedback, labels = dataset[0]
    assert items.tolist() == [1, 2, 3, 4]
    assert feedback.tolist() == [2, 1, 2, 1]
    assert labels.tolist() == [-100, -100, -100, 4]


def test_feedback_embedding_changes_both_models():
    torch.manual_seed(4)
    sasrec = SASRec(4, 3, 8, 2, 1).eval()
    items = torch.tensor([[1, 2, 3]])
    positive = torch.tensor([[2, 2, 2]])
    mixed = torch.tensor([[2, 1, 2]])
    assert not torch.allclose(sasrec(items, positive), sasrec(items, mixed))

    generator = SemanticIDTransformer([4], 3, 8, 2, 1).eval()
    codes = items[:, :, None]
    positive_memory = generator.encode_history(codes, positive).memory
    mixed_memory = generator.encode_history(codes, mixed).memory
    assert not torch.allclose(positive_memory, mixed_memory)
