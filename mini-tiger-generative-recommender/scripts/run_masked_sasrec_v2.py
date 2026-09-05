"""完整正负上下文、仅正目标监督的公平 SASRec V2 baseline。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.feedback import load_feedback_sequences, positive_target_leave_two_out
from src.data.feedback_dataset import (
    MaskedSASRecDataset,
    SASRecPositiveEvalDataset,
    SelectedPositiveTargetSASRecDataset,
)
from src.models.sasrec import SASRec
from src.training.evaluate import evaluate_sasrec
from src.training.sasrec import train_masked_sasrec


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(_resolve(args.config).read_text())
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    features, sequences = load_feedback_sequences(
        _resolve(config["interactions_path"]),
        _resolve(config["item_features_path"]),
        min_positive_events=config.get("min_positive_events", 5),
    )
    train, validation, test = positive_target_leave_two_out(
        sequences,
        min_train_positive=config.get("min_train_positive", 3),
    )
    max_targets = config.get("max_train_samples_per_user")
    if max_targets is None:
        train_data = MaskedSASRecDataset(
            train,
            config["max_history"],
            target_stride=config.get("target_stride"),
        )
        training_dataset = "dense_masked_windows"
    else:
        train_data = SelectedPositiveTargetSASRecDataset(
            train,
            config["max_history"],
            max_targets_per_sequence=int(max_targets),
        )
        training_dataset = "selected_positive_targets"
    validation_data = SASRecPositiveEvalDataset(
        validation, config["max_history"]
    )
    test_data = SASRecPositiveEvalDataset(test, config["max_history"])
    sasrec = config["sasrec"]
    model = SASRec(
        len(features),
        config["max_history"],
        sasrec["hidden_dim"],
        sasrec["num_heads"],
        sasrec["num_layers"],
        sasrec.get("dropout", 0.1),
    )
    model, history = train_masked_sasrec(
        model,
        train_data,
        validation_data,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        ks=config["topk"],
        monitor=config.get("monitor_metric", "hitrate@20"),
        patience=config.get("early_stopping_patience", 3),
        device=config.get("device", "cpu"),
        weight_decay=config.get("weight_decay", 0.01),
        gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
        checkpoint_path=output / "masked_sasrec_checkpoint.pt",
        mixed_precision=config.get("mixed_precision", False),
        resume=config.get("resume_training", True),
        warmup_epochs=config.get("warmup_epochs", 2),
        min_learning_rate=config.get("min_learning_rate", 1e-5),
    )
    evaluation = {
        "validation": evaluate_sasrec(
            model,
            validation_data,
            config["topk"],
            batch_size=config.get("evaluation_batch_size", config["batch_size"]),
            device=config.get("device", "cpu"),
            exclude_seen=False,
        ),
        "test": evaluate_sasrec(
            model,
            test_data,
            config["topk"],
            batch_size=config.get("evaluation_batch_size", config["batch_size"]),
            device=config.get("device", "cpu"),
            exclude_seen=False,
        ),
    }
    result = {
        "system": "SASRec V2 full-feedback masked-positive baseline",
        "protocol": {
            "full_event_context": True,
            "feedback_type_embedding": True,
            "negative_target_ignore_index": -100,
            "split_on_last_two_positive_events": True,
            "exclude_seen_items": False,
            "max_positive_targets_per_user": max_targets,
        },
        "data": {
            "users": len(test),
            "items": len(features),
            "training_dataset": training_dataset,
            "train_examples": len(train_data),
        },
        **evaluation,
        "training_history": history,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
