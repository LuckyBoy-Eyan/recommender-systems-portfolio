"""独立训练和评估全目录 SASRec baseline。"""

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

from src.data.dataset import SASRecDataset
from src.data.load import load_interactions
from src.data.split import temporal_leave_two_out
from src.models.sasrec import SASRec
from src.training.evaluate import evaluate_sasrec
from src.training.sasrec import train_sasrec_baseline


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(_resolve(args.config).read_text())
    output = _resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.get("torch_num_threads"):
        torch.set_num_threads(int(config["torch_num_threads"]))

    features, sequences = load_interactions(
        _resolve(config["interactions_path"]),
        _resolve(config["item_features_path"]),
        min_sequence_length=config.get("min_sequence_length", 3),
        max_sequence_length=config.get("max_sequence_length"),
    )
    train_sequences, validation_sequences, test_sequences = temporal_leave_two_out(
        sequences, config.get("min_train_sequence_length", 3)
    )
    common = {"max_history": config["max_history"]}
    train_data = SASRecDataset(
        train_sequences,
        **common,
        max_samples_per_sequence=config.get("max_train_samples_per_user"),
    )
    validation_data = SASRecDataset(
        validation_sequences, **common, last_only=True
    )
    test_data = SASRecDataset(test_sequences, **common, last_only=True)

    sasrec = config.get("sasrec", {})
    model = SASRec(
        len(features),
        config["max_history"],
        sasrec.get("hidden_dim", config["hidden_dim"]),
        sasrec.get("num_heads", config["num_heads"]),
        sasrec.get("num_layers", config["num_layers"]),
        dropout=sasrec.get("dropout", 0.1),
    )
    monitor = config.get("monitor_metric", "hitrate@20")
    model, history = train_sasrec_baseline(
        model,
        train_data,
        validation_data,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        ks=config["topk"],
        monitor=monitor,
        patience=config.get("early_stopping_patience"),
        device=config.get("device", "cpu"),
        weight_decay=config.get("weight_decay", 0.01),
        gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
        checkpoint_path=output / "sasrec_checkpoint.pt",
        exclude_seen=config.get("exclude_seen_items", False),
    )
    result = {
        "system": "standalone SASRec baseline",
        "role": "baseline only; not used for SID or reranking",
        "data": {
            "users": len(sequences),
            "items": len(features),
            "train_samples": len(train_data),
        },
        "validation": evaluate_sasrec(
            model,
            validation_data,
            config["topk"],
            batch_size=config["batch_size"],
            device=config.get("device", "cpu"),
            exclude_seen=config.get("exclude_seen_items", False),
        ),
        "test": evaluate_sasrec(
            model,
            test_data,
            config["topk"],
            batch_size=config["batch_size"],
            device=config.get("device", "cpu"),
            exclude_seen=config.get("exclude_seen_items", False),
        ),
        "training_history": history,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
