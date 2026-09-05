"""完整正负上下文、仅正目标监督的 MiniTIGER V2 主线。"""

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

from scripts.run_demo import make_validation_callback, train_from_config
from src.data.feedback import load_feedback_sequences, positive_target_leave_two_out
from src.data.feedback_dataset import PositiveTargetDataset
from src.indexing.semantic_ids import codebook_diagnostics, collision_rate
from src.models.generative import SemanticIDTransformer
from src.training.evaluate import evaluate_model


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--semantic-codes",
        help="可选：覆盖配置中的 Semantic ID 文件，用于公平 A/B",
    )
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
    codes_path = args.semantic_codes or config["semantic_codes_path"]
    codes = np.load(_resolve(codes_path)).astype(np.int64)
    if len(codes) != len(features) or collision_rate(codes) != 0:
        raise ValueError("Semantic codes must uniquely cover the complete catalog")
    train, validation, test = positive_target_leave_two_out(
        sequences,
        min_train_positive=config.get("min_train_positive", 3),
    )
    train_data = PositiveTargetDataset(
        train,
        codes,
        config["max_history"],
        max_samples_per_sequence=config.get("max_train_samples_per_user"),
    )
    validation_data = PositiveTargetDataset(
        validation, codes, config["max_history"], last_only=True
    )
    test_data = PositiveTargetDataset(
        test, codes, config["max_history"], last_only=True
    )
    sizes = [int(codes[:, level].max()) + 1 for level in range(codes.shape[1])]
    model = SemanticIDTransformer(
        sizes,
        config["max_history"],
        config["hidden_dim"],
        config["num_heads"],
        config["num_layers"],
        config.get("feedforward_dim"),
        decoder_layers=config.get("decoder_layers"),
        dropout=config.get("dropout", 0.1),
    )
    # V2 每个样本都是正目标；负事件仅作为带类型标记的上下文。
    validation_config = dict(config)
    validation_config["exclude_seen_items"] = False
    train_from_config(
        model,
        train_data,
        validation_config,
        make_validation_callback(validation_data, codes, validation_config),
        output / "semantic_checkpoint.pt",
    )
    kwargs = {
        "batch_size": config.get("evaluation_batch_size", 64),
        "device": config.get("device", "cpu"),
        "exclude_seen": False,
    }
    mode = config.get("evaluation_mode", "beam")
    if mode == "beam":
        kwargs["beam_size"] = config.get("beam_size", 200)
    else:
        kwargs["catalog_chunk_size"] = config.get("catalog_chunk_size", 512)
    result = {
        "system": "MiniTIGER V2 full-feedback positive-target generator",
        "protocol": {
            "full_event_context": True,
            "feedback_type_embedding": True,
            "positive_targets_only": True,
            "split_on_last_two_positive_events": True,
            "exclude_seen_items": False,
        },
        "data": {
            "users": len(test),
            "items": len(features),
            "train_samples": len(train_data),
        },
        "semantic_id": {
            "codebook_sizes": sizes,
            "collision_rate": collision_rate(codes),
            "diagnostics": codebook_diagnostics(codes, sizes),
        },
        "validation": evaluate_model(
            model, validation_data, codes, config["topk"], mode=mode, **kwargs
        ),
        "test": evaluate_model(
            model, test_data, codes, config["topk"], mode=mode, **kwargs
        ),
        "training_history": model.training_history,
    }
    if config.get("run_exact_evaluation", False):
        result["exact_test"] = evaluate_model(
            model,
            test_data,
            codes,
            config["topk"],
            mode="exact",
            batch_size=config.get("exact_evaluation_batch_size", 8),
            catalog_chunk_size=config.get("catalog_chunk_size", 512),
            device=config.get("device", "cpu"),
            exclude_seen=False,
        )
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
