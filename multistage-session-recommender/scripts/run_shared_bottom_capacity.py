"""固定候选与训练协议，比较不同Shared-Bottom隐藏层容量。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import load_events
from src.data.split import (
    drop_ambiguous_target_sessions,
    leave_last_event_out,
    split_sessions,
)
from src.evaluation.metrics import WEIGHTS, evaluate_rankings
from src.features.point_in_time import build_point_in_time_dataset
from src.ranking.model import (
    attach_labels,
    sample_hard_negatives,
    score_ranker_system,
    train_ranker_system,
)


CAPACITIES = ((32, 16), (64, 32), (128, 64), (256, 128))


def evaluate_actions(system, features, labels, topk: int) -> dict[str, float]:
    metrics = {}
    for action in WEIGHTS:
        scored = score_ranker_system(system, features, action)
        action_labels = labels[labels["target_type"] == action]
        result = evaluate_rankings(scored, action_labels, topk)
        metrics[f"recall_{action}@{topk}"] = result[
            f"recall_{action}@{topk}"
        ]
    metrics[f"weighted_recall@{topk}"] = sum(
        WEIGHTS[action] * metrics[f"recall_{action}@{topk}"]
        for action in WEIGHTS
    )
    return metrics


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="Shared-Bottom容量验证实验")
    parser.add_argument("--config", default="configs/retailrocket.json")
    parser.add_argument(
        "--output",
        default="outputs/shared_bottom_capacity_validation.json",
    )
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    started = time.perf_counter()

    events = drop_ambiguous_target_sessions(
        load_events(ROOT / config["data_path"], config.get("max_sessions"))
    )
    train_events, valid_events, _ = split_sessions(
        events, config["train_ratio"], config["valid_ratio"]
    )
    train_history, train_labels = leave_last_event_out(train_events)
    valid_history, valid_labels = leave_last_event_out(valid_events)

    train_features, _, train_audit = build_point_in_time_dataset(
        train_events,
        train_history,
        train_labels,
        config["snapshot_interval"],
        config["candidates_per_source"],
        config["seed"],
        config.get("embedding"),
        progress_prefix="capacity_train",
    )
    labeled = sample_hard_negatives(
        attach_labels(train_features, train_labels),
        config["max_negative_per_session"],
        config["seed"],
    )

    validation_sessions = set(train_events["session"]) | set(
        valid_events["session"]
    )
    reference_pool = events[events["session"].isin(validation_sessions)]
    valid_features, _, valid_audit = build_point_in_time_dataset(
        reference_pool,
        valid_history,
        valid_labels,
        config["snapshot_interval"],
        config["candidates_per_source"],
        config["seed"],
        config.get("embedding"),
        progress_prefix="capacity_validation",
    )

    base_ranker = config["rankers"]["shared_bottom"]
    results = {}
    topk = int(config["topk"])
    for hidden_dims in CAPACITIES:
        name = f"{hidden_dims[0]}x{hidden_dims[1]}"
        ranker_config = dict(base_ranker)
        ranker_config["hidden_dims"] = list(hidden_dims)
        model_started = time.perf_counter()
        system = train_ranker_system(labeled, config["seed"], ranker_config)
        training_seconds = time.perf_counter() - model_started
        metrics = evaluate_actions(
            system, valid_features, valid_labels, topk
        )
        parameters = sum(
            parameter.numel()
            for parameter in system["bundle"].model.parameters()
            if parameter.requires_grad
        )
        results[name] = {
            "hidden_dims": list(hidden_dims),
            "parameters": int(parameters),
            "training_seconds": training_seconds,
            "metrics": metrics,
        }
        print(
            f"[capacity] hidden={name} parameters={parameters} "
            f"weighted_recall@{topk}={metrics[f'weighted_recall@{topk}']:.6f} "
            f"training_seconds={training_seconds:.2f}",
            flush=True,
        )

    weighted_key = f"weighted_recall@{topk}"
    best = max(results, key=lambda name: results[name]["metrics"][weighted_key])
    output = {
        "experiment": "shared_bottom_capacity",
        "split": "validation",
        "fixed_protocol": {
            "seed": config["seed"],
            "epochs": base_ranker["epochs"],
            "batch_size": base_ranker["batch_size"],
            "learning_rate": base_ranker["learning_rate"],
            "weight_decay": base_ranker["weight_decay"],
            "train_rows": int(len(labeled)),
            "validation_rows": int(len(valid_features)),
            "train_snapshots": int(len(train_audit)),
            "validation_snapshots": int(len(valid_audit)),
        },
        "results": results,
        "best_by_validation_weighted_recall": best,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(ROOT / args.output, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
