"""在验证集上执行 Shared-Bottom 特征组屏蔽实验。

该实验复用已训练模型，只重建一次验证集 Point-in-Time 特征。屏蔽某组时，
把对应原始特征替换成“经过训练期预处理后标准化值为 0”的训练均值等价值，
然后比较 Weighted Recall@20 的变化。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np

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
from src.ranking.model import score_ranker_system


FEATURE_GROUPS = {
    "recall_confidence": [
        "source_rank_recent",
        "source_score_recent",
        "source_rank_popular",
        "source_score_popular",
        "source_rank_itemcf",
        "source_score_itemcf",
        "source_rank_item2vec",
        "source_score_item2vec",
        "source_count",
        "best_source_rank",
        "rrf_score",
    ],
    "item_popularity": [
        "item_events",
        "item_sessions",
        "item_last_ts",
    ],
    "item_action": [
        "item_clicks",
        "item_carts",
        "item_orders",
        "cart_rate",
        "order_rate",
    ],
    "session_context": [
        "session_length",
        "session_unique_items",
        "session_last_ts",
    ],
    "session_item_cross": [
        "in_session_count",
        "pair_last_ts",
        "seconds_since_seen",
    ],
}


def mask_to_training_mean(features, bundle, columns: list[str]):
    """把指定列替换为预处理器训练均值对应的原始空间数值。"""
    masked = features.copy()
    preprocessor = bundle.preprocessor
    column_to_index = {
        column: index for index, column in enumerate(preprocessor.columns)
    }
    missing = sorted(set(columns) - set(column_to_index))
    if missing:
        raise ValueError(f"模型预处理器缺少待屏蔽特征: {missing}")
    for column in columns:
        transformed_mean = float(
            preprocessor.scaler.mean_[column_to_index[column]]
        )
        raw_mean = (
            float(np.expm1(transformed_mean))
            if column in preprocessor.log_columns
            else transformed_mean
        )
        masked[column] = raw_mean
    return masked


def evaluate_actions(system, features, labels, topk: int) -> dict[str, float]:
    """使用指定候选特征计算三个任务及加权 Recall@K。"""
    metrics = {}
    for action in WEIGHTS:
        scored = score_ranker_system(system, features, action)
        action_labels = labels[labels["target_type"] == action]
        evaluated = evaluate_rankings(scored, action_labels, topk)
        metrics[f"recall_{action}@{topk}"] = evaluated[
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
    parser = argparse.ArgumentParser(description="Shared-Bottom验证集特征组屏蔽")
    parser.add_argument("--config", default="configs/retailrocket.json")
    parser.add_argument(
        "--model",
        default=(
            "outputs/retailrocket_top3000_session20000_final/"
            "ranker_systems.joblib"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/feature_masking_validation.json",
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
    valid_history, valid_labels = leave_last_event_out(valid_events)
    validation_sessions = set(train_events["session"]) | set(
        valid_events["session"]
    )
    reference_pool = events[events["session"].isin(validation_sessions)]
    features, _, audit = build_point_in_time_dataset(
        reference_pool,
        valid_history,
        valid_labels,
        config["snapshot_interval"],
        config["candidates_per_source"],
        config["seed"],
        config.get("embedding"),
        progress_prefix="feature_masking",
    )

    systems = joblib.load(ROOT / args.model)
    if "shared_bottom" not in systems:
        raise KeyError("模型产物中不存在 shared_bottom")
    system = systems["shared_bottom"]
    if system["method"] != "shared_bottom":
        raise ValueError("shared_bottom系统的方法字段不匹配")

    topk = int(config["topk"])
    baseline = evaluate_actions(system, features, valid_labels, topk)
    weighted_key = f"weighted_recall@{topk}"
    groups = {}
    for name, columns in FEATURE_GROUPS.items():
        group_started = time.perf_counter()
        masked = mask_to_training_mean(features, system["bundle"], columns)
        metrics = evaluate_actions(system, masked, valid_labels, topk)
        groups[name] = {
            "columns": columns,
            "metrics": metrics,
            "absolute_drop": {
                key: baseline[key] - metrics[key] for key in baseline
            },
            "weighted_relative_drop": (
                (baseline[weighted_key] - metrics[weighted_key])
                / baseline[weighted_key]
                if baseline[weighted_key]
                else 0.0
            ),
            "runtime_seconds": time.perf_counter() - group_started,
        }
        print(
            f"[mask] group={name} {weighted_key}={metrics[weighted_key]:.6f} "
            f"drop={groups[name]['absolute_drop'][weighted_key]:.6f}",
            flush=True,
        )

    result = {
        "experiment": "shared_bottom_feature_group_masking",
        "split": "validation",
        "method": (
            "replace each raw feature with the value mapping to zero after "
            "the fitted training preprocessor"
        ),
        "model_path": args.model,
        "sessions": int(valid_labels["session"].nunique()),
        "feature_rows": int(len(features)),
        "snapshots": int(len(audit)),
        "baseline": baseline,
        "groups": groups,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(ROOT / args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
