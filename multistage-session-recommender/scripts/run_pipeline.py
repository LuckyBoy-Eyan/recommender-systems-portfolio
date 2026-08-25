"""运行多阶段会话推荐的端到端训练与离线评估。

默认只评估验证集；方案确定后显式添加 ``--evaluate-test`` 才输出测试集结果。
"""

from __future__ import annotations

import json
import os
import sys
import argparse
import hashlib
import platform
import time
from pathlib import Path

# 限制 sklearn/loky 的并行资源探测，保证实验在不同机器上稳定运行。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import load_events
from src.data.split import drop_ambiguous_target_sessions, leave_last_event_out, split_sessions
from src.evaluation.metrics import (
    candidate_recall,
    candidate_recall_by_action,
    evaluate_rankings,
    source_recall,
)
from src.features.point_in_time import build_point_in_time_dataset
from src.ranking.model import (
    attach_labels,
    heuristic_score,
    ranker_system_audit,
    sample_hard_negatives,
    score_ranker_system,
    train_ranker_system,
)


def write_json_atomic(path: Path, value: dict) -> None:
    """原子写入JSON，避免长实验中断留下半个结果文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    """流式计算文件指纹，记录实验实际使用的数据版本。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configured_rankers(config: dict) -> tuple[dict[str, dict], str]:
    """解析唯一的正式Shared-Bottom排序器配置。"""
    if config.get("rankers"):
        rankers = {
            str(name): dict(ranker_config)
            for name, ranker_config in config["rankers"].items()
        }
    else:
        ranker_config = dict(config.get("ranker", {"method": "shared_bottom"}))
        rankers = {ranker_config.get("method", "shared_bottom"): ranker_config}
    if set(rankers) != {"shared_bottom"}:
        raise ValueError("正式流水线只允许配置一个 shared_bottom 排序器")
    if rankers["shared_bottom"].get("method") != "shared_bottom":
        raise ValueError("shared_bottom 配置的 method 必须为 shared_bottom")
    primary = config.get("primary_ranker", "shared_bottom")
    if primary != "shared_bottom":
        raise ValueError("primary_ranker 必须为 shared_bottom")
    return rankers, primary


def validate_data_profile(data_path: Path, events, expected: dict | None) -> dict | None:
    """核对预处理元数据，防止配置静默读取不匹配的数据。

    当前主实验固定使用 Top3000 商品和 20,000 个合格 Session。
    如果只修改配置却忘记重新生成 CSV，模型仍能运行，但所有实验口径都会错误。
    因此正式配置通过同名 metadata 文件校验 Session 数、目录上限和清洗参数。
    """
    if not expected:
        return None
    metadata_path = data_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"缺少数据元数据文件 {metadata_path}，无法验证实验口径"
        )
    metadata = json.loads(metadata_path.read_text())
    mismatches = {}
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": actual_value}
    actual_sessions = int(events["session"].nunique())
    if actual_sessions != expected.get("sessions", actual_sessions):
        mismatches["loaded_sessions"] = {
            "expected": expected.get("sessions"),
            "actual": actual_sessions,
        }
    if mismatches:
        raise ValueError(f"数据口径与配置不一致: {mismatches}")
    return metadata


def select_formal_and_warmup_events(
    events: pd.DataFrame,
    label_start_ts: int | None,
    warmup_days: int,
    day_interval: int,
    max_formal_sessions: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """固定正式标签Session，并选择其前连续若干天作为无监督Warm-up历史。

    所有实验共享 ``target_ts >= label_start_ts`` 的正式Session。Warm-up只改变
    Point-in-Time全局参考池，不产生标签，从而保证0/3/7/14天实验的训练、验证、
    测试标签完全一致。
    """
    if label_start_ts is None:
        if warmup_days:
            raise ValueError("设置 warmup_days 时必须同时设置 label_start_ts")
        empty = events.iloc[0:0].copy()
        return events.copy(), empty, {
            "enabled": False,
            "label_start_ts": None,
            "warmup_days": 0,
            "warmup_start_ts": None,
            "warmup_sessions": 0,
            "warmup_events": 0,
        }
    if warmup_days < 0:
        raise ValueError("warmup_days 不能为负数")
    if day_interval <= 0:
        raise ValueError("day_interval 必须为正数")
    label_start_ts = int(label_start_ts)
    warmup_start_ts = label_start_ts - int(warmup_days) * int(day_interval)
    session_times = events.groupby("session")["ts"].agg(["min", "max"]).reset_index()
    formal_candidates = session_times[
        session_times["max"] >= label_start_ts
    ].sort_values(["max", "session"], kind="mergesort")
    if max_formal_sessions is not None:
        if max_formal_sessions < 1:
            raise ValueError("max_formal_sessions 必须为正数")
        formal_candidates = formal_candidates.head(int(max_formal_sessions))
    formal_sessions = set(formal_candidates["session"])
    # Warm-up只接收完整落在窗口内的Session，避免3/7/14天组读到窗口起点之前的事件。
    warmup_sessions = set(
        session_times[
            (session_times["min"] >= warmup_start_ts)
            & (session_times["max"] < label_start_ts)
        ]["session"]
    )
    if not formal_sessions:
        raise ValueError("label_start_ts 之后没有可用正式Session")
    if formal_sessions & warmup_sessions:
        raise AssertionError("正式标签Session与Warm-up Session发生重叠")
    formal = events[events["session"].isin(formal_sessions)].copy()
    warmup = events[events["session"].isin(warmup_sessions)].copy()
    if not warmup.empty and int(warmup["ts"].max()) >= label_start_ts:
        raise AssertionError("Warm-up事件触及正式标签起点")
    return formal.reset_index(drop=True), warmup.reset_index(drop=True), {
        "enabled": True,
        "label_start_ts": label_start_ts,
        "warmup_days": int(warmup_days),
        "warmup_start_ts": warmup_start_ts,
        "max_formal_sessions": max_formal_sessions,
        "formal_sessions": int(formal["session"].nunique()),
        "formal_events": len(formal),
        "warmup_sessions": int(warmup["session"].nunique()),
        "warmup_events": len(warmup),
    }


def main():
    """加载配置，完成数据切分、召回、训练、评估并写出实验产物。

    核心调用链：
        ``load_events``
        -> ``split_sessions`` / ``leave_last_event_out``
        -> ``build_point_in_time_dataset``
        -> ``attach_labels`` / ``sample_hard_negatives``
        -> ``train_ranker_system``
        -> 内部 ``evaluate_split``
        -> ``candidate_recall`` / ``evaluate_rankings`` / ``source_recall``。

    命令行参数由 ``argparse`` 读取，因此该函数没有 Python 形参；运行结果写入
    ``--output`` 指定目录，并把完整指标 JSON 打印到标准输出，供消融脚本读取。
    """
    parser = argparse.ArgumentParser(description="运行多阶段会话推荐实验")
    parser.add_argument(
        "--config",
        default="configs/retailrocket.json",
        help="相对于项目根目录的真实数据实验 JSON 配置路径",
    )
    parser.add_argument(
        "--output",
        default="outputs/retailrocket_top3000_session20000",
        help="相对于项目根目录的实验输出目录",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="方案在验证集上确定后，额外评估一次独立测试集",
    )
    parser.add_argument(
        "--warmup-days",
        type=int,
        default=None,
        help="可选：覆盖配置中的Warm-up天数，用于0/3/7/14天公平消融",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    if args.warmup_days is not None:
        config["warmup_days"] = int(args.warmup_days)
    # 项目只运行真实数据实验；缺少 data_path 时立即报错，避免误用虚构数据。
    if not config.get("data_path"):
        raise ValueError("配置文件必须提供真实数据路径 data_path")
    data_path = ROOT / config["data_path"]
    events = load_events(data_path, config.get("max_sessions"))
    data_metadata = validate_data_profile(
        data_path, events, config.get("expected_data_profile")
    )
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    experiment_started = time.perf_counter()
    ranker_configs, primary_ranker = configured_rankers(config)
    manifest = {
        "status": "running",
        "config_path": str(
            config_path.relative_to(ROOT)
            if config_path.is_relative_to(ROOT)
            else config_path
        ),
        "config": config,
        "data_path": str(data_path.relative_to(ROOT)),
        "data_sha256": file_sha256(data_path),
        "data_metadata": data_metadata,
        "evaluate_test": bool(args.evaluate_test),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    # 即使预处理已清洗过，也在入口处防御性删除歧义目标 Session。
    input_events = len(events)
    input_sessions = int(events["session"].nunique())
    events = drop_ambiguous_target_sessions(events)
    dropped_ambiguous_sessions = input_sessions - int(events["session"].nunique())
    events, warmup_events, warmup_profile = select_formal_and_warmup_events(
        events,
        config.get("label_start_ts"),
        int(config.get("warmup_days", 0)),
        int(config["snapshot_interval"]),
        config.get("max_formal_sessions"),
    )
    manifest["warmup"] = warmup_profile
    # 先按目标时间切分完整 Session，再分别执行 Leave-Last-Event-Out。
    train_events, valid_events, test_events = split_sessions(
        events, config["train_ratio"], config["valid_ratio"]
    )
    train_history, train_labels = leave_last_event_out(train_events)
    valid_history, valid_labels = leave_last_event_out(valid_events)
    test_history, test_labels = leave_last_event_out(test_events)

    # 每个训练样本使用严格早于目标时间的因果快照，防止较晚事件进入较早样本的
    # 召回图、Item2Vec 或商品统计。
    train_candidate_started = time.perf_counter()
    train_reference_pool = pd.concat(
        [warmup_events, train_events], ignore_index=True
    )
    train_features, train_recalled, train_audit = build_point_in_time_dataset(
        train_reference_pool,
        train_history,
        train_labels,
        config["snapshot_interval"],
        config["candidates_per_source"],
        config["seed"],
        config.get("embedding"),
        progress_prefix="train",
    )
    train_candidate_seconds = time.perf_counter() - train_candidate_started
    labeled = attach_labels(train_features, train_labels)
    labeled = sample_hard_negatives(labeled, config["max_negative_per_session"], config["seed"])
    ranker_systems = {}
    ranker_training_seconds = {}
    for name, ranker_config in ranker_configs.items():
        print(f"[ranker] training name={name} method={ranker_config.get('method')}", flush=True)
        ranker_started = time.perf_counter()
        ranker_systems[name] = train_ranker_system(
            labeled, config["seed"], ranker_config
        )
        ranker_training_seconds[name] = time.perf_counter() - ranker_started
        print(
            f"[ranker] completed name={name} "
            f"seconds={ranker_training_seconds[name]:.2f}",
            flush=True,
        )

    def evaluate_split(reference_pool, history, labels, split_name: str):
        """评估一个时间分区。

        参数：
            reference_pool:
                该分区在滚动回放时允许访问的全局事件池。
            history:
                该分区每个 Session 的本地可见历史。
            labels:
                该分区的真实目标表。

        返回：
            ``(split_metrics, recalled, audit)``，分别是指标字典、召回长表和
            Point-in-Time 审计表。
        """
        features, recalled, audit = build_point_in_time_dataset(
            reference_pool,
            history,
            labels,
            config["snapshot_interval"],
            config["candidates_per_source"],
            config["seed"],
            config.get("embedding"),
            progress_prefix=split_name,
        )
        baseline = heuristic_score(features)
        # Shared-Bottom面对相同候选，分别输出点击、加购、购买三份排序结果。
        ranker_metrics = {}
        for ranker_name, ranker_system in ranker_systems.items():
            action_metrics = {}
            for action in ("clicks", "carts", "orders"):
                action_ranked = score_ranker_system(
                    ranker_system, features, action
                )
                action_labels = labels[labels["target_type"] == action]
                evaluated = evaluate_rankings(
                    action_ranked, action_labels, config["topk"]
                )
                action_metrics[f"recall_{action}@{config['topk']}"] = evaluated[
                    f"recall_{action}@{config['topk']}"
                ]
            # Weighted Recall 使用固定任务权重：点击 0.1、加购 0.3、购买 0.6。
            action_metrics[f"weighted_recall@{config['topk']}"] = sum(
                weight * action_metrics[f"recall_{action}@{config['topk']}"]
                for action, weight in {
                    "clicks": 0.10,
                    "carts": 0.30,
                    "orders": 0.60,
                }.items()
            )
            ranker_metrics[ranker_name] = action_metrics
        split_metrics = {
            "candidate_recall": candidate_recall(recalled, labels),
            "candidate_recall_by_action": candidate_recall_by_action(
                recalled, labels
            ),
            "baseline": evaluate_rankings(baseline, labels, config["topk"]),
            "primary_ranker": primary_ranker,
            "ranker": ranker_metrics[primary_ranker],
            "rankers": ranker_metrics,
            "sources": source_recall(recalled, labels, config["topk"]),
            "point_in_time_audit": {
                "snapshots": len(audit),
                "all_snapshots_causal": bool(
                    audit["max_reference_ts"].isna().all()
                    or (
                        audit["max_reference_ts"].dropna()
                        < audit.loc[audit["max_reference_ts"].notna(), "min_target_ts"]
                    ).all()
                ),
            },
        }
        return split_metrics, recalled, audit

    # 验证滚动回放不能访问测试 Session；只开放训练和验证阶段的事件。
    validation_pool_sessions = set(train_events["session"]) | set(valid_events["session"])
    validation_reference_pool = pd.concat(
        [
            warmup_events,
            events[events["session"].isin(validation_pool_sessions)],
        ],
        ignore_index=True,
    )
    validation_started = time.perf_counter()
    validation_metrics, valid_recalled, valid_audit = evaluate_split(
        validation_reference_pool, valid_history, valid_labels, "validation"
    )
    validation_seconds = time.perf_counter() - validation_started
    metrics = {
        "data": {
            "events": len(events),
            "input_events": input_events,
            "dropped_ambiguous_target_sessions": dropped_ambiguous_sessions,
            "formal_sessions": int(events["session"].nunique()),
            "warmup_sessions": int(warmup_events["session"].nunique()),
            "warmup_events": len(warmup_events),
            "train_sessions": int(train_events["session"].nunique()),
            "valid_sessions": int(valid_events["session"].nunique()),
            "test_sessions": int(test_events["session"].nunique()),
        },
        "validation": validation_metrics,
        "point_in_time": {
            "snapshot_interval": config["snapshot_interval"],
            "embedding": config.get("embedding", {"method": "item2vec"}),
            "train_snapshots": len(train_audit),
            "validation_snapshots": len(valid_audit),
        },
        "primary_ranker": primary_ranker,
        "rankers": ranker_configs,
        "ranker_training_audit": {
            name: ranker_system_audit(system)
            for name, system in ranker_systems.items()
        },
        "runtime_seconds": {
            "train_candidate_and_features": train_candidate_seconds,
            "rankers": ranker_training_seconds,
            "validation": validation_seconds,
        },
    }
    test_recalled = None
    test_audit = None
    if args.evaluate_test:
        # 测试只在显式开关开启时执行，避免调参过程中反复查看测试指标。
        test_started = time.perf_counter()
        test_reference_pool = pd.concat(
            [warmup_events, events], ignore_index=True
        )
        test_metrics, test_recalled, test_audit = evaluate_split(
            test_reference_pool, test_history, test_labels, "test"
        )
        metrics["runtime_seconds"]["test"] = time.perf_counter() - test_started
        metrics["test"] = test_metrics
        metrics["point_in_time"]["test_snapshots"] = len(test_audit)
    # 输出指标、因果审计、候选明细和一份最多 1000 行的训练样本，便于人工检查。
    metrics["runtime_seconds"]["total"] = time.perf_counter() - experiment_started
    write_json_atomic(output / "metrics.json", metrics)
    temporary_model = output / "ranker_systems.joblib.tmp"
    joblib.dump(ranker_systems, temporary_model)
    os.replace(temporary_model, output / "ranker_systems.joblib")
    train_audit.to_csv(output / "train_point_in_time_audit.csv", index=False)
    valid_audit.to_csv(output / "validation_point_in_time_audit.csv", index=False)
    valid_recalled.to_csv(output / "validation_recall_candidates.csv", index=False)
    if test_recalled is not None:
        test_recalled.to_csv(output / "test_recall_candidates.csv", index=False)
        test_audit.to_csv(output / "test_point_in_time_audit.csv", index=False)
    labeled.sample(min(1000, len(labeled)), random_state=config["seed"]).to_csv(
        output / "training_sample.csv", index=False
    )
    manifest["status"] = "completed"
    manifest["metrics_path"] = "metrics.json"
    manifest["model_path"] = "ranker_systems.joblib"
    manifest["runtime_seconds"] = metrics["runtime_seconds"]
    write_json_atomic(output / "manifest.json", manifest)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
