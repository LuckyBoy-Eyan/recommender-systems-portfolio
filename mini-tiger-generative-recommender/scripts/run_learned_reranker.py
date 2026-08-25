"""生成候选缓存，并训练候选级学习式精排器。

数据协议：
1. 基础生成模型和 SASRec 只使用原训练时间窗训练；
2. 原验证用户固定拆成 70% stacking-train、30% calibration；
3. calibration 选择精排 checkpoint，测试集只做最终评估。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_hybrid import _datasets, _load_models, _resolve, _subset_pair
from src.data.load import load_interactions
from src.data.split import temporal_leave_two_out
from src.models.candidate_ranker import CandidateRanker
from src.training.learned_reranker import (
    FEATURE_NAMES,
    build_candidate_cache,
    diagnose_candidate_cache,
    evaluate_candidate_ranker,
    item_popularity,
    load_candidate_cache,
    save_candidate_cache,
    train_candidate_ranker,
)


def _atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_or_build_cache(
    path,
    semantic,
    sasrec,
    pair,
    codes,
    popularity,
    args,
    split_name,
):
    if path.exists() and not args.rebuild_cache:
        cache = load_candidate_cache(path)
        expected = {
            "candidate_k": args.candidate_k,
            "beam_size": args.beam_size,
            "users": len(pair[0]),
        }
        for key, value in expected.items():
            if cache.metadata.get(key) != value:
                raise ValueError(
                    f"{path} has {key}={cache.metadata.get(key)!r}, "
                    f"expected {value!r}; pass --rebuild-cache"
                )
        print(f"loaded cache={path}")
        return cache
    cache = build_candidate_cache(
        semantic,
        sasrec,
        *pair,
        codes,
        popularity,
        candidate_k=args.candidate_k,
        beam_size=args.beam_size,
        batch_size=args.generation_batch_size,
        device=args.device,
        exclude_seen=args.exclude_seen,
        split_name=split_name,
    )
    save_candidate_cache(cache, path)
    print(f"saved cache={path}")
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kuairec_big_cuda.json")
    parser.add_argument("--artifacts", default="outputs/kuairec_big_cuda")
    parser.add_argument(
        "--sasrec-artifacts",
        help="SASRec checkpoint 目录；省略时与 --artifacts 相同",
    )
    parser.add_argument("--work-dir", default="outputs/kuairec_big_cuda/reranker")
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--beam-size", type=int, default=500)
    parser.add_argument("--device", default=None)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--stacking-train-fraction", type=float, default=0.7)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="只生成验证/测试候选缓存，不训练精排器",
    )
    args = parser.parse_args()

    config = json.loads(_resolve(args.config).read_text())
    args.device = args.device or config.get("device", "cpu")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu on Mac")
    if not 0.0 < args.stacking_train_fraction < 1.0:
        raise SystemExit("--stacking-train-fraction must be between zero and one")
    if config.get("torch_num_threads"):
        torch.set_num_threads(int(config["torch_num_threads"]))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    artifacts = _resolve(args.artifacts)
    sasrec_artifacts = (
        _resolve(args.sasrec_artifacts) if args.sasrec_artifacts else artifacts
    )
    work_dir = _resolve(args.work_dir)
    features, sequences = load_interactions(
        _resolve(config["interactions_path"]),
        _resolve(config["item_features_path"]),
        min_sequence_length=config.get("min_sequence_length", 3),
        max_sequence_length=config.get("max_sequence_length"),
    )
    train_sequences, _, _ = temporal_leave_two_out(
        sequences, config.get("min_train_sequence_length", 3)
    )
    popularity = item_popularity(train_sequences, len(features))
    codes = np.load(artifacts / "semantic_codes.npy")
    datasets = _datasets(sequences, codes, config)
    validation_pair = _subset_pair(datasets["validation"], args.max_users, seed)
    test_pair = _subset_pair(datasets["test"], args.max_users, seed + 1)
    semantic, sasrec = _load_models(
        config, features, codes, artifacts, args.device, sasrec_artifacts
    )
    args.exclude_seen = config.get("exclude_seen_items", False)

    suffix = f"top{args.candidate_k}"
    validation_cache = _load_or_build_cache(
        work_dir / f"validation_{suffix}.npz",
        semantic,
        sasrec,
        validation_pair,
        codes,
        popularity,
        args,
        "validation",
    )
    test_cache = _load_or_build_cache(
        work_dir / f"test_{suffix}.npz",
        semantic,
        sasrec,
        test_pair,
        codes,
        popularity,
        args,
        "test",
    )
    diagnostics = {
        "validation": diagnose_candidate_cache(validation_cache),
        "test": diagnose_candidate_cache(test_cache),
    }
    (work_dir / f"diagnostics_{suffix}.json").write_text(
        json.dumps(diagnostics, indent=2)
    )
    if args.cache_only:
        print(json.dumps(diagnostics, indent=2))
        return

    indices = np.random.default_rng(seed).permutation(
        len(validation_cache.targets)
    )
    split = int(len(indices) * args.stacking_train_fraction)
    stacking_train = indices[:split]
    calibration = indices[split:]
    baseline_ranker = CandidateRanker(len(FEATURE_NAMES), base_alpha=0.25)
    baseline_metrics = {
        "calibration": evaluate_candidate_ranker(
            baseline_ranker,
            validation_cache,
            indices=calibration,
            ks=tuple(config["topk"]),
            batch_size=args.training_batch_size,
            device=args.device,
        ),
        "test": evaluate_candidate_ranker(
            baseline_ranker,
            test_cache,
            ks=tuple(config["topk"]),
            batch_size=args.training_batch_size,
            device=args.device,
        ),
    }
    ranker = CandidateRanker(len(FEATURE_NAMES), base_alpha=0.25)
    ranker, history = train_candidate_ranker(
        ranker,
        validation_cache,
        stacking_train,
        calibration,
        epochs=args.epochs,
        batch_size=args.training_batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        monitor="ndcg@20",
        guard_metric="recall@20",
        ks=tuple(config["topk"]),
        device=args.device,
    )
    calibration_metrics = evaluate_candidate_ranker(
        ranker,
        validation_cache,
        indices=calibration,
        ks=tuple(config["topk"]),
        batch_size=args.training_batch_size,
        device=args.device,
    )
    test_metrics = evaluate_candidate_ranker(
        ranker,
        test_cache,
        ks=tuple(config["topk"]),
        batch_size=args.training_batch_size,
        device=args.device,
    )
    checkpoint = {
        "state_dict": ranker.state_dict(),
        "feature_names": list(FEATURE_NAMES),
        "candidate_k": args.candidate_k,
        "beam_size": args.beam_size,
        "selection_metric": "ndcg@20",
        "selection_guard": "recall@20 must not fall below fixed fusion",
        "base_fusion_alpha": ranker.base_alpha,
        "residual_scale": ranker.residual_scale,
        "data_protocol": (
            "70% validation users for stacking train; "
            "30% validation users for calibration; test untouched"
        ),
    }
    _atomic_torch_save(checkpoint, work_dir / f"candidate_ranker_{suffix}.pt")
    result = {
        "system": "Learned Gen-Rerank",
        "candidate_k": args.candidate_k,
        "beam_size": args.beam_size,
        "feature_names": list(FEATURE_NAMES),
        "base_fusion_alpha": ranker.base_alpha,
        "safe_fallback": "epoch 0 exactly reproduces fixed fusion",
        "data_protocol": checkpoint["data_protocol"],
        "users": {
            "stacking_train": int(len(stacking_train)),
            "calibration": int(len(calibration)),
            "test": int(len(test_cache.targets)),
        },
        "diagnostics": diagnostics,
        "fixed_fusion_baseline": baseline_metrics,
        "training_history": history,
        "calibration": calibration_metrics,
        "test": test_metrics,
    }
    output = work_dir / f"learned_reranker_{suffix}_metrics.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
