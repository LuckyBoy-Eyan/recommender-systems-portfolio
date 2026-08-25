"""评估工业 Gen-Rerank：Semantic ID 生成候选，SASRec 候选内精排。

示例：
    python scripts/run_hybrid.py \
      --config configs/kuairec_big_cuda.json \
      --artifacts outputs/kuairec_big_cuda \
      --output outputs/kuairec_big_cuda/hybrid_metrics.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import NextItemDataset, SASRecDataset
from src.data.load import load_interactions
from src.data.split import temporal_leave_two_out
from src.models.generative import SemanticIDTransformer
from src.models.sasrec import SASRec
from src.training.hybrid import evaluate_generated_candidates_with_sasrec


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _datasets(sequences, codes, config):
    """用完全相同的目标顺序构建生成模型与 SASRec 验证/测试集。"""
    _, validation_sequences, test_sequences = temporal_leave_two_out(
        sequences, config.get("min_train_sequence_length", 3)
    )
    common = {"max_history": config["max_history"], "last_only": True}
    return {
        "validation": (
            NextItemDataset(validation_sequences, codes, **common),
            SASRecDataset(validation_sequences, **common),
        ),
        "test": (
            NextItemDataset(test_sequences, codes, **common),
            SASRecDataset(test_sequences, **common),
        ),
    }


def _subset_pair(pair, max_users: int | None, seed: int):
    if max_users is None or max_users >= len(pair[0]):
        return pair
    indices = np.random.default_rng(seed).choice(
        len(pair[0]), size=max_users, replace=False
    )
    selected = indices.tolist()
    return Subset(pair[0], selected), Subset(pair[1], selected)


def _load_models(
    config,
    features,
    codes,
    artifacts: Path,
    device: str,
    sasrec_artifacts: Path | None = None,
):
    semantic_sizes = [int(codes[:, level].max()) + 1 for level in range(codes.shape[1])]
    semantic = SemanticIDTransformer(
        semantic_sizes,
        config["max_history"],
        config["hidden_dim"],
        config["num_heads"],
        config["num_layers"],
        config.get("feedforward_dim"),
    )
    semantic.load_state_dict(
        torch.load(
            artifacts / "semantic_model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )

    sas_config = config.get("sasrec", {})
    sasrec = SASRec(
        num_items=len(features),
        max_history=config["max_history"],
        hidden_dim=sas_config.get("hidden_dim", config["hidden_dim"]),
        num_heads=sas_config.get("num_heads", config["num_heads"]),
        num_layers=sas_config.get("num_layers", config["num_layers"]),
        dropout=sas_config.get("dropout", 0.1),
    )
    sasrec_source = sasrec_artifacts or artifacts
    checkpoint = torch.load(
        sasrec_source / "sasrec_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    sasrec.load_state_dict(checkpoint["best_state"])
    return semantic.to(device).eval(), sasrec.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kuairec_big_cuda.json")
    parser.add_argument("--artifacts", default="outputs/kuairec_big_cuda")
    parser.add_argument(
        "--sasrec-artifacts",
        help="SASRec checkpoint 目录；省略时与 --artifacts 相同",
    )
    parser.add_argument(
        "--output", default="outputs/kuairec_big_cuda/hybrid_metrics.json"
    )
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--beam-size", type=int, default=500)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-users", type=int)
    args = parser.parse_args()

    config = json.loads(_resolve(args.config).read_text())
    device = args.device or config.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu on Mac")
    if config.get("torch_num_threads"):
        torch.set_num_threads(int(config["torch_num_threads"]))
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    artifacts = _resolve(args.artifacts)
    sasrec_artifacts = (
        _resolve(args.sasrec_artifacts) if args.sasrec_artifacts else artifacts
    )
    features, sequences = load_interactions(
        _resolve(config["interactions_path"]),
        _resolve(config["item_features_path"]),
        min_sequence_length=config.get("min_sequence_length", 3),
        max_sequence_length=config.get("max_sequence_length"),
    )
    codes = np.load(artifacts / "semantic_codes.npy")
    datasets = _datasets(sequences, codes, config)
    semantic, sasrec = _load_models(
        config, features, codes, artifacts, device, sasrec_artifacts
    )
    alphas = tuple(float(value) for value in args.alphas.split(","))
    candidate_ks = tuple(
        value for value in (50, 100, 200, 500) if value <= args.candidate_k
    )
    common = {
        "item_codes": codes,
        "candidate_k": args.candidate_k,
        "beam_size": args.beam_size,
        "candidate_ks": candidate_ks,
        "final_ks": tuple(config["topk"]),
        "batch_size": args.batch_size,
        "device": device,
        "exclude_seen": config.get("exclude_seen_items", False),
    }
    validation_pair = _subset_pair(datasets["validation"], args.max_users, seed)
    validation = evaluate_generated_candidates_with_sasrec(
        semantic,
        sasrec,
        *validation_pair,
        fusion_alphas=alphas,
        **common,
    )
    monitor = config.get("monitor_metric", "recall@20")
    best_key = max(
        validation["rerank_metrics"],
        key=lambda key: validation["rerank_metrics"][key][monitor],
    )
    best_alpha = float(best_key.split("=")[1])

    test_pair = _subset_pair(datasets["test"], args.max_users, seed + 1)
    test_alphas = tuple(dict.fromkeys((0.0, best_alpha, 1.0)))
    test = evaluate_generated_candidates_with_sasrec(
        semantic,
        sasrec,
        *test_pair,
        # alpha=0/1 是预先声明的 SASRec-rerank/Pure-Gen 基线；
        # best_alpha 是唯一由验证集选择的融合配置。
        fusion_alphas=test_alphas,
        **common,
    )
    result = {
        "system": "Gen-Rerank",
        "selection_protocol": "fusion alpha selected on validation only",
        "artifacts": str(artifacts),
        "sasrec_artifacts": str(sasrec_artifacts),
        "users": {
            "validation": len(validation_pair[0]),
            "test": len(test_pair[0]),
        },
        "best_fusion_alpha": best_alpha,
        "validation": validation,
        "test": test,
    }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
