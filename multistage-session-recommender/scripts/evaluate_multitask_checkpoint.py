"""Evaluate a saved MMoE/PLE checkpoint without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_multitask_ranker import evaluate
from src.ranking.neural import MMoE, PLE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["mmoe", "ple"], default="mmoe")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    columns = checkpoint["feature_columns"]
    category_vocab, type_vocab = checkpoint["category_vocab"], checkpoint["type_vocab"]
    model_args = {"category_vocab_size": len(category_vocab) + 1, "type_vocab_size": len(type_vocab) + 1}
    model = MMoE(len(columns), **model_args) if args.model == "mmoe" else PLE(len(columns), **model_args)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)
    metrics = evaluate(
        model, Path(args.features), columns, checkpoint["mean"], checkpoint["std"], args.device,
        category_vocab, type_vocab,
    )
    report = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "features": args.features,
        "validation": metrics,
        "training_performed": False,
        "test_evaluated": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
