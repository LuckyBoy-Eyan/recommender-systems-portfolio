"""Fail-fast CUDA and project-data preflight for the Windows training machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Validate CUDA, mixed precision, config paths and a representative forward pass."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/kuairec_big_cuda.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Reinstall the CUDA build of PyTorch and verify "
            "the NVIDIA driver with nvidia-smi."
        )
    interactions = ROOT / config["interactions_path"]
    features_path = ROOT / config["item_features_path"]
    missing = [str(path) for path in (interactions, features_path) if not path.exists()]
    if missing:
        raise SystemExit(f"Missing transferred data files: {missing}")

    device = torch.device("cuda")
    features = np.load(features_path, mmap_mode="r")
    # Representative attention workload; this also catches CUDA OOM early.
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    probe = torch.randn(
        batch_size,
        config["max_history"],
        hidden_dim,
        device=device,
        dtype=torch.float16,
    )
    layer = torch.nn.TransformerEncoderLayer(
        hidden_dim,
        config["num_heads"],
        hidden_dim * 4,
        batch_first=True,
        device=device,
        dtype=torch.float16,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = layer(probe)
    del output, probe, layer
    torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(0)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "vram_gb": round(properties.total_memory / 1024**3, 2),
                "compute_capability": [properties.major, properties.minor],
                "item_features": list(features.shape),
                "batch_size_probe": batch_size,
                "mixed_precision_probe": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
