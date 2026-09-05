"""对比两个 RQ-KMeans SID 索引的结构质量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _summary(path: str) -> dict:
    manifest = json.loads(Path(path).read_text())
    levels = manifest["prefix_diagnostics"]["levels"]
    rq_levels = manifest["rq_kmeans"]["level_metrics"]
    return {
        "codebook_sizes": manifest["codebook_sizes"],
        "prefix_collision_rate": manifest["prefix_collision_rate"],
        "tail_size": manifest["tail_size"],
        "complete_collision_rate": manifest["complete_collision_rate"],
        "level_utilization": [level["utilization"] for level in levels],
        "normalized_entropy": [level["normalized_entropy"] for level in levels],
        "largest_bucket": [level["largest_bucket"] for level in levels],
        "quantization_mse": [level["quantization_mse"] for level in rq_levels],
        "final_residual_l2_mean": rq_levels[-1]["residual_l2_mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid32", required=True)
    parser.add_argument("--sid64", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        "sid_32_32_32": _summary(args.sid32),
        "sid_64_32_32": _summary(args.sid64),
        "decision_rule": {
            "required": "complete_collision_rate must be 0 for both",
            "prefer_64_if": [
                "prefix collision rate or tail size is materially lower",
                "first-level normalized entropy remains healthy",
                "final residual or quantization error does not worsen",
            ],
            "otherwise": "keep 32x32x32 and avoid unnecessary first-token branching",
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
