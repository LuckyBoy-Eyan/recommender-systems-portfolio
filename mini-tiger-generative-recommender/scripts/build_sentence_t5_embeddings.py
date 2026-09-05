"""使用 Sentence-T5 将对齐的物品文本编码为 768 维内容向量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="sentence-transformers/sentence-t5-base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit(
            "Install sentence-transformers before building Sentence-T5 embeddings"
        ) from error
    frame = pd.read_csv(args.texts).sort_values("item_id")
    model = SentenceTransformer(args.model, device=args.device)
    embeddings = model.encode(
        frame["text"].fillna("无可用文本").tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)
    manifest = {
        "model": args.model,
        "items": len(frame),
        "dimension": int(embeddings.shape[1]),
        "normalized": True,
        "item_order": "ascending item_id from item_texts.csv",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
