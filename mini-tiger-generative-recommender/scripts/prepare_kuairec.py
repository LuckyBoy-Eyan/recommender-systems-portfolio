"""把 KuaiRec Big Matrix 转换为 MiniTIGER 可训练格式。

脚本可直接读取官方 ``KuaiRec.zip``，无需预先解压。它会：

1. 分块读取 big_matrix.csv；
2. 根据观看比例与播放时长筛选正反馈；
3. 进行用户/物品 k-core 过滤；
4. 合并层级类别、标签、caption 和封面文字；
5. 使用 TF-IDF + TruncatedSVD 构建紧凑内容向量；
6. 写出 interactions.csv、item_features.npy、item_ids.npy 和 stats.json。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

# sklearn、NumPy 和 PyTorch 在部分 macOS 环境同时初始化 OpenMP 会冲突。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@contextmanager
def open_dataset_file(source: Path, filename: str):
    """从官方 ZIP 或已解压目录中打开指定文件。"""
    if source.is_file():
        with ZipFile(source) as archive:
            matches = [name for name in archive.namelist() if name.endswith(filename)]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one {filename} in {source}, found {matches}"
                )
            with archive.open(matches[0]) as handle:
                yield handle
    else:
        matches = list(source.rglob(filename))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one {filename} under {source}, found {matches}"
            )
        yield matches[0]


def read_metadata(source: Path, filename: str) -> pd.DataFrame:
    """读取目录中规模较小的视频元数据表。"""
    with open_dataset_file(source, filename) as handle:
        # caption 中存在很长的用户文本，Python 解析器比 C 解析器更稳健。
        if filename == "kuairec_caption_category.csv":
            return pd.read_csv(handle, engine="python", on_bad_lines="warn")
        return pd.read_csv(handle)


def read_positive_events(
    source: Path,
    *,
    min_watch_ratio: float,
    min_play_seconds: float,
    chunk_size: int,
) -> pd.DataFrame:
    """分块读取 Big Matrix 并仅保留满足阈值的观看行为。"""
    usecols = [
        "user_id",
        "video_id",
        "timestamp",
        "play_duration",
        "watch_ratio",
    ]
    positive_chunks = []
    with open_dataset_file(source, "big_matrix.csv") as handle:
        chunks = pd.read_csv(
            handle,
            usecols=usecols,
            dtype={
                "user_id": "int32",
                "video_id": "int32",
                "timestamp": "float64",
                "play_duration": "float32",
                "watch_ratio": "float32",
            },
            chunksize=chunk_size,
        )
        for chunk_number, chunk in enumerate(chunks, start=1):
            positive = chunk[
                chunk["watch_ratio"].ge(min_watch_ratio)
                & chunk["play_duration"].ge(min_play_seconds * 1000.0)
            ][["user_id", "video_id", "timestamp"]]
            positive_chunks.append(positive)
            print(
                f"chunk={chunk_number} rows={len(chunk):,} "
                f"positive={len(positive):,}"
            )
    if not positive_chunks:
        raise ValueError("No chunks were read from big_matrix.csv")
    events = pd.concat(positive_chunks, ignore_index=True)
    if events.empty:
        raise ValueError("No positive events remain; lower the feedback thresholds")
    return events


def k_core_filter(
    events: pd.DataFrame,
    *,
    min_user_events: int,
    min_item_events: int,
) -> pd.DataFrame:
    """迭代过滤低频用户和物品，直到用户/物品集合稳定。"""
    filtered = events
    while True:
        before = len(filtered)
        user_counts = filtered["user_id"].value_counts()
        filtered = filtered[
            filtered["user_id"].isin(user_counts[user_counts.ge(min_user_events)].index)
        ]
        item_counts = filtered["video_id"].value_counts()
        filtered = filtered[
            filtered["video_id"].isin(item_counts[item_counts.ge(min_item_events)].index)
        ]
        if len(filtered) == before:
            break
    return filtered


def limit_catalog(
    events: pd.DataFrame,
    *,
    max_users: int | None,
    max_items: int | None,
    min_user_events: int,
    min_item_events: int,
) -> pd.DataFrame:
    """为快速实验可选地保留最活跃用户和最常见视频，再重新做 k-core。"""
    limited = events
    if max_items is not None:
        items = limited["video_id"].value_counts().head(max_items).index
        limited = limited[limited["video_id"].isin(items)]
    if max_users is not None:
        users = limited["user_id"].value_counts().head(max_users).index
        limited = limited[limited["user_id"].isin(users)]
    return k_core_filter(
        limited,
        min_user_events=min_user_events,
        min_item_events=min_item_events,
    )


def _parse_tags(value) -> list[str]:
    """把 item_categories.csv 中形如 ``[27, 9]`` 的字段转为 token。"""
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    return [f"tag={int(tag)}" for tag in parsed]


def build_item_features(
    source: Path,
    item_ids: np.ndarray,
    *,
    feature_dim: int,
    text_max_features: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """由视频层级类别、标签、caption 和封面文字构建内容特征。

    文本使用字符 1-2 gram TF-IDF，离散类别使用 DictVectorizer one-hot；两部分
    拼接后通过 TruncatedSVD 压缩到 feature_dim，并进行 L2 归一化。
    """
    categories = read_metadata(source, "item_categories.csv")
    category_map = categories.set_index("video_id")["feat"].to_dict()

    try:
        captions = read_metadata(source, "kuairec_caption_category.csv")
    except FileNotFoundError:
        captions = pd.DataFrame({"video_id": item_ids})
    # 官方 caption 文件含少量不规则文本行，稳健解析后 video_id 会成为 object；
    # 显式转回整数并丢弃无法恢复的行，确保能与 Big Matrix 的整数 video_id 对齐。
    captions["video_id"] = pd.to_numeric(captions["video_id"], errors="coerce")
    captions = captions.dropna(subset=["video_id"])
    captions["video_id"] = captions["video_id"].astype(np.int64)
    captions = captions.drop_duplicates("video_id").set_index("video_id")

    text_columns = [
        "manual_cover_text",
        "caption",
        "topic_tag",
        "first_level_category_name",
        "second_level_category_name",
        "third_level_category_name",
    ]
    category_columns = [
        "first_level_category_id",
        "second_level_category_id",
        "third_level_category_id",
    ]
    texts, categorical_rows = [], []
    items_with_caption = 0
    for item in item_ids.tolist():
        text_parts, categorical = [], {}
        if item in captions.index:
            row = captions.loc[item]
            for column in text_columns:
                if column in row and pd.notna(row[column]):
                    text_parts.append(str(row[column]))
            for column in category_columns:
                if column in row and pd.notna(row[column]):
                    categorical[f"{column}={int(row[column])}"] = 1.0
            if text_parts:
                items_with_caption += 1
        for token in _parse_tags(category_map.get(item)):
            categorical[token] = 1.0
        texts.append(" ".join(text_parts) or "无文本")
        categorical_rows.append(categorical or {"metadata=missing": 1.0})

    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 2),
        min_df=2,
        max_features=text_max_features,
        sublinear_tf=True,
    )
    text_matrix = tfidf.fit_transform(texts)
    categorical_matrix = DictVectorizer(sparse=True).fit_transform(categorical_rows)
    combined = hstack([text_matrix, categorical_matrix], format="csr")

    components = min(feature_dim, combined.shape[0] - 1, combined.shape[1] - 1)
    if components < 1:
        raise ValueError("Not enough item metadata to build content features")
    features = TruncatedSVD(n_components=components, random_state=seed).fit_transform(
        combined
    )
    if components < feature_dim:
        features = np.pad(features, ((0, 0), (0, feature_dim - components)))
    features = normalize(features).astype(np.float32)
    diagnostics = {
        "feature_dim": feature_dim,
        "text_vocabulary_size": len(tfidf.vocabulary_),
        "categorical_feature_count": categorical_matrix.shape[1],
        "items_with_text_or_category_name": items_with_caption,
    }
    return features, diagnostics


def build_collaborative_features(
    events: pd.DataFrame,
    item_ids: np.ndarray,
    *,
    feature_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """仅用每个用户验证/测试之前的交互构建协同 item embedding。

    先从每个用户序列去掉最后两个事件，再建立 item-user 稀疏矩阵并执行
    TruncatedSVD。这样 Semantic ID 同时反映“哪些用户共同观看这些视频”，
    又不会看到验证与测试目标。
    """
    if feature_dim <= 0:
        return np.empty((len(item_ids), 0), dtype=np.float32), {
            "feature_dim": 0,
            "train_events": 0,
        }
    ordered = events.sort_values(["user_id", "timestamp"]).copy()
    reverse_position = ordered.groupby("user_id").cumcount(ascending=False)
    train_events = ordered[reverse_position.ge(2)]
    user_codes, users = pd.factorize(train_events["user_id"], sort=True)
    item_positions = np.searchsorted(item_ids, train_events["video_id"].to_numpy())
    matrix = coo_matrix(
        (
            np.ones(len(train_events), dtype=np.float32),
            (item_positions, user_codes),
        ),
        shape=(len(item_ids), len(users)),
    ).tocsr()
    matrix.data[:] = 1.0
    components = min(feature_dim, matrix.shape[0] - 1, matrix.shape[1] - 1)
    embedding = TruncatedSVD(
        n_components=components, random_state=seed
    ).fit_transform(matrix)
    if components < feature_dim:
        embedding = np.pad(embedding, ((0, 0), (0, feature_dim - components)))
    embedding = normalize(embedding).astype(np.float32)
    return embedding, {
        "feature_dim": feature_dim,
        "train_events": len(train_events),
        "users": len(users),
        "nonzero_item_user_pairs": int(matrix.nnz),
    }


def main():
    """执行 KuaiRec Big Matrix 的完整准备流程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="官方 KuaiRec.zip 或其解压目录",
    )
    parser.add_argument("--output", default="data/kuairec_big")
    parser.add_argument("--min-watch-ratio", type=float, default=0.7)
    parser.add_argument("--min-play-seconds", type=float, default=5.0)
    parser.add_argument("--min-user-events", type=int, default=20)
    parser.add_argument("--min-item-events", type=int, default=5)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument(
        "--collaborative-dim",
        type=int,
        default=64,
        help="总特征中由无泄漏 item-user SVD 提供的维度",
    )
    parser.add_argument("--text-max-features", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    events = read_positive_events(
        source,
        min_watch_ratio=args.min_watch_ratio,
        min_play_seconds=args.min_play_seconds,
        chunk_size=args.chunk_size,
    )
    raw_positive_events = len(events)
    events = k_core_filter(
        events,
        min_user_events=args.min_user_events,
        min_item_events=args.min_item_events,
    )
    events = limit_catalog(
        events,
        max_users=args.max_users,
        max_items=args.max_items,
        min_user_events=args.min_user_events,
        min_item_events=args.min_item_events,
    )
    events = events.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    if events.empty:
        raise ValueError("No interactions remain after k-core/catalog filtering")

    item_ids = np.sort(events["video_id"].unique()).astype(np.int64)
    if not 0 <= args.collaborative_dim < args.feature_dim:
        raise ValueError("collaborative-dim must be in [0, feature-dim)")
    content_features, content_diagnostics = build_item_features(
        source,
        item_ids,
        feature_dim=args.feature_dim - args.collaborative_dim,
        text_max_features=args.text_max_features,
        seed=args.seed,
    )
    collaborative_features, collaborative_diagnostics = build_collaborative_features(
        events,
        item_ids,
        feature_dim=args.collaborative_dim,
        seed=args.seed,
    )
    features = normalize(
        np.concatenate([content_features, collaborative_features], axis=1)
    ).astype(np.float32)
    events = events.rename(columns={"video_id": "item_id"})
    events.to_csv(output / "interactions.csv", index=False)
    np.save(output / "item_features.npy", features)
    np.save(output / "content_features.npy", content_features)
    np.save(output / "collaborative_features.npy", collaborative_features)
    np.save(output / "item_ids.npy", item_ids)

    per_user = events.groupby("user_id").size()
    stats = {
        "source": "KuaiRec Big Matrix",
        "positive_definition": {
            "min_watch_ratio": args.min_watch_ratio,
            "min_play_seconds": args.min_play_seconds,
        },
        "filters": {
            "min_user_events": args.min_user_events,
            "min_item_events": args.min_item_events,
            "max_users": args.max_users,
            "max_items": args.max_items,
        },
        "raw_positive_events": raw_positive_events,
        "events": len(events),
        "users": int(events["user_id"].nunique()),
        "items": len(item_ids),
        "sequence_length": {
            "min": int(per_user.min()),
            "median": float(per_user.median()),
            "mean": float(per_user.mean()),
            "max": int(per_user.max()),
        },
        "item_features": {
            "feature_dim": args.feature_dim,
            "content": content_diagnostics,
            "collaborative": collaborative_diagnostics,
        },
    }
    (output / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
