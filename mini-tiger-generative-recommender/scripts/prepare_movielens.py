"""把 GroupLens MovieLens latest-small 转换为 MiniTIGER 输入格式。

输入：
    <source>/ratings.csv
    <source>/movies.csv

输出：
    <output>/interactions.csv
    <output>/item_features.npy

输出文件随后由 src.data.load.load_interactions 读取。这个脚本只做数据准备，
不构造 Semantic ID，也不训练推荐模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """解析 MovieLens 目录和过滤条件，生成交互表与 genre 特征。

    参数:
        无 Python 函数参数。命令行参数包括：
        ``--source``: 必填，包含 ratings.csv 和 movies.csv 的目录；
        ``--output``: 转换后文件的保存目录；
        ``--catalog-size``: 按正反馈次数保留的热门电影上限；
        ``--min-rating``: 被视为正反馈的最低评分；
        ``--min-sequence-length``: 用户至少需要保留的正反馈数。

    返回:
        None。函数写出 CSV/NPY 文件，并打印用户数、物品数和事件数。

    调用:
        文件底部入口直接调用；生成的文件由 load_interactions 间接使用。
    """
    # 命令行参数用于指定 MovieLens 原始目录、输出目录和过滤规则。
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Directory containing ratings.csv and movies.csv")
    parser.add_argument("--output", default="data/movielens", help="转换结果目录")
    parser.add_argument(
        "--catalog-size", type=int, default=500, help="保留的热门电影数量上限"
    )
    parser.add_argument(
        "--min-rating", type=float, default=4.0, help="正反馈最低评分"
    )
    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=5,
        help="过滤后每个用户至少保留的交互数",
    )
    args = parser.parse_args()

    # 转成 Path 后可用 / 运算符安全拼接文件名。
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # ratings.csv 是用户评分行为；movies.csv 提供电影标题和 genre 元数据。
    ratings = pd.read_csv(source / "ratings.csv")
    movies = pd.read_csv(source / "movies.csv")
    # 只保留高评分作为正反馈，低评分和负反馈不参与这个简化实验。
    ratings = ratings[ratings["rating"] >= args.min_rating]
    # 只取交互最多的前 catalog_size 部电影，控制实验规模。
    catalog = ratings["movieId"].value_counts().head(args.catalog_size).index
    ratings = ratings[ratings["movieId"].isin(catalog)]
    # 过滤交互太少的用户，保证每个用户至少能构造出历史和目标。
    lengths = ratings.groupby("userId").size()
    ratings = ratings[ratings["userId"].isin(lengths[lengths >= args.min_sequence_length].index)]
    # 统一列名为项目内部约定的 user_id、item_id、timestamp。
    ratings = ratings.rename(
        columns={"userId": "user_id", "movieId": "item_id"}
    )[["user_id", "item_id", "timestamp"]].sort_values(["user_id", "timestamp"])
    # 保存成通用交互文件，后续 load_interactions 会读取它。
    ratings.to_csv(output / "interactions.csv", index=False)

    # item_features.npy 的行顺序必须和排序后的 item_id 一致。
    ordered_items = sorted(ratings["item_id"].unique())
    movies = movies.set_index("movieId").loc[ordered_items]
    # 收集所有 genre，构造 genre multi-hot 特征。
    genres = sorted({genre for value in movies["genres"] for genre in value.split("|")})
    # 第 i 行第 j 列表示第 i 部电影是否具有第 j 个 genre。
    features = np.asarray(
        [[float(genre in value.split("|")) for genre in genres] for value in movies["genres"]],
        dtype=np.float32,
    )
    # 按每部电影的 genre 数量做归一化，避免 genre 多的电影特征整体更大。
    features /= np.maximum(features.sum(axis=1, keepdims=True), 1.0)
    # 保存物品特征矩阵，后续会用它生成 Semantic ID。
    np.save(output / "item_features.npy", features)
    # 打印准备后的数据规模，方便确认过滤结果。
    print(
        {"users": ratings["user_id"].nunique(), "items": len(ordered_items), "events": len(ratings)}
    )


if __name__ == "__main__":
    main()
