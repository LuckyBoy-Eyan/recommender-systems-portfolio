"""把 RetailRocket 原始事件转换为项目统一的会话推荐数据。

命令示例：

```
python scripts/prepare_retailrocket.py \
  --events /path/to/raw/events.csv \
  --output data/retailrocket/events.csv
```

脚本会同时生成同名 ``.metadata.json``，记录最终数据量和清洗参数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.split import drop_ambiguous_target_sessions


# 将 RetailRocket 行为统一为 OTTO 风格的多目标事件类型。
TYPE_MAP = {"view": "clicks", "addtocart": "carts", "transaction": "orders"}


def prepare_events(
    events: pd.DataFrame,
    catalog_size: int = 3000,
    max_sessions: int = 20000,
    min_session_length: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """清洗 RetailRocket 事件并生成固定规模的 Session 子集。

    参数：
        events:
            RetailRocket 原始事件表，必须包含 ``timestamp、visitorid、event、itemid``。
        catalog_size:
            按事件频次保留的热门商品数量上限。该限制在 Session 化之后应用，避免删除
            长尾事件后人为制造超过 30 分钟的空档。
        max_sessions:
            最终最多保留的 Session 数，按照原始 Session 开始时间从早到晚选择。
            删除歧义 Session 后会从后续合格 Session 递补。
        min_session_length:
            商品目录过滤后，一个 Session 至少需要保留的事件数。

    返回：
        二元组 ``(output, metadata)``：

        - ``output``：统一为 ``session、aid、ts、type`` 的事件表；
        - ``metadata``：数据规模、行为分布、清洗数量和参数记录。

    Session 规则：
        对同一 ``visitorid`` 按时间排序，相邻事件间隔超过 30 分钟时开启新 Session。
        最大时间戳对应多个事件的 Session 会整体删除，因为无法确定唯一预测标签。
    """
    # 只保留三类推荐目标行为，再按用户和时间排序。
    events = events[events["event"].isin(TYPE_MAP)].sort_values(
        ["visitorid", "timestamp"]
    ).copy()

    # 必须先在完整行为流上切 Session。若先删除长尾商品，原本连续的访问可能被人为
    # 制造出超过 30 分钟的空档，从而错误切成多个 Session。
    gap = events.groupby("visitorid")["timestamp"].diff().fillna(0)
    events["session_index"] = gap.gt(30 * 60 * 1000).groupby(events["visitorid"]).cumsum()
    keys = pd.MultiIndex.from_arrays([events["visitorid"], events["session_index"]])
    events["session"] = pd.factorize(keys)[0]
    original_starts = events.groupby("session")["timestamp"].min()

    # Session 化后再限制商品目录；过滤商品会缩短 Session，因此需要重新检查长度。
    popular = events["itemid"].value_counts().head(catalog_size).index
    events = events[events["itemid"].isin(popular)].copy()
    lengths = events.groupby("session").size()
    events = events[events["session"].isin(lengths[lengths >= min_session_length].index)]

    # 最大时间戳并列时无法定义唯一下一事件标签，删除整个 Session。
    eligible_sessions = events["session"].nunique()
    events = drop_ambiguous_target_sessions(events, ts_column="timestamp")
    dropped_ambiguous_sessions = eligible_sessions - events["session"].nunique()

    # 使用过滤前记录的原始 Session 开始时间选取最早的合格会话。
    eligible_ids = events["session"].unique()
    starts = original_starts.reindex(eligible_ids).nsmallest(max_sessions)
    events = events[events["session"].isin(starts.index)]

    output = events.rename(
        columns={"itemid": "aid", "timestamp": "ts", "event": "type"}
    )[["session", "aid", "ts", "type"]]
    output["type"] = output["type"].map(TYPE_MAP)
    output = output.sort_values(["session", "ts"]).reset_index(drop=True)
    metadata = {
        "sessions": int(output["session"].nunique()),
        "items": int(output["aid"].nunique()),
        "events": len(output),
        "types": {key: int(value) for key, value in output["type"].value_counts().items()},
        "dropped_ambiguous_target_sessions": int(dropped_ambiguous_sessions),
        "catalog_size_limit": catalog_size,
        "max_sessions": max_sessions,
        "min_session_length": min_session_length,
        "sessionization_before_catalog_filter": True,
    }
    return output, metadata


def main():
    """解析命令行参数、执行预处理并写出数据与元数据文件。"""
    parser = argparse.ArgumentParser(description="预处理 RetailRocket 原始事件数据")
    parser.add_argument("--events", required=True, help="原始 events.csv 路径")
    parser.add_argument(
        "--output", default="data/retailrocket/events.csv", help="标准化事件表输出路径"
    )
    parser.add_argument(
        "--catalog-size", type=int, default=3000, help="保留的热门商品数量上限"
    )
    parser.add_argument(
        "--max-sessions", type=int, default=20000, help="最终保留的最大 Session 数"
    )
    parser.add_argument(
        "--min-session-length", type=int, default=3, help="过滤商品后的最短 Session 长度"
    )
    args = parser.parse_args()

    # 仅加载推荐链路需要的字段，降低读取完整公开数据集时的内存占用。
    events = pd.read_csv(args.events, usecols=["timestamp", "visitorid", "event", "itemid"])
    output, metadata = prepare_events(
        events,
        catalog_size=args.catalog_size,
        max_sessions=args.max_sessions,
        min_session_length=args.min_session_length,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    # 元数据与 CSV 同目录同主文件名，例如 events.csv -> events.metadata.json。
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(metadata)


if __name__ == "__main__":
    main()
