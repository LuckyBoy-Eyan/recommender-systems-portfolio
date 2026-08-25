"""按时间桶构建严格因果的召回图、候选和排序特征。"""

from __future__ import annotations

import time

import pandas as pd

from src.features.build import build_candidate_features
from src.recall.sources import (
    build_embedding_neighbors,
    build_itemcf,
    build_popularity,
    recall_candidates,
)


def build_point_in_time_dataset(
    all_events: pd.DataFrame,
    sample_history: pd.DataFrame,
    labels: pd.DataFrame,
    snapshot_interval: int,
    candidates_per_source: int,
    seed: int,
    embedding_config: dict | None = None,
    *,
    progress_prefix: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """使用时间桶快照构建 Point-in-Time 正确的候选与特征。

    参数：
        all_events:
            当前阶段允许访问的全局参考事件池。训练阶段通常只含训练事件；验证阶段
            可以包含训练和已发生的验证事件；测试阶段用于滚动回放。
        sample_history:
            待构造样本的 Session 本地历史，每条事件必须严格早于对应目标时间。
        labels:
            每个 Session 一行的标签表，至少包含 ``session`` 和 ``target_ts``；
            通常还包含 ``target_aid`` 和 ``target_type``。
        snapshot_interval:
            快照时间桶宽度，单位必须与 ``ts`` 一致。RetailRocket 的毫秒时间戳使用
            ``86400000``，即一天。
        candidates_per_source:
            每一路召回对每个 Session 最多保留的候选数。
        seed:
            向量召回训练、负采样和初始化使用的随机种子。
        embedding_config:
            向量召回配置。``method`` 可取 ``item2vec``、``svd`` 或 ``none``；
            其余参数传递给相应构建函数。省略时使用 Item2Vec 默认参数。
        progress_prefix:
            可选的进度标签。正式长实验传入 ``train``、``validation`` 或
            ``test``，每个快照完成后输出耗时；单元测试保持 ``None`` 静默运行。

    返回：
        三元组 ``(features, recalled, audit)``：

        - ``features``：每个 ``(session, aid)`` 一行的排序特征；
        - ``recalled``：保留召回来源、排名和分数的长表；
        - ``audit``：每个快照一行的时间审计表，记录快照时间、最大参考事件时间、
          最小目标时间、样本 Session 数和参考事件数。

    因果约束：
        对目标时间 ``t``，先计算
        ``snapshot_ts = floor(t / snapshot_interval) * snapshot_interval``。
        全局统计只使用 ``event.ts < snapshot_ts <= t`` 的事件；当前 Session
        历史可以使用到目标之前，但必须满足 ``history.ts < target_ts``。

    异常：
        ValueError:
            快照间隔非法、标签缺少时间列，或历史 Session 找不到对应标签时抛出。
        AssertionError:
            本地历史或全局参考事件触及目标时间，存在未来信息泄漏时抛出。
    """
    if snapshot_interval <= 0:
        raise ValueError("snapshot_interval must be positive")
    embedding_config = dict(embedding_config or {})
    embedding_method = embedding_config.pop("method", "item2vec")
    required = {"session", "target_ts"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels are missing point-in-time columns: {sorted(missing)}")

    # 首先审计 Session 本地历史，防止目标事件或同时事件混入实时上下文。
    local_times = sample_history[["session", "ts"]].merge(
        labels[["session", "target_ts"]], on="session", how="left", validate="many_to_one"
    )
    if local_times["target_ts"].isna().any():
        raise ValueError("sample history contains a session without a target timestamp")
    if (local_times["ts"] >= local_times["target_ts"]).any():
        raise AssertionError("point-in-time violation: session history reaches target time")

    # 向下取整到时间桶起点。例如日级快照会映射到目标所在日期 00:00。
    timed_labels = labels[["session", "target_ts"]].copy()
    timed_labels["snapshot_ts"] = (
        timed_labels["target_ts"] // snapshot_interval
    ) * snapshot_interval

    feature_batches = []
    recall_batches = []
    audit_rows = []
    grouped_labels = list(timed_labels.groupby("snapshot_ts", sort=True))
    for snapshot_index, (snapshot_ts, label_batch) in enumerate(grouped_labels, start=1):
        snapshot_started = time.perf_counter()
        # 同一快照下的多个 Session 共享一套全局召回图和商品统计。
        session_ids = set(label_batch["session"])
        history_batch = sample_history[sample_history["session"].isin(session_ids)]
        # 严格小于快照时间，意味着目标所在时间桶内的全局事件不会进入统计。
        reference_events = all_events[all_events["ts"] < snapshot_ts]

        max_reference_ts = None if reference_events.empty else int(reference_events["ts"].max())
        min_target_ts = int(label_batch["target_ts"].min())
        if max_reference_ts is not None and not max_reference_ts < min_target_ts:
            raise AssertionError("point-in-time violation: reference event reaches target time")

        # 空快照不能训练全局结构，此时保留空列表/字典，Recent 仍可正常召回。
        popularity = build_popularity(reference_events) if not reference_events.empty else []
        itemcf = build_itemcf(reference_events) if not reference_events.empty else {}
        # 至少需要两个 Session 和两个商品才能训练有意义的向量表示。这里传入的
        # reference_events 已严格早于 snapshot_ts，Item2Vec 不会看到当天或未来事件。
        embedding = (
            build_embedding_neighbors(
                reference_events,
                method=embedding_method,
                seed=seed,
                **embedding_config,
            )
            if reference_events["session"].nunique() >= 2
            and reference_events["aid"].nunique() >= 2
            else {}
        )
        recalled = recall_candidates(
            history_batch,
            popularity,
            itemcf,
            candidates_per_source,
            embedding,
            embedding_source=embedding_method,
        )
        features = build_candidate_features(history_batch, recalled, reference_events)
        # snapshot_ts 仅用于审计和追踪，排序模型会把它排除在训练特征之外。
        features["snapshot_ts"] = int(snapshot_ts)
        recalled["snapshot_ts"] = int(snapshot_ts)
        feature_batches.append(features)
        recall_batches.append(recalled)
        audit_rows.append(
            {
                "snapshot_ts": int(snapshot_ts),
                "max_reference_ts": max_reference_ts,
                "min_target_ts": min_target_ts,
                "sessions": len(session_ids),
                "reference_events": len(reference_events),
                "embedding_method": embedding_method,
                "duration_seconds": time.perf_counter() - snapshot_started,
            }
        )
        if progress_prefix:
            duration = audit_rows[-1]["duration_seconds"]
            print(
                f"[{progress_prefix}] snapshot={snapshot_index}/{len(grouped_labels)} "
                f"ts={int(snapshot_ts)} sessions={len(session_ids)} "
                f"reference_events={len(reference_events)} seconds={duration:.2f}",
                flush=True,
            )

    return (
        pd.concat(feature_batches, ignore_index=True).fillna(0),
        pd.concat(recall_batches, ignore_index=True),
        pd.DataFrame(audit_rows),
    )
