"""End-to-end metrics for one action-agnostic recommendation list per sample."""

from __future__ import annotations

import numpy as np
import pandas as pd


ACTION_WEIGHTS = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}


def _uauc(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"uauc": None, "weighted_uauc": None, "eligible_users": 0}
    ranked = frame.copy()
    ranked["score_rank"] = ranked.groupby("visitorid")["final_score"].rank(
        method="average", ascending=True
    )
    grouped = ranked.groupby("visitorid").agg(
        rows=("label", "size"), positives=("label", "sum")
    )
    positive_rank_sum = ranked.loc[ranked["label"].eq(1)].groupby("visitorid")["score_rank"].sum()
    grouped["positive_rank_sum"] = positive_rank_sum
    grouped["negatives"] = grouped["rows"] - grouped["positives"]
    grouped = grouped[(grouped["positives"] > 0) & (grouped["negatives"] > 0)].copy()
    if grouped.empty:
        return {"uauc": None, "weighted_uauc": None, "eligible_users": 0}
    grouped["pairs"] = grouped["positives"] * grouped["negatives"]
    grouped["auc"] = (
        grouped["positive_rank_sum"]
        - grouped["positives"] * (grouped["positives"] + 1) / 2.0
    ) / grouped["pairs"]
    return {
        "uauc": float(grouped["auc"].mean()),
        "weighted_uauc": float(np.average(grouped["auc"], weights=grouped["pairs"])),
        "eligible_users": int(len(grouped)),
    }


def evaluate_unified_toplist(
    scored: pd.DataFrame,
    *,
    k: int = 20,
    sample_users: pd.DataFrame | None = None,
) -> dict:
    required = {"sample_id", "aid", "target_type", "label", "final_score"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"scored缺少字段: {sorted(missing)}")
    ranked = scored.sort_values(
        ["sample_id", "final_score", "aid"], ascending=[True, False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked["rank"] = ranked.groupby("sample_id", sort=False).cumcount() + 1
    samples = ranked[["sample_id", "target_type"]].drop_duplicates("sample_id").copy()
    positives = ranked.loc[ranked["label"].eq(1), ["sample_id", "rank", "final_score"]]
    if positives["sample_id"].duplicated().any():
        raise ValueError("每个样本最多只能有一个目标商品")
    positive_rank = positives.set_index("sample_id")["rank"]
    ranks = samples["sample_id"].map(positive_rank).fillna(np.inf).to_numpy()
    hits = ranks <= k
    reciprocal = np.where(hits, 1.0 / ranks, 0.0)
    ndcg = np.where(hits, 1.0 / np.log2(ranks + 1.0), 0.0)
    action_weight = samples["target_type"].map(ACTION_WEIGHTS).to_numpy(float)

    result = {
        "samples": int(len(samples)),
        "candidate_recall": float(np.isfinite(ranks).mean()),
        f"hit_rate_at_{k}": float(hits.mean()),
        f"recall_at_{k}": float(hits.mean()),  # compatibility alias
        f"mrr_at_{k}": float(reciprocal.mean()),
        f"ndcg_at_{k}": float(ndcg.mean()),
        f"event_weighted_hit_rate_at_{k}": float(np.average(hits, weights=action_weight)),
        f"event_weighted_recall_at_{k}": float(np.average(hits, weights=action_weight)),  # compatibility alias
        f"event_weighted_mrr_at_{k}": float(np.average(reciprocal, weights=action_weight)),
        f"event_weighted_ndcg_at_{k}": float(np.average(ndcg, weights=action_weight)),
    }
    action_recalls = {}
    for action in ACTION_WEIGHTS:
        mask = samples["target_type"].eq(action).to_numpy()
        action_recalls[action] = float(hits[mask].mean()) if mask.any() else 0.0
        result[f"{action}_recall_at_{k}"] = action_recalls[action]
        result[f"{action}_hit_rate_at_{k}"] = action_recalls[action]
    result[f"macro_weighted_recall_at_{k}"] = float(sum(
        ACTION_WEIGHTS[action] * action_recalls[action] for action in ACTION_WEIGHTS
    ))

    positive_score = positives.set_index("sample_id")["final_score"]
    negatives = ranked.loc[ranked["label"].eq(0), ["sample_id", "final_score"]].copy()
    negatives["positive_score"] = negatives["sample_id"].map(positive_score)
    eligible = negatives["positive_score"].notna()
    comparisons = (
        (negatives.loc[eligible, "positive_score"] > negatives.loc[eligible, "final_score"]).astype(float)
        + 0.5 * (negatives.loc[eligible, "positive_score"] == negatives.loc[eligible, "final_score"]).astype(float)
    )
    negatives.loc[eligible, "pair_auc"] = comparisons
    session_auc = negatives.loc[eligible].groupby("sample_id")["pair_auc"].mean()
    all_session_auc = samples["sample_id"].map(session_auc).fillna(0.0).to_numpy()
    result.update({
        "conditional_candidate_auc": float(comparisons.mean()) if len(comparisons) else None,
        "conditional_session_gauc": float(session_auc.mean()) if len(session_auc) else None,
        "conditional_auc_eligible_sessions": int(len(session_auc)),
        "end_to_end_session_gauc": float(all_session_auc.mean()),
        "event_weighted_end_to_end_session_gauc": float(
            np.average(all_session_auc, weights=action_weight)
        ),
    })

    if sample_users is not None:
        users = sample_users[["sample_id", "visitorid"]].drop_duplicates("sample_id")
        with_users = ranked[["sample_id", "label", "final_score"]].merge(
            users, on="sample_id", how="left", validate="many_to_one"
        )
        eligible_samples = set(positives["sample_id"].astype(int))
        conditional = with_users[with_users["sample_id"].isin(eligible_samples)]
        conditional_uauc = _uauc(conditional)
        missing_samples = samples.loc[
            ~samples["sample_id"].isin(eligible_samples), ["sample_id"]
        ].merge(users, on="sample_id", how="left")
        synthetic = missing_samples.assign(label=1, final_score=-np.inf)[
            ["sample_id", "label", "final_score", "visitorid"]
        ]
        end_to_end_uauc = _uauc(pd.concat([with_users, synthetic], ignore_index=True))
        result["conditional_uauc"] = conditional_uauc
        result["end_to_end_uauc"] = end_to_end_uauc
    return result
