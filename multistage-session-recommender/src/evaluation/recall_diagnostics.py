"""多路召回的独占命中、边际增量、重合度和条件 AUC/GAUC。"""

from __future__ import annotations

from itertools import combinations

import pandas as pd


def route_diagnostics(recalled: pd.DataFrame, labels: pd.DataFrame) -> dict:
    labels = labels.set_index("session")
    source_sets = {
        source: set(zip(group["session"].astype(int), group["aid"].astype(int)))
        for source, group in recalled.groupby("source")
    }
    target_pairs = {
        (int(session), int(row.target_aid)) for session, row in labels.iterrows()
    }
    hits = {source: pairs & target_pairs for source, pairs in source_sets.items()}
    union_hits = set().union(*hits.values()) if hits else set()
    output = {
        "union_recall": len(union_hits) / len(labels),
        "sources": {},
        "pairwise_candidate_jaccard": {},
    }
    for source in sorted(hits):
        other_union = set().union(
            *(value for name, value in hits.items() if name != source)
        ) if len(hits) > 1 else set()
        output["sources"][source] = {
            "recall": len(hits[source]) / len(labels),
            "exclusive_hits": len(hits[source] - other_union),
            "leave_one_out_recall_drop": len(union_hits - other_union) / len(labels),
        }
    for left, right in combinations(sorted(source_sets), 2):
        union = source_sets[left] | source_sets[right]
        output["pairwise_candidate_jaccard"][f"{left}__{right}"] = (
            len(source_sets[left] & source_sets[right]) / len(union) if union else 0.0
        )
    return output


def conditional_auc_gauc(recalled: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """目标已召回时，计算来源候选集内的成对 AUC 和 Session GAUC。"""
    target = dict(zip(labels["session"].astype(int), labels["target_aid"].astype(int)))
    output = {}
    for source, source_frame in recalled.groupby("source"):
        session_aucs = []
        pair_wins = 0.0
        pair_total = 0
        for session, group in source_frame.groupby("session"):
            target_aid = target.get(int(session))
            positive = group[group["aid"].eq(target_aid)]
            negatives = group[~group["aid"].eq(target_aid)]
            if positive.empty or negatives.empty:
                continue
            score = float(positive["source_score"].max())
            wins = float((score > negatives["source_score"]).sum())
            ties = float((score == negatives["source_score"]).sum())
            auc = (wins + 0.5 * ties) / len(negatives)
            session_aucs.append(auc)
            pair_wins += wins + 0.5 * ties
            pair_total += len(negatives)
        output[source] = {
            "candidate_auc": pair_wins / pair_total if pair_total else None,
            "session_gauc": sum(session_aucs) / len(session_aucs)
            if session_aucs
            else None,
            "eligible_sessions": len(session_aucs),
            "total_sessions": int(labels["session"].nunique()),
        }
    return output
