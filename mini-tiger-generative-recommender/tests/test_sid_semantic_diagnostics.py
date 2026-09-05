import numpy as np

from scripts.analyze_sid_semantics import analyze_category


def test_category_diagnostics_detect_semantic_prefixes():
    labels = np.asarray(["A", "A", "A", "B", "B", "B"])
    codes = np.asarray(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]]
    )
    result = analyze_category(labels, codes)
    assert result["shared_prefix"][0]["same_category_match_rate"] == 1.0
    assert result["shared_prefix"][0]["different_category_match_rate"] == 0.0
    assert (
        result["mean_hamming_distance"]["same_category"]
        < result["mean_hamming_distance"]["different_category"]
    )
