"""KuaiRec Big Matrix 准备流程的轻量正确性测试。"""

import numpy as np
import pandas as pd

from scripts.prepare_kuairec import (
    build_collaborative_features,
    build_item_features,
    k_core_filter,
)


def test_k_core_filter_removes_low_frequency_users_and_items():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2],
            "video_id": [0, 1, 0, 1, 2],
            "timestamp": [1, 2, 1, 2, 1],
        }
    )
    filtered = k_core_filter(events, min_user_events=2, min_item_events=2)
    assert set(filtered["user_id"]) == {0, 1}
    assert set(filtered["video_id"]) == {0, 1}


def test_content_features_align_with_requested_item_order(tmp_path):
    pd.DataFrame(
        {
            "video_id": [0, 1, 2, 3],
            "feat": ["[1]", "[1, 2]", "[2]", "[3]"],
        }
    ).to_csv(tmp_path / "item_categories.csv", index=False)
    pd.DataFrame(
        {
            "video_id": [0, 1, 2, 3],
            "caption": ["小狗日常", "小狗玩耍", "数码手机", "美食日常"],
            "first_level_category_id": [1, 1, 2, 3],
            "first_level_category_name": ["宠物", "宠物", "数码", "美食"],
        }
    ).to_csv(tmp_path / "kuairec_caption_category.csv", index=False)

    features, diagnostics = build_item_features(
        tmp_path,
        np.asarray([3, 1, 0, 2]),
        feature_dim=3,
        text_max_features=50,
        seed=7,
    )
    assert features.shape == (4, 3)
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)
    assert diagnostics["items_with_text_or_category_name"] == 4


def test_collaborative_features_exclude_last_two_events_per_user():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "video_id": [0, 1, 2, 3, 1, 2, 3, 0],
            "timestamp": [1, 2, 3, 4, 1, 2, 3, 4],
        }
    )
    features, diagnostics = build_collaborative_features(
        events,
        np.asarray([0, 1, 2, 3]),
        feature_dim=2,
        seed=7,
    )
    assert features.shape == (4, 2)
    assert diagnostics["train_events"] == 4
