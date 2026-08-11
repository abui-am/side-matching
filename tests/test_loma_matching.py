"""Tests for LoMa similarity matrix construction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sides_matching.loma_matching import (
    LomaImageFeatures,
    LomaMatcher,
    _square_dim,
    loma_resize_tag,
)


def _fake_features(tag: str) -> LomaImageFeatures:
    idx = {"a": 0, "b": 1, "c": 2}[tag]
    k = torch.tensor([[[float(idx), 0.0]]], dtype=torch.float32)
    d = torch.tensor([[[float(idx)]]], dtype=torch.float32)
    return LomaImageFeatures(keypoints=k, descriptors=d, height=64, width=64)


def test_match_count_delegates_to_model():
    matcher = LomaMatcher(model=MagicMock())
    matcher.model.return_value = {"scores": torch.zeros(1, 1, 1)}
    matcher.threshold = 0.1

    with patch(
        "sides_matching.loma_matching.filter_matches",
        return_value=(torch.tensor([[0]]), None, None, None),
    ):
        count = matcher.match_count(_fake_features("a"), _fake_features("b"))

    assert count == 1
    matcher.model.assert_called_once()


def test_compute_similarity_sets_diagonal_to_neg_inf():
    matcher = LomaMatcher(model=MagicMock())
    matcher.match_count = MagicMock(side_effect=[10, 20])

    sim = matcher.compute_similarity(
        [_fake_features("a"), _fake_features("b")],
        [_fake_features("a"), _fake_features("b")],
        ignore_diagonal=True,
        show_progress=False,
    )

    assert sim.shape == (2, 2)
    assert sim[0, 0] == -np.inf
    assert sim[1, 1] == -np.inf
    assert sim[0, 1] == 10
    assert sim[1, 0] == 20


def test_compute_shortlist_similarity_only_fills_shortlist():
    matcher = LomaMatcher(model=MagicMock())
    matcher.match_count = MagicMock(side_effect=[7, 8, 9])

    sim = matcher.compute_shortlist_similarity(
        [_fake_features("a"), _fake_features("b")],
        [_fake_features("a"), _fake_features("b"), _fake_features("c")],
        shortlists=[[1, 2], [0]],
        show_progress=False,
    )

    assert sim.shape == (2, 3)
    assert sim[0, 0] == -np.inf
    assert sim[0, 1] == 7
    assert sim[0, 2] == 8
    assert sim[1, 0] == 9
    assert matcher.match_count.call_count == 3


def test_feature_roundtrip_cpu_dict():
    feat = _fake_features("c")
    restored = LomaImageFeatures.from_cpu_dict(feat.to_cpu_dict())
    assert restored.height == feat.height
    assert restored.width == feat.width
    assert torch.equal(restored.keypoints, feat.keypoints)
    assert torch.equal(restored.descriptors, feat.descriptors)


def test_square_dim_patch_aligns():
    assert _square_dim(512) == 504
    assert _square_dim(504) == 504
    assert loma_resize_tag(resize=None, square_size=512) == "sq512"
    assert loma_resize_tag(resize=1024, square_size=None) == "max1024"
