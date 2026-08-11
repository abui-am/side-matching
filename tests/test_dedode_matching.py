"""Tests for DeDoDe similarity matrix construction."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from sides_matching.dedode_matching import DeDoDeImageFeatures, DeDoDeMatcher


def _fake_features(tag: str) -> DeDoDeImageFeatures:
    idx = {"a": 0, "b": 1, "c": 2}[tag]
    k = torch.tensor([[[float(idx), 0.0]]], dtype=torch.float32)
    c = torch.tensor([[1.0]], dtype=torch.float32)
    d = torch.tensor([[[float(idx)]]], dtype=torch.float32)
    return DeDoDeImageFeatures(keypoints=k, confidence=c, descriptions=d)


def test_match_count_uses_matcher():
    matcher = DeDoDeMatcher.__new__(DeDoDeMatcher)
    matcher.device = torch.device("cpu")
    matcher.matcher = MagicMock()
    matcher.inv_temp = 20.0
    matcher.threshold = 0.1
    matcher.matcher.match.return_value = (torch.zeros(3), None, None)

    count = matcher.match_count(_fake_features("a"), _fake_features("b"))
    assert count == 3
    matcher.matcher.match.assert_called_once()


def test_compute_similarity_sets_diagonal_to_neg_inf():
    matcher = DeDoDeMatcher.__new__(DeDoDeMatcher)
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
    matcher = DeDoDeMatcher.__new__(DeDoDeMatcher)
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
    assert sim[1, 1] == -np.inf
    assert sim[1, 2] == -np.inf
    assert matcher.match_count.call_count == 3


def test_feature_roundtrip_cpu_dict():
    feat = _fake_features("c")
    restored = DeDoDeImageFeatures.from_cpu_dict(feat.to_cpu_dict())
    assert torch.equal(restored.keypoints, feat.keypoints)
    assert torch.equal(restored.confidence, feat.confidence)
    assert torch.equal(restored.descriptions, feat.descriptions)
