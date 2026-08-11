"""Tests for DeDoDe-v2+LightGlue shortlist similarity."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch

from sides_matching.dedode_lg_matching import DeDoDeLGFeatures, DeDoDeLightGlueMatcher


def _fake(tag: str) -> DeDoDeLGFeatures:
    idx = {"a": 0, "b": 1, "c": 2}[tag]
    k = torch.zeros(1, 2, 2)
    k[0, 0, 0] = float(idx)
    d = torch.zeros(1, 2, 256)
    d[0, 0, 0] = float(idx)
    return DeDoDeLGFeatures(keypoints=k, descriptors=d, image_size=(64, 64))


def test_shortlist_similarity_only_fills_shortlist():
    matcher = DeDoDeLightGlueMatcher.__new__(DeDoDeLightGlueMatcher)
    matcher.match_count = MagicMock(side_effect=[11, 12, 13])
    sim = matcher.compute_shortlist_similarity(
        [_fake("a"), _fake("b")],
        [_fake("a"), _fake("b"), _fake("c")],
        shortlists=[[1, 2], [0]],
        show_progress=False,
    )
    assert sim.shape == (2, 3)
    assert sim[0, 0] == -np.inf
    assert sim[0, 1] == 11
    assert sim[0, 2] == 12
    assert sim[1, 0] == 13


def test_feature_roundtrip():
    feat = _fake("c")
    restored = DeDoDeLGFeatures.from_cpu_dict(feat.to_cpu_dict())
    assert restored.image_size == feat.image_size
    assert torch.equal(restored.keypoints, feat.keypoints)
