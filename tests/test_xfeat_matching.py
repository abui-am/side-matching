"""Tests for XFeat shortlist similarity construction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from sides_matching.xfeat_matching import (
    XFeatImageFeatures,
    XFeatMatcher,
    load_xfeat_image,
    xfeat_cache_tag,
    xfeat_resize_tag,
)


def _fake_features(tag: str) -> XFeatImageFeatures:
    idx = {"a": 0, "b": 1, "c": 2}[tag]
    k = torch.tensor([[float(idx), 0.0]], dtype=torch.float32)
    d = torch.tensor([[float(idx)] * 64], dtype=torch.float32)
    s = torch.tensor([1.0], dtype=torch.float32)
    return XFeatImageFeatures(keypoints=k, descriptors=d, scores=s)


def test_compute_shortlist_similarity_only_fills_shortlist():
    matcher = XFeatMatcher.__new__(XFeatMatcher)
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
    restored = XFeatImageFeatures.from_cpu_dict(feat.to_cpu_dict())
    assert torch.equal(restored.keypoints, feat.keypoints)
    assert torch.equal(restored.descriptors, feat.descriptors)
    assert torch.equal(restored.scores, feat.scores)


def test_xfeat_resize_tag():
    assert xfeat_resize_tag(max_size=800, square_size=None) == "max800"
    assert xfeat_resize_tag(max_size=None, square_size=512) == "sq512"
    assert xfeat_resize_tag(max_size=None, square_size=None) == "native"


def test_xfeat_cache_tag_kepala():
    assert xfeat_cache_tag(max_size=None, square_size=512, use_kepala=False) == "sq512"
    assert xfeat_cache_tag(max_size=None, square_size=512, use_kepala=True) == "kepala_sq512"
    assert (
        xfeat_cache_tag(
            max_size=None,
            square_size=512,
            use_kepala=True,
            min_area_fraction=0.10,
        )
        == "kepala_min10_sq512"
    )


def test_load_xfeat_image_square_vs_native(tmp_path: Path):
    path = tmp_path / "im.png"
    Image.new("RGB", (800, 600), color=(10, 20, 30)).save(path)
    device = torch.device("cpu")

    square = load_xfeat_image(
        path, flip=False, device=device, max_size=None, square_size=512
    )
    native = load_xfeat_image(
        path, flip=False, device=device, max_size=None, square_size=None
    )
    assert square.shape == (1, 3, 512, 512)
    assert native.shape == (1, 3, 600, 800)
