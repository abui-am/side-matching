"""Tests for MegaDescriptor extraction with kepala preprocessing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from sides_matching.megadescriptor_matching import (
    MegaDescriptorExtractor,
    md_feature_cache_path,
    md_similarity_cache_path,
)


def test_md_cache_paths():
    cache = Path("/tmp/cache")
    assert md_feature_cache_path(cache, "X", True, preprocess="kepala").name == (
        "X_md_kepala_flipTrue_feats.npy"
    )
    assert md_similarity_cache_path(cache, "X", preprocess="kepala", flip_query=True).name == (
        "X_md_kepala_flipTrue_sim.npy"
    )


def test_megadescriptor_extractor_batch(tmp_path: Path):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (64, 48), color=(i, 0, 0)).save(p)
        paths.append(p)

    cropper = MagicMock()
    cropper.crop.side_effect = lambda im: im

    fake_model = MagicMock()
    fake_model.return_value = torch.ones(2, 4)

    with patch("timm.create_model", return_value=fake_model):
        extractor = MegaDescriptorExtractor(
            device="cpu", img_size=32, batch_size=2, cropper=cropper
        )

    with patch.object(extractor, "model", fake_model):
        fake_model.side_effect = [
            torch.ones(2, 4),
            torch.ones(1, 4) * 2,
        ]
        feats = extractor.extract_features(paths, flip=False)

    assert feats.shape == (3, 4)
    assert cropper.crop.call_count == 3
