"""Tests for YOLO kepala preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from sides_matching.yolo_kepala_preprocessing import (
    KepalaCropper,
    apply_kepala_crop,
    box_area_fraction,
    expand_box,
    preprocess_tag,
    select_best_box_xyxy,
)


@dataclass
class _FakeBox:
    conf: float
    xyxy: list[float]


class _FakeBoxes:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self._boxes = boxes

    def __len__(self) -> int:
        return len(self._boxes)

    @property
    def conf(self):
        return np.array([b.conf for b in self._boxes])

    def __getitem__(self, idx: int) -> _FakeBox:
        return self._boxes[idx]


def test_preprocess_tag():
    assert preprocess_tag(use_kepala=False) == "full"
    assert preprocess_tag(use_kepala=True) == "kepala"
    assert preprocess_tag(use_kepala=True, min_area_fraction=0.10) == "kepala_min10"


def test_box_area_fraction():
    assert box_area_fraction(0, 0, 50, 50, 100, 100) == 0.25
    assert box_area_fraction(0, 0, 10, 10, 100, 100) == 0.01


def test_select_best_box_xyxy():
    boxes = _FakeBoxes(
        [
            _FakeBox(conf=0.2, xyxy=[0, 0, 10, 10]),
            _FakeBox(conf=0.9, xyxy=[1, 2, 3, 4]),
        ]
    )
    assert select_best_box_xyxy(boxes) == (1.0, 2.0, 3.0, 4.0)
    assert select_best_box_xyxy(_FakeBoxes([])) is None


def test_expand_box_clamps_to_image():
    x1, y1, x2, y2 = expand_box(
        5, 5, 15, 15, pad_fraction=0.5, width=20, height=20
    )
    assert x1 >= 0
    assert y1 >= 0
    assert x2 <= 20
    assert y2 <= 20
    assert x2 > x1 and y2 > y1


def test_apply_kepala_crop_no_cropper():
    im = Image.new("RGB", (40, 30), color=(1, 2, 3))
    out = apply_kepala_crop(im, None)
    assert out.size == im.size


def test_kepala_cropper_crops_to_detection():
    im = Image.new("RGB", (100, 80), color=(255, 0, 0))
    cropper = KepalaCropper(weights=MagicMock(), conf=0.05)

    fake_boxes = _FakeBoxes([_FakeBox(conf=0.8, xyxy=[20, 10, 60, 50])])
    fake_result = MagicMock()
    fake_result.boxes = fake_boxes
    fake_model = MagicMock()
    fake_model.predict.return_value = [fake_result]

    with patch.object(cropper, "_load_model", return_value=fake_model):
        cropped = cropper.crop(im)

    assert cropped.size == (44, 44)
    fake_model.predict.assert_called_once()


def test_kepala_cropper_fallback_full_image_when_box_too_small():
    im = Image.new("RGB", (100, 100), color=(0, 0, 255))
    cropper = KepalaCropper(
        weights=MagicMock(), conf=0.05, min_area_fraction=0.10
    )
    fake_boxes = _FakeBoxes([_FakeBox(conf=0.9, xyxy=[0, 0, 20, 20])])
    fake_result = MagicMock()
    fake_result.boxes = fake_boxes
    fake_model = MagicMock()
    fake_model.predict.return_value = [fake_result]

    with patch.object(cropper, "_load_model", return_value=fake_model):
        out = cropper.crop(im)

    assert out.size == im.size


def test_kepala_cropper_fallback_full_image_when_no_detection():
    im = Image.new("RGB", (100, 80), color=(0, 255, 0))
    cropper = KepalaCropper(weights=MagicMock(), conf=0.05)
    fake_result = MagicMock()
    fake_result.boxes = _FakeBoxes([])
    fake_model = MagicMock()
    fake_model.predict.return_value = [fake_result]

    with patch.object(cropper, "_load_model", return_value=fake_model):
        out = cropper.crop(im)

    assert out.size == im.size
