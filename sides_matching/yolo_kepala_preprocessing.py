"""YOLO head (kepala) detection and crop preprocessing.

Uses weights from ``yolo_kepala/`` (Ultralytics YOLO11, single class ``kepala``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEPALA_WEIGHTS = REPO_ROOT / "yolo_kepala" / "kepala.pt"


def preprocess_tag(*, use_kepala: bool, min_area_fraction: float = 0.0) -> str:
    if not use_kepala:
        return "full"
    if min_area_fraction > 0:
        return f"kepala_min{int(round(min_area_fraction * 100))}"
    return "kepala"


def box_area_fraction(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> float:
    """Detected box area as a fraction of the full image (before padding)."""
    return max(0.0, (x2 - x1) * (y2 - y1)) / max(1, width * height)


def _clamp_box(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[int, int, int, int]:
    x1_i = max(0, min(width - 1, int(round(x1))))
    y1_i = max(0, min(height - 1, int(round(y1))))
    x2_i = max(x1_i + 1, min(width, int(round(x2))))
    y2_i = max(y1_i + 1, min(height, int(round(y2))))
    return x1_i, y1_i, x2_i, y2_i


def expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    pad_fraction: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * pad_fraction
    pad_y = box_h * pad_fraction
    return _clamp_box(
        x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width, height
    )


def select_best_box_xyxy(boxes) -> tuple[float, float, float, float] | None:
    """Return highest-confidence ``xyxy`` box, or ``None`` if empty."""
    if boxes is None or len(boxes) == 0:
        return None
    idx = int(boxes.conf.argmax())
    xyxy = boxes[idx].xyxy
    if hasattr(xyxy, "__len__") and len(xyxy) == 4 and all(
        isinstance(v, (int, float)) for v in xyxy
    ):
        vals = xyxy
    else:
        row = xyxy[0]
        vals = row.tolist() if hasattr(row, "tolist") else list(row)
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


@dataclass
class KepalaCropper:
    """Lazy-loaded YOLO head detector; crops PIL images to the best kepala box."""

    weights: Path = DEFAULT_KEPALA_WEIGHTS
    conf: float = 0.25
    pad_fraction: float = 0.05
    min_area_fraction: float = 0.10
    imgsz: int = 640
    _model: object | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO kepala weights not found: {self.weights}")
        from ultralytics import YOLO

        self._model = YOLO(str(self.weights))
        return self._model

    def detect_xyxy(self, pil_im: Image.Image) -> tuple[float, float, float, float] | None:
        model = self._load_model()
        arr = np.asarray(pil_im.convert("RGB"))
        results = model.predict(
            arr,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        return select_best_box_xyxy(results[0].boxes)

    def crop(self, pil_im: Image.Image) -> Image.Image:
        """Crop to best kepala detection; return original if none or box too small."""
        box = self.detect_xyxy(pil_im)
        if box is None:
            return pil_im
        width, height = pil_im.size
        if box_area_fraction(*box, width, height) < self.min_area_fraction:
            return pil_im
        x1, y1, x2, y2 = expand_box(
            *box, pad_fraction=self.pad_fraction, width=width, height=height
        )
        return pil_im.crop((x1, y1, x2, y2))


def apply_kepala_crop(
    pil_im: Image.Image,
    cropper: KepalaCropper | None,
) -> Image.Image:
    if cropper is None:
        return pil_im
    return cropper.crop(pil_im)


def load_rgb_pil(
    path: Path,
    *,
    flip: bool,
    cropper: KepalaCropper | None = None,
) -> Image.Image:
    """Open RGB image, optionally mirror, then optionally YOLO kepala crop."""
    from PIL import ImageOps

    pil_im = Image.open(path).convert("RGB")
    if flip:
        pil_im = ImageOps.mirror(pil_im)
    return apply_kepala_crop(pil_im, cropper)
