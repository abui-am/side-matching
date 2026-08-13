"""YOLO head (kepala) detection and crop preprocessing.

Uses weights from ``yolo_kepala/`` (Ultralytics YOLO11, single class ``kepala``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CropAction = Literal["cropped", "skipped_small", "no_detection"]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEPALA_WEIGHTS = REPO_ROOT / "yolo_kepala" / "kepala.pt"


def preprocess_tag(
    *,
    use_kepala: bool,
    min_area_fraction: float = 0.0,
    pad_fraction: float = 0.0,
) -> str:
    if not use_kepala:
        return "full"
    parts = ["kepala"]
    if pad_fraction > 0:
        parts.append(f"pad{int(round(pad_fraction * 100))}")
    if min_area_fraction > 0:
        parts.append(f"min{int(round(min_area_fraction * 1000)):03d}")
    return "_".join(parts)


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


def select_best_box_xyxy(
    boxes,
) -> tuple[tuple[float, float, float, float], float] | None:
    """Return (xyxy, conf) for the highest-confidence box, or ``None`` if empty."""
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
    conf = boxes.conf
    conf_val = float(conf[idx] if hasattr(conf, "__getitem__") else conf)
    return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])), conf_val


@dataclass(frozen=True)
class CropDecision:
    action: CropAction
    box_xyxy: tuple[float, float, float, float] | None
    crop_xyxy: tuple[int, int, int, int] | None
    conf: float | None
    area_fraction: float | None


@dataclass
class KepalaCropper:
    """Lazy-loaded YOLO head detector; crops PIL images to the best kepala box."""

    weights: Path = DEFAULT_KEPALA_WEIGHTS
    conf: float = 0.25
    pad_fraction: float = 0.50
    min_area_fraction: float = 0.005
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

    def detect_xyxy(
        self, pil_im: Image.Image
    ) -> tuple[tuple[float, float, float, float], float] | None:
        model = self._load_model()
        arr = np.asarray(pil_im.convert("RGB"))
        results = model.predict(
            arr,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        return select_best_box_xyxy(results[0].boxes)

    def decide(self, pil_im: Image.Image) -> CropDecision:
        """Return crop action without mutating the image."""
        detected = self.detect_xyxy(pil_im)
        if detected is None:
            return CropDecision("no_detection", None, None, None, None)
        box, conf = detected
        width, height = pil_im.size
        area = box_area_fraction(*box, width, height)
        crop_xyxy = expand_box(
            *box, pad_fraction=self.pad_fraction, width=width, height=height
        )
        if area < self.min_area_fraction:
            return CropDecision("skipped_small", box, crop_xyxy, conf, area)
        return CropDecision("cropped", box, crop_xyxy, conf, area)

    def crop(self, pil_im: Image.Image) -> Image.Image:
        """Crop to best kepala detection; return original if none or box too small."""
        decision = self.decide(pil_im)
        if decision.action != "cropped" or decision.crop_xyxy is None:
            return pil_im
        return pil_im.crop(decision.crop_xyxy)


CROP_OVERLAY_COLORS: dict[CropAction, tuple[int, int, int]] = {
    "cropped": (0, 200, 70),
    "skipped_small": (230, 140, 0),
    "no_detection": (220, 40, 40),
}


def draw_crop_overlay(pil_im: Image.Image, decision: CropDecision) -> Image.Image:
    """Copy of ``pil_im`` with the detection/crop box and action label."""
    overlay = pil_im.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    color = CROP_OVERLAY_COLORS[decision.action]
    width, height = overlay.size
    stroke = max(3, min(width, height) // 200)
    if decision.box_xyxy is not None:
        x1, y1, x2, y2 = decision.box_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)
    if decision.crop_xyxy is not None and decision.action == "cropped":
        draw.rectangle(list(decision.crop_xyxy), outline=(255, 255, 255), width=max(1, stroke - 1))
    conf_txt = f"{decision.conf:.2f}" if decision.conf is not None else "-"
    area_txt = f"{100 * decision.area_fraction:.1f}%" if decision.area_fraction is not None else "-"
    label = f"{decision.action}  conf={conf_txt}  area={area_txt}"
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.rectangle([8, 8, 8 + 12 * len(label), 36], fill=(0, 0, 0))
    draw.text((12, 12), label, fill=color, font=font)
    return overlay


def apply_kepala_crop(
    pil_im: Image.Image,
    cropper: KepalaCropper | None,
) -> Image.Image:
    if cropper is None:
        return pil_im
    return cropper.crop(pil_im)


def parse_bbox_xywh(bbox) -> tuple[int, int, int, int] | None:
    """Parse dataset ``[x, y, w, h]`` pixel box; return ``None`` if missing."""
    if bbox is None:
        return None
    if isinstance(bbox, float) and np.isnan(bbox):
        return None
    try:
        vals = [float(v) for v in bbox]
    except TypeError:
        return None
    if len(vals) != 4:
        return None
    x, y, w, h = vals
    if w <= 0 or h <= 0:
        return None
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def apply_bbox_xywh(pil_im: Image.Image, bbox) -> Image.Image:
    """Crop to dataset xywh box (Wildlife ``img_load='bbox'``)."""
    parsed = parse_bbox_xywh(bbox)
    if parsed is None:
        return pil_im
    x, y, w, h = parsed
    return pil_im.crop((x, y, x + w, y + h))


def load_rgb_pil(
    path: Path,
    *,
    flip: bool,
    cropper: KepalaCropper | None = None,
    bbox=None,
) -> Image.Image:
    """Open RGB image: bbox crop (optional) → flip (optional) → YOLO crop (optional)."""
    from PIL import ImageOps

    pil_im = Image.open(path).convert("RGB")
    pil_im = apply_bbox_xywh(pil_im, bbox)
    if flip:
        pil_im = ImageOps.mirror(pil_im)
    return apply_kepala_crop(pil_im, cropper)
