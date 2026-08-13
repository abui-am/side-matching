#!/usr/bin/env python3
"""Draw YOLO kepala crop boxes on dataset images and write a CSV log.

Green = crop applied, orange = detected but skipped (too small), red = no detection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sides_matching import (  # noqa: E402
    amvrakikos,
    reunion_green,
    reunion_hawksbill,
    zakynthos,
)
from sides_matching.loma_matching import resolve_image_paths  # noqa: E402
from sides_matching.yolo_kepala_preprocessing import (  # noqa: E402
    DEFAULT_KEPALA_WEIGHTS,
    KepalaCropper,
    draw_crop_overlay,
    load_rgb_pil,
)

DATASETS = {
    "Amvrakikos": ("AmvrakikosTurtles", amvrakikos),
    "ReunionGreen": ("ReunionTurtles", reunion_green),
    "ReunionHawksbill": ("ReunionTurtles", reunion_hawksbill),
    "Zakynthos": ("ZakynthosTurtles", zakynthos),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--datasets", nargs="+", default=["Zakynthos"], choices=sorted(DATASETS))
    p.add_argument("--yolo-weights", type=Path, default=DEFAULT_KEPALA_WEIGHTS)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-pad", type=float, default=0.50)
    p.add_argument("--yolo-min-area", type=float, default=0.005)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "docs/figures/kepala_crops",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cropper = KepalaCropper(
        weights=args.yolo_weights,
        conf=args.yolo_conf,
        pad_fraction=args.yolo_pad,
        min_area_fraction=args.yolo_min_area,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for name in args.datasets:
        subdir, loader = DATASETS[name]
        root = args.data_dir / subdir
        df = loader(str(root)).df.reset_index(drop=True)
        if args.max_images is not None:
            df = df.head(args.max_images)
        overlay_dir = args.out_dir / name / "overlay"
        crop_dir = args.out_dir / name / "crop"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)
        paths = resolve_image_paths(df["path"].tolist(), root)

        for i, path in enumerate(tqdm(paths, desc=f"kepala overlay {name}")):
            pil = load_rgb_pil(path, flip=False, cropper=None)
            decision = cropper.decide(pil)
            overlay = draw_crop_overlay(pil, decision)
            stem = f"{i:04d}_{path.stem}"
            overlay.save(overlay_dir / f"{stem}.jpg", quality=85)
            if decision.action == "cropped" and decision.crop_xyxy is not None:
                pil.crop(decision.crop_xyxy).save(crop_dir / f"{stem}.jpg", quality=85)

            box = decision.box_xyxy
            all_rows.append(
                {
                    "dataset": name,
                    "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                    "identity": df.iloc[i]["identity"] if "identity" in df.columns else "",
                    "action": decision.action,
                    "conf": decision.conf,
                    "area_fraction": decision.area_fraction,
                    "x1": None if box is None else box[0],
                    "y1": None if box is None else box[1],
                    "x2": None if box is None else box[2],
                    "y2": None if box is None else box[3],
                    "overlay": str((overlay_dir / f"{stem}.jpg").relative_to(args.out_dir)),
                }
            )

        counts = pd.Series([r["action"] for r in all_rows if r["dataset"] == name]).value_counts()
        print(f"\n{name} n={len(df)}  {counts.to_dict()}")

    csv_path = args.out_dir / "kepala_crop_log.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
