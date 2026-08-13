#!/usr/bin/env python3
"""Compare MD→XFeat N shortlist vs A CalShortlist (isotonic+PCHIP fusion).

Default: ReunionGreen, flip=True, N=10, XFeat 512×512.
Calibration uses one left + one right photo per bilateral identity (seed=0);
remaining photos are the test split. Identities without both sides are excluded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_base_models import MODS, cascade_similarity, default_device  # noqa: E402
from sides_matching import (  # noqa: E402
    Combined,
    MegaDescriptor,
    Prediction,
    amvrakikos,
    get_transform,
    reunion_green,
    reunion_hawksbill,
    zakynthos,
)
from sides_matching.evaluation import (  # noqa: E402
    filter_bilateral_df,
    identities_with_both_sides,
    split_calibration_one_per_side,
    subset_identities_df,
)
from sides_matching.predictions import _calibrate_matrix  # noqa: E402
from sides_matching.megadescriptor_matching import (  # noqa: E402
    MegaDescriptorExtractor,
    load_or_compute_md_similarity,
)
from sides_matching.xfeat_matching import (  # noqa: E402
    XFeatMatcher,
    load_or_compute_xfeat_shortlist_similarity,
)
from sides_matching.yolo_kepala_preprocessing import (  # noqa: E402
    DEFAULT_KEPALA_WEIGHTS,
    KepalaCropper,
)


DATASETS = {
    "Amvrakikos": ("AmvrakikosTurtles", amvrakikos),
    "ReunionGreen": ("ReunionTurtles", reunion_green),
    "ReunionHawksbill": ("ReunionTurtles", reunion_hawksbill),
    "Zakynthos": ("ZakynthosTurtles", zakynthos),
}

SPLIT_MODE = "one_per_side"


def shortlist_order(md_row: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.flatnonzero(np.isfinite(md_row))
    order = finite[np.argsort(-md_row[finite])]
    return order[:top_n], order[top_n:]


def rank_from_shortlist(
    md_sim: np.ndarray,
    short_scores: np.ndarray,
    short: np.ndarray,
    rest: np.ndarray,
    i: int,
) -> np.ndarray:
    scores = np.where(np.isfinite(short_scores), short_scores, -np.inf)
    rerank = short[np.lexsort((-md_sim[i, short], -scores))]
    ranked = np.concatenate([rerank, rest])
    out = np.full(md_sim.shape[1], -np.inf, dtype=np.float64)
    out[ranked] = np.arange(len(ranked), 0, -1, dtype=np.float64)
    return out


def calibrated_shortlist_fusion(
    md_sim: np.ndarray,
    cal_md: np.ndarray,
    cal_local: np.ndarray,
    top_n: int,
) -> np.ndarray:
    out = np.full_like(md_sim, -np.inf, dtype=np.float64)
    for i in range(md_sim.shape[0]):
        short, rest = shortlist_order(md_sim[i], top_n)
        fused = 0.5 * (cal_md[i, short] + cal_local[i, short])
        out[i] = rank_from_shortlist(md_sim, fused, short, rest, i)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument(
        "--features-dir",
        type=Path,
        default=Path("/Users/abui/@code/tes/model/features"),
    )
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/xfeat_resize_compare"))
    p.add_argument("--datasets", nargs="+", default=["ReunionGreen"], choices=sorted(DATASETS))
    p.add_argument("--device", default=default_device())
    p.add_argument("--shortlist-n", type=int, default=10)
    p.add_argument(
        "--shortlist-fraction",
        type=float,
        default=None,
        help="If set, N = round(fraction × n_images) per dataset (overrides --shortlist-n)",
    )
    p.add_argument("--square-size", type=int, default=512)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-identities",
        type=int,
        default=None,
        help="If set, randomly sample this many bilateral identities (seeded)",
    )
    p.add_argument(
        "--yolo-kepala",
        action="store_true",
        help="Crop images to YOLO kepala (head) box before MD and XFeat extraction",
    )
    p.add_argument(
        "--yolo-weights",
        type=Path,
        default=DEFAULT_KEPALA_WEIGHTS,
        help="Ultralytics weights for kepala detector",
    )
    p.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence for kepala crop",
    )
    p.add_argument(
        "--yolo-pad",
        type=float,
        default=0.50,
        help="Fractional padding around detected kepala box (0.50 = 50 percent on each side)",
    )
    p.add_argument(
        "--yolo-min-area",
        type=float,
        default=0.005,
        help="Minimum detected box area (fraction of image) to apply crop",
    )
    p.add_argument(
        "--bbox-crop",
        action="store_true",
        help="Crop XFeat inputs to dataset bbox (same as Wildlife img_load=bbox); no-op if no bbox column",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/results/reunion_md_xfeat_cal_shortlist_N10.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cropper = None
    md_extractor = None
    if args.yolo_kepala:
        cropper = KepalaCropper(
            weights=args.yolo_weights,
            conf=args.yolo_conf,
            pad_fraction=args.yolo_pad,
            min_area_fraction=args.yolo_min_area,
        )
        md_extractor = MegaDescriptorExtractor(
            device=args.device,
            img_size=args.img_size,
            cropper=cropper,
        )
    matcher = XFeatMatcher(
        device=args.device,
        max_size=None,
        square_size=args.square_size,
        cropper=cropper,
        use_bbox=args.bbox_crop,
    )
    print(
        f"device={args.device} XFeat={matcher.resize_tag} "
        f"split={SPLIT_MODE} seed={args.seed}"
        + (" bbox_crop" if args.bbox_crop else "")
        + (f" yolo_kepala={args.yolo_weights.name}" if args.yolo_kepala else "")
        + (
            f" shortlist_fraction={args.shortlist_fraction}"
            if args.shortlist_fraction is not None
            else f" shortlist_n={args.shortlist_n}"
        )
        + (
            f" max_identities={args.max_identities}"
            if args.max_identities is not None
            else ""
        )
    )

    rows: list[dict] = []
    flip = True

    for name in args.datasets:
        subdir, loader = DATASETS[name]
        root = args.data_dir / subdir
        df_full = loader(
            str(root),
            transform=get_transform(
                flip=False, grayscale=False, img_size=args.img_size, normalize=False
            ),
        ).df
        n_full = len(df_full)
        n_identities_full = int(df_full["identity"].nunique())
        bilateral = identities_with_both_sides(df_full)
        n_excluded_identities = n_identities_full - len(bilateral)

        df, _ = filter_bilateral_df(df_full)
        if args.max_identities is not None:
            df = subset_identities_df(df, args.max_identities, seed=args.seed)
        n = len(df)
        n_identities = int(df["identity"].nunique())
        if args.shortlist_fraction is not None:
            if not 0.0 < args.shortlist_fraction <= 1.0:
                raise ValueError("shortlist-fraction must be in (0, 1]")
            shortlist_n = max(1, int(round(n * args.shortlist_fraction)))
        else:
            shortlist_n = args.shortlist_n
        cache_name = (
            f"{name}_id{args.max_identities}"
            if args.max_identities is not None
            else name
        )
        paths = df["path"].tolist()
        path_to_full = {p: i for i, p in enumerate(df_full["path"].tolist())}
        full_idx = np.array([path_to_full[p] for p in paths], dtype=np.int64)
        print(
            f"\n=== {name}: n={n_full}->{n} identities={n_identities} "
            f"(bilateral pool {len(bilateral)}, excluded {n_excluded_identities}) "
            f"flip={flip} N={shortlist_n} ==="
        )

        md_q = args.features_dir / f"MegaDescriptor_{name}_flip={flip}_grayscale=False.pickle"
        md_db = args.features_dir / f"MegaDescriptor_{name}_flip=False_grayscale=False.pickle"
        if args.yolo_kepala:
            assert md_extractor is not None
            md_sim = load_or_compute_md_similarity(
                name=cache_name,
                data_root=root,
                paths=paths,
                flip_query=flip,
                cache_dir=args.cache_dir,
                extractor=md_extractor,
            )
        else:
            md_sim_full = MegaDescriptor(str(md_q), str(md_db)).compute_similarity(
                ignore="diagonal"
            )
            md_sim = md_sim_full[np.ix_(full_idx, full_idx)]

        xfeat_sim = load_or_compute_xfeat_shortlist_similarity(
            name=cache_name,
            data_root=root,
            paths=paths,
            flip_query=flip,
            md_sim=md_sim,
            top_n=shortlist_n,
            cache_dir=args.cache_dir,
            matcher=matcher,
            bboxes=df["bbox"].tolist() if args.bbox_crop and "bbox" in df.columns else None,
        )

        val_idx, test_idx = split_calibration_one_per_side(df, seed=args.seed)
        print(
            f"  calibration queries={len(val_idx)} test queries={len(test_idx)} "
            f"(one left + one right per identity)"
        )

        md_pred = Prediction(df, md_sim, k=n - 1)
        xfeat_pred = Prediction(df, xfeat_sim, k=n - 1)
        fused = Combined(
            md_pred,
            xfeat_pred,
            method="calibrated",
            val_indices=val_idx,
        )
        assert fused.calibrators is not None
        np.testing.assert_array_equal(fused.test_indices, test_idx)

        cal_md = _calibrate_matrix(fused.calibrators[0], md_sim)
        cal_xfeat = _calibrate_matrix(fused.calibrators[1], xfeat_sim)
        cal_short = calibrated_shortlist_fusion(
            md_sim, cal_md, cal_xfeat, shortlist_n
        )
        naive_casc = cascade_similarity(md_sim, xfeat_sim, shortlist_n)

        methods = {
            "MegaDescriptor": md_sim,
            f"MD→XFeat N={shortlist_n}": naive_casc,
            f"A CalShortlist N={shortlist_n}": cal_short,
        }
        for method, sim in methods.items():
            pred = Prediction(df, sim, k=n - 1, query_indices=test_idx)
            pred.compute_accuracy(MODS)
            row = {
                "dataset": name,
                "method": method,
                "flip": flip,
                "shortlist_n": shortlist_n,
                "shortlist_fraction": args.shortlist_fraction,
                "resize": matcher.resize_tag,
                "yolo_kepala": args.yolo_kepala,
                "bbox_crop": args.bbox_crop,
                "split_mode": SPLIT_MODE,
                "seed": args.seed,
                "max_identities": args.max_identities,
                "n_identities": n_identities,
                "n_excluded_identities": n_excluded_identities,
                "n_images": n,
                "n_queries": int(len(pred.true)),
                "full_top1": float(pred.accuracy["full"]["top 1"]),
                "full_top5": float(pred.accuracy["full"]["top 5"]),
                "opp_top1": float(pred.accuracy["different orientation"]["top 1"]),
            }
            rows.append(row)
            print(
                f"  {method:32s} "
                f"top1={row['full_top1']:.3f} top5={row['full_top5']:.3f} "
                f"opp1={row['opp_top1']:.3f} n={row['n_queries']}"
            )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    args.out.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2))

    print(f"\n=== SUMMARY (flip=True, test split n={rows[-1]['n_queries']}) ===")
    sub = out[out["method"].str.contains("XFeat|CalShortlist")][
        ["method", "full_top1", "full_top5", "opp_top1"]
    ]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
