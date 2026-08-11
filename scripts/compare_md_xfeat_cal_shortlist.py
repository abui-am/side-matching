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
)
from sides_matching.evaluation import (  # noqa: E402
    filter_bilateral_df,
    identities_with_both_sides,
    split_calibration_one_per_side,
)
from sides_matching.predictions import _calibrate_matrix  # noqa: E402
from sides_matching.xfeat_matching import (  # noqa: E402
    XFeatMatcher,
    load_or_compute_xfeat_shortlist_similarity,
)


DATASETS = {
    "Amvrakikos": ("AmvrakikosTurtles", amvrakikos),
    "ReunionGreen": ("ReunionTurtles", reunion_green),
    "ReunionHawksbill": ("ReunionTurtles", reunion_hawksbill),
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
    p.add_argument("--square-size", type=int, default=512)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/results/reunion_md_xfeat_cal_shortlist_N10.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    matcher = XFeatMatcher(device=args.device, max_size=None, square_size=args.square_size)
    print(
        f"device={args.device} N={args.shortlist_n} XFeat={matcher.resize_tag} "
        f"split={SPLIT_MODE} seed={args.seed}"
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

        df, keep_idx = filter_bilateral_df(df_full)
        n = len(df)
        paths_full = df_full["path"].tolist()
        print(
            f"\n=== {name}: n={n_full}->{n} identities={len(bilateral)} "
            f"(excluded {n_excluded_identities}) flip={flip} N={args.shortlist_n} ==="
        )

        md_q = args.features_dir / f"MegaDescriptor_{name}_flip={flip}_grayscale=False.pickle"
        md_db = args.features_dir / f"MegaDescriptor_{name}_flip=False_grayscale=False.pickle"
        md_sim_full = MegaDescriptor(str(md_q), str(md_db)).compute_similarity(
            ignore="diagonal"
        )
        md_sim = md_sim_full[np.ix_(keep_idx, keep_idx)]

        xfeat_sim_full = load_or_compute_xfeat_shortlist_similarity(
            name=name,
            data_root=root,
            paths=paths_full,
            flip_query=flip,
            md_sim=md_sim_full,
            top_n=args.shortlist_n,
            cache_dir=args.cache_dir,
            matcher=matcher,
        )
        xfeat_sim = xfeat_sim_full[np.ix_(keep_idx, keep_idx)]

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
            md_sim, cal_md, cal_xfeat, args.shortlist_n
        )
        naive_casc = cascade_similarity(md_sim, xfeat_sim, args.shortlist_n)

        methods = {
            "MegaDescriptor": md_sim,
            f"MD→XFeat N={args.shortlist_n}": naive_casc,
            f"A CalShortlist N={args.shortlist_n}": cal_short,
        }
        for method, sim in methods.items():
            pred = Prediction(df, sim, k=n - 1, query_indices=test_idx)
            pred.compute_accuracy(MODS)
            row = {
                "dataset": name,
                "method": method,
                "flip": flip,
                "shortlist_n": args.shortlist_n,
                "resize": matcher.resize_tag,
                "split_mode": SPLIT_MODE,
                "seed": args.seed,
                "n_identities": len(bilateral),
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
