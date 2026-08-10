#!/usr/bin/env python3
"""Benchmark smart MD→ALIKED rerank variants A (calibrated shortlist) and B (gate).

Uses the same feature/cache layout as compare_base_models.py. Accuracy is reported
on the held-out query half after fitting calibrators (seed=0, val_fraction=0.5),
so methods are comparable to FusionCalibrated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_base_models import (  # noqa: E402
    DATASETS,
    MODS,
    cascade_similarity,
    default_device,
    load_or_compute_aliked,
)
from sides_matching import Combined, MegaDescriptor, Prediction, get_transform  # noqa: E402
from sides_matching.predictions import _calibrate_matrix  # noqa: E402


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
    """Return a full ranking score row: short by short_scores, then MD tail."""
    scores = np.where(np.isfinite(short_scores), short_scores, -np.inf)
    rerank = short[np.lexsort((-md_sim[i, short], -scores))]
    ranked = np.concatenate([rerank, rest])
    out = np.full(md_sim.shape[1], -np.inf, dtype=np.float64)
    out[ranked] = np.arange(len(ranked), 0, -1, dtype=np.float64)
    return out


def calibrated_shortlist_fusion(
    md_sim: np.ndarray,
    cal_md: np.ndarray,
    cal_al: np.ndarray,
    top_n: int,
) -> np.ndarray:
    """A: MD shortlist, rerank by mean of calibrated MD and ALIKED."""
    out = np.full_like(md_sim, -np.inf, dtype=np.float64)
    for i in range(md_sim.shape[0]):
        short, rest = shortlist_order(md_sim[i], top_n)
        fused = 0.5 * (cal_md[i, short] + cal_al[i, short])
        out[i] = rank_from_shortlist(md_sim, fused, short, rest, i)
    return out


def confidence_gated_rerank(
    md_sim: np.ndarray,
    cal_md: np.ndarray,
    cal_al: np.ndarray,
    top_n: int,
    tau: float,
) -> np.ndarray:
    """B: if max cal-ALIKED on shortlist >= tau, rerank by cal-ALIKED; else MD."""
    out = np.full_like(md_sim, -np.inf, dtype=np.float64)
    for i in range(md_sim.shape[0]):
        short, rest = shortlist_order(md_sim[i], top_n)
        al = cal_al[i, short]
        al_f = np.where(np.isfinite(al), al, -np.inf)
        if al_f.max() >= tau:
            out[i] = rank_from_shortlist(md_sim, al_f, short, rest, i)
        else:
            out[i] = rank_from_shortlist(md_sim, md_sim[i, short], short, rest, i)
    return out


def confidence_gated_pairwise(
    md_sim: np.ndarray,
    cal_md: np.ndarray,
    cal_al: np.ndarray,
    top_n: int,
    tau: float,
) -> np.ndarray:
    """B-pair: within shortlist, use cal-ALIKED if >= tau else cal-MD."""
    out = np.full_like(md_sim, -np.inf, dtype=np.float64)
    for i in range(md_sim.shape[0]):
        short, rest = shortlist_order(md_sim[i], top_n)
        md = cal_md[i, short]
        al = cal_al[i, short]
        md_f = np.where(np.isfinite(md), md, -np.inf)
        al_f = np.where(np.isfinite(al), al, -np.inf)
        scores = np.where(al_f >= tau, al_f, md_f)
        out[i] = rank_from_shortlist(md_sim, scores, short, rest, i)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument(
        "--features-dir",
        type=Path,
        default=Path("/Users/abui/@code/tes/model/features"),
    )
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/base_model_sims"))
    p.add_argument(
        "--reunion-aliked-cache",
        type=Path,
        default=Path("/tmp/reunion_green_sims"),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["Zakynthos", "ReunionGreen"],
        choices=sorted(DATASETS),
    )
    p.add_argument("--device", default=default_device())
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--shortlist-n", type=int, default=30)
    p.add_argument("--val-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-top1", type=float, default=0.80)
    p.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8],
        help="Confidence thresholds for method B",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/smart_rerank_benchmark.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reuse = args.reunion_aliked_cache if args.reunion_aliked_cache.is_dir() else None
    rows: list[dict] = []

    for name in args.datasets:
        subdir, loader = DATASETS[name]
        root = args.data_dir / subdir
        df = loader(
            str(root),
            transform=get_transform(
                flip=False, grayscale=False, img_size=args.img_size, normalize=False
            ),
        ).df
        n = len(df)
        print(f"\n=== {name}: n={n} shortlist={args.shortlist_n} ===")

        for flip in (False, True):
            md_q = (
                args.features_dir
                / f"MegaDescriptor_{name}_flip={flip}_grayscale=False.pickle"
            )
            md_db = (
                args.features_dir
                / f"MegaDescriptor_{name}_flip=False_grayscale=False.pickle"
            )
            al_q = (
                args.features_dir
                / f"Aliked_{name}_flip={flip}_grayscale=False_{args.img_size}.pickle"
            )
            al_db = (
                args.features_dir
                / f"Aliked_{name}_flip=False_grayscale=False_{args.img_size}.pickle"
            )

            md_sim = MegaDescriptor(str(md_q), str(md_db)).compute_similarity(
                ignore="diagonal"
            )
            al_sim = load_or_compute_aliked(
                name=name,
                flip=flip,
                path_query=str(al_q),
                path_database=str(al_db),
                cache_dir=args.cache_dir,
                device=args.device,
                reuse_reunion_cache=reuse,
            )

            md_pred = Prediction(df, md_sim, k=n - 1)
            al_pred = Prediction(df, al_sim, k=n - 1)
            fused = Combined(
                md_pred,
                al_pred,
                method="calibrated",
                val_fraction=args.val_fraction,
                seed=args.seed,
            )
            assert fused.calibrators is not None
            cal_md = _calibrate_matrix(fused.calibrators[0], md_sim)
            cal_al = _calibrate_matrix(fused.calibrators[1], al_sim)
            fus_sim = fused.compute_similarity(ignore="diagonal")
            test_idx = fused.test_indices

            methods: dict[str, np.ndarray] = {
                "MegaDescriptor": md_sim,
                "FusionCalibrated": fus_sim,
                f"Naive MD→ALIKED N={args.shortlist_n}": cascade_similarity(
                    md_sim, al_sim, args.shortlist_n
                ),
                f"A CalShortlist N={args.shortlist_n}": calibrated_shortlist_fusion(
                    md_sim, cal_md, cal_al, args.shortlist_n
                ),
            }
            for tau in args.taus:
                methods[f"B Gate τ={tau:g} N={args.shortlist_n}"] = (
                    confidence_gated_rerank(
                        md_sim, cal_md, cal_al, args.shortlist_n, tau
                    )
                )
                methods[f"B Pair τ={tau:g} N={args.shortlist_n}"] = (
                    confidence_gated_pairwise(
                        md_sim, cal_md, cal_al, args.shortlist_n, tau
                    )
                )

            for method, sim in methods.items():
                pred = Prediction(df, sim, k=n - 1, query_indices=test_idx)
                pred.compute_accuracy(MODS)
                row = {
                    "dataset": name,
                    "method": method,
                    "flip": flip,
                    "n_queries": int(len(pred.true)),
                    "full_top1": float(pred.accuracy["full"]["top 1"]),
                    "full_top5": float(pred.accuracy["full"]["top 5"]),
                    "opp_top1": float(
                        pred.accuracy["different orientation"]["top 1"]
                    ),
                    "gap_to_target_top1": float(
                        args.target_top1 - pred.accuracy["full"]["top 1"]
                    ),
                }
                rows.append(row)
                print(
                    f"  flip={flip} {method:32s} "
                    f"top1={row['full_top1']:.3f} top5={row['full_top5']:.3f} "
                    f"opp1={row['opp_top1']:.3f}"
                )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\n=== SUMMARY flip=True (held-out n_queries, vs {args.target_top1:.0%}) ===")
    sub = out[out["flip"] == True][
        ["dataset", "method", "n_queries", "full_top1", "full_top5", "opp_top1"]
    ]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
