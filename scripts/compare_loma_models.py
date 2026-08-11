#!/usr/bin/env python3
"""Compare MegaDescriptor, LoMa-B, calibrated fusion, cascade, and A CalShortlist.

See docs/base-model-comparison.md for protocol. ALIKED results remain in the
historical appendix; this script uses LoMa as the local matcher.

Example:
  .venv/bin/python scripts/compare_loma_models.py \\
    --features-dir /Users/abui/@code/tes/model/features \\
    --data-dir data \\
    --cache-dir /tmp/base_model_sims \\
    --out docs/results/base_model_comparison_loma.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_base_models import (  # noqa: E402
    DATASETS,
    MODS,
    cascade_similarity,
    default_device,
)
from sides_matching import (  # noqa: E402
    Combined,
    MegaDescriptor,
    Prediction,
    get_transform,
    reunion_green,
    zakynthos,
)
from sides_matching.loma_matching import LomaMatcher, load_or_compute_loma_similarity  # noqa: E402
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


def evaluate_dataset(
    *,
    name: str,
    data_root: Path,
    features_dir: Path,
    cache_dir: Path,
    matcher: LomaMatcher,
    grayscale: bool,
    img_size: int,
    shortlist_n: int,
    val_fraction: float,
    seed: int,
    target_top1: float,
) -> tuple[list[dict], dict[str, list[float]]]:
    subdir, loader = DATASETS[name]
    root = data_root / subdir
    df = loader(
        str(root),
        transform=get_transform(
            flip=False, grayscale=grayscale, img_size=img_size, normalize=False
        ),
    ).df
    n = len(df)
    image_paths = df["path"].tolist()
    print(f"\n=== {name}: n={n} ids={df['identity'].nunique()} ===")

    rows: list[dict] = []
    curves: dict[str, list[float]] = {}

    for flip in (False, True):
        md_q = features_dir / f"MegaDescriptor_{name}_flip={flip}_grayscale={grayscale}.pickle"
        md_db = features_dir / f"MegaDescriptor_{name}_flip=False_grayscale={grayscale}.pickle"

        md_sim = MegaDescriptor(str(md_q), str(md_db)).compute_similarity(ignore="diagonal")
        loma_sim = load_or_compute_loma_similarity(
            name=name,
            data_root=root,
            paths=image_paths,
            flip_query=flip,
            cache_dir=cache_dir,
            matcher=matcher,
        )

        md_pred = Prediction(df, md_sim, k=n - 1)
        loma_pred = Prediction(df, loma_sim, k=n - 1)

        methods: dict[str, Prediction] = {
            "MegaDescriptor": md_pred,
            "LoMa": loma_pred,
        }

        fused = Combined(
            md_pred,
            loma_pred,
            method="calibrated",
            val_fraction=val_fraction,
            seed=seed,
        )
        fus_sim = fused.compute_similarity(ignore="diagonal")
        methods["FusionCalibrated"] = Prediction(
            df, fus_sim, k=n - 1, query_indices=fused.test_indices
        )

        casc = cascade_similarity(md_sim, loma_sim, shortlist_n)
        methods[f"MD→LoMa N={shortlist_n}"] = Prediction(df, casc, k=n - 1)

        assert fused.calibrators is not None
        cal_md = _calibrate_matrix(fused.calibrators[0], md_sim)
        cal_loma = _calibrate_matrix(fused.calibrators[1], loma_sim)
        cal_short = calibrated_shortlist_fusion(md_sim, cal_md, cal_loma, shortlist_n)
        methods[f"A CalShortlist N={shortlist_n}"] = Prediction(
            df, cal_short, k=n - 1, query_indices=fused.test_indices
        )

        for method, pred in methods.items():
            pred.compute_accuracy(MODS)
            row = {
                "dataset": name,
                "method": method,
                "flip": flip,
                "n_queries": int(len(pred.true)),
                "full_top1": float(pred.accuracy["full"]["top 1"]),
                "full_top5": float(pred.accuracy["full"]["top 5"]),
                "opp_top1": float(pred.accuracy["different orientation"]["top 1"]),
                "opp_top5": float(pred.accuracy["different orientation"]["top 5"]),
                "gap_to_target_top1": float(target_top1 - pred.accuracy["full"]["top 1"]),
            }
            rows.append(row)
            curves[f"{name}|{method}|flip={flip}|full"] = [
                float(pred.accuracy["full"][f"top {i}"]) for i in range(1, 11)
            ]
            print(
                f"  flip={flip} {method:26s} "
                f"top1={row['full_top1']:.3f} top5={row['full_top5']:.3f} "
                f"opp1={row['opp_top1']:.3f} gap={row['gap_to_target_top1']:+.3f}"
            )

    return rows, curves


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
        "--datasets",
        nargs="+",
        default=["Zakynthos", "ReunionGreen"],
        choices=sorted(DATASETS),
    )
    p.add_argument("--device", default=default_device())
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--grayscale", action="store_true", default=False)
    p.add_argument("--shortlist-n", type=int, default=30)
    p.add_argument("--val-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-top1", type=float, default=0.80)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/results/base_model_comparison_loma.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.features_dir.is_dir():
        raise SystemExit(f"features dir not found: {args.features_dir}")
    if not args.data_dir.is_dir():
        raise SystemExit(f"data dir not found: {args.data_dir}")

    print(f"device={args.device}")
    print(f"features={args.features_dir}")
    print(f"data={args.data_dir}")

    t0 = time.time()
    matcher = LomaMatcher()
    print(f"LoMa-B loaded in {time.time() - t0:.1f}s")

    all_rows: list[dict] = []
    all_curves: dict[str, list[float]] = {}

    for name in args.datasets:
        rows, curves = evaluate_dataset(
            name=name,
            data_root=args.data_dir,
            features_dir=args.features_dir,
            cache_dir=args.cache_dir,
            matcher=matcher,
            grayscale=args.grayscale,
            img_size=args.img_size,
            shortlist_n=args.shortlist_n,
            val_fraction=args.val_fraction,
            seed=args.seed,
            target_top1=args.target_top1,
        )
        all_rows.extend(rows)
        all_curves.update(curves)

    out = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps({"rows": all_rows, "curves": all_curves}, indent=2))

    print(f"\n=== SUMMARY (flip=True, full top-1 vs {args.target_top1:.0%}) ===")
    sub = out[out["flip"] == True][
        [
            "dataset",
            "method",
            "n_queries",
            "full_top1",
            "full_top5",
            "opp_top1",
            "gap_to_target_top1",
        ]
    ]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {args.out}")
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
