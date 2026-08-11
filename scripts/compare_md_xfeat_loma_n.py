#!/usr/bin/env python3
"""Compare MD→XFeat vs MD→LoMa shortlist (default N=10).

Datasets: Amvrakikos, ReunionGreen, ReunionHawksbill.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_base_models import MODS, cascade_similarity, default_device  # noqa: E402
from sides_matching import (  # noqa: E402
    MegaDescriptor,
    Prediction,
    amvrakikos,
    get_transform,
    reunion_green,
    reunion_hawksbill,
)
from sides_matching.loma_matching import LomaMatcher, load_or_compute_loma_shortlist_similarity  # noqa: E402
from sides_matching.xfeat_matching import (  # noqa: E402
    XFeatMatcher,
    load_or_compute_xfeat_shortlist_similarity,
)

DATASETS = {
    "Amvrakikos": ("AmvrakikosTurtles", amvrakikos),
    "ReunionGreen": ("ReunionTurtles", reunion_green),
    "ReunionHawksbill": ("ReunionTurtles", reunion_hawksbill),
}


def evaluate_dataset(
    *,
    name: str,
    data_root: Path,
    features_dir: Path,
    cache_dir: Path,
    xfeat: XFeatMatcher,
    loma: LomaMatcher,
    shortlist_n: int,
    img_size: int,
    target_top1: float,
) -> list[dict]:
    subdir, loader = DATASETS[name]
    root = data_root / subdir
    df = loader(
        str(root),
        transform=get_transform(
            flip=False, grayscale=False, img_size=img_size, normalize=False
        ),
    ).df
    n = len(df)
    paths = df["path"].tolist()
    print(f"\n=== {name}: n={n} ids={df['identity'].nunique()} N={shortlist_n} ===")

    rows: list[dict] = []
    for flip in (False, True):
        md_q = features_dir / f"MegaDescriptor_{name}_flip={flip}_grayscale=False.pickle"
        md_db = features_dir / f"MegaDescriptor_{name}_flip=False_grayscale=False.pickle"
        md_sim = MegaDescriptor(str(md_q), str(md_db)).compute_similarity(ignore="diagonal")

        md_pred = Prediction(df, md_sim, k=n - 1)
        md_pred.compute_accuracy(MODS)

        xfeat_sim = load_or_compute_xfeat_shortlist_similarity(
            name=name,
            data_root=root,
            paths=paths,
            flip_query=flip,
            md_sim=md_sim,
            top_n=shortlist_n,
            cache_dir=cache_dir,
            matcher=xfeat,
        )
        loma_sim = load_or_compute_loma_shortlist_similarity(
            name=name,
            data_root=root,
            paths=paths,
            flip_query=flip,
            md_sim=md_sim,
            top_n=shortlist_n,
            cache_dir=cache_dir,
            matcher=loma,
        )

        methods = {
            "MegaDescriptor": md_pred,
            f"MD→XFeat N={shortlist_n}": Prediction(
                df, cascade_similarity(md_sim, xfeat_sim, shortlist_n), k=n - 1
            ),
            f"MD→LoMa N={shortlist_n}": Prediction(
                df, cascade_similarity(md_sim, loma_sim, shortlist_n), k=n - 1
            ),
        }
        for method, pred in methods.items():
            if method != "MegaDescriptor":
                pred.compute_accuracy(MODS)
            row = {
                "dataset": name,
                "method": method,
                "flip": flip,
                "shortlist_n": shortlist_n,
                "n_queries": int(len(pred.true)),
                "full_top1": float(pred.accuracy["full"]["top 1"]),
                "full_top5": float(pred.accuracy["full"]["top 5"]),
                "opp_top1": float(pred.accuracy["different orientation"]["top 1"]),
                "gap_to_target_top1": float(target_top1 - pred.accuracy["full"]["top 1"]),
            }
            rows.append(row)
            print(
                f"  flip={flip} {method:22s} "
                f"top1={row['full_top1']:.3f} top5={row['full_top5']:.3f} "
                f"opp1={row['opp_top1']:.3f}"
            )
    return rows


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
        default=["Amvrakikos", "ReunionGreen", "ReunionHawksbill"],
        choices=sorted(DATASETS),
    )
    p.add_argument("--device", default=default_device())
    p.add_argument("--shortlist-n", type=int, default=10)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--target-top1", type=float, default=0.80)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/results/md_xfeat_vs_loma_N10.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"device={args.device} N={args.shortlist_n} datasets={args.datasets}")

    t0 = time.time()
    xfeat = XFeatMatcher(device=args.device)
    print(f"XFeat loaded in {time.time() - t0:.1f}s")
    t0 = time.time()
    loma = LomaMatcher()
    print(f"LoMa-B loaded in {time.time() - t0:.1f}s")

    rows: list[dict] = []
    for name in args.datasets:
        rows.extend(
            evaluate_dataset(
                name=name,
                data_root=args.data_dir,
                features_dir=args.features_dir,
                cache_dir=args.cache_dir,
                xfeat=xfeat,
                loma=loma,
                shortlist_n=args.shortlist_n,
                img_size=args.img_size,
                target_top1=args.target_top1,
            )
        )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    args.out.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2))

    print(f"\n=== SUMMARY (flip=True, N={args.shortlist_n}) ===")
    sub = out[out["flip"] == True][
        ["dataset", "method", "full_top1", "full_top5", "opp_top1", "gap_to_target_top1"]
    ]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
