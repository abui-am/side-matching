#!/usr/bin/env python3
"""Compare MD→XFeat vs MD→LoMa at square ~512 resize (flip=True, N=10).

LoMa snaps 512→504 for DINOv2 patch alignment; XFeat uses exact 512×512.
Default dataset: ReunionGreen.
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
from sides_matching import MegaDescriptor, Prediction, get_transform, reunion_green  # noqa: E402
from sides_matching.loma_matching import (  # noqa: E402
    LomaMatcher,
    _square_dim,
    load_or_compute_loma_shortlist_similarity,
)
from sides_matching.xfeat_matching import (  # noqa: E402
    XFeatMatcher,
    load_or_compute_xfeat_shortlist_similarity,
)

DATASETS = {
    "ReunionGreen": ("ReunionTurtles", reunion_green),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument(
        "--features-dir",
        type=Path,
        default=Path("/Users/abui/@code/tes/model/features"),
    )
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/xfeat_loma_sq512"))
    p.add_argument("--datasets", nargs="+", default=["ReunionGreen"], choices=sorted(DATASETS))
    p.add_argument("--device", default=default_device())
    p.add_argument("--shortlist-n", type=int, default=10)
    p.add_argument("--square-size", type=int, default=512)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/results/reunion_md_xfeat_vs_loma_sq512_N10.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loma_side = _square_dim(args.square_size)
    print(
        f"device={args.device} N={args.shortlist_n} flip=True "
        f"XFeat={args.square_size}x{args.square_size} "
        f"LoMa={loma_side}x{loma_side} (patch-aligned from {args.square_size})"
    )

    t0 = time.time()
    xfeat = XFeatMatcher(
        device=args.device, max_size=None, square_size=args.square_size
    )
    print(f"XFeat {xfeat.resize_tag} loaded in {time.time() - t0:.1f}s")
    t0 = time.time()
    loma = LomaMatcher(resize=None, square_size=args.square_size)
    print(f"LoMa {loma.resize_tag} loaded in {time.time() - t0:.1f}s")

    rows: list[dict] = []
    flip = True
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
        paths = df["path"].tolist()
        print(f"\n=== {name}: n={n} N={args.shortlist_n} flip={flip} ===")

        md_q = args.features_dir / f"MegaDescriptor_{name}_flip={flip}_grayscale=False.pickle"
        md_db = args.features_dir / f"MegaDescriptor_{name}_flip=False_grayscale=False.pickle"
        md_sim = MegaDescriptor(str(md_q), str(md_db)).compute_similarity(ignore="diagonal")

        xfeat_sim = load_or_compute_xfeat_shortlist_similarity(
            name=name,
            data_root=root,
            paths=paths,
            flip_query=flip,
            md_sim=md_sim,
            top_n=args.shortlist_n,
            cache_dir=args.cache_dir,
            matcher=xfeat,
        )
        loma_sim = load_or_compute_loma_shortlist_similarity(
            name=name,
            data_root=root,
            paths=paths,
            flip_query=flip,
            md_sim=md_sim,
            top_n=args.shortlist_n,
            cache_dir=args.cache_dir,
            matcher=loma,
        )

        methods = {
            "MegaDescriptor": Prediction(df, md_sim, k=n - 1),
            f"MD→XFeat N={args.shortlist_n} sq{args.square_size}": Prediction(
                df, cascade_similarity(md_sim, xfeat_sim, args.shortlist_n), k=n - 1
            ),
            f"MD→LoMa N={args.shortlist_n} sq{args.square_size}": Prediction(
                df, cascade_similarity(md_sim, loma_sim, args.shortlist_n), k=n - 1
            ),
        }
        for method, pred in methods.items():
            pred.compute_accuracy(MODS)
            row = {
                "dataset": name,
                "method": method,
                "flip": flip,
                "shortlist_n": args.shortlist_n,
                "square_size": args.square_size,
                "full_top1": float(pred.accuracy["full"]["top 1"]),
                "full_top5": float(pred.accuracy["full"]["top 5"]),
                "opp_top1": float(pred.accuracy["different orientation"]["top 1"]),
            }
            rows.append(row)
            print(
                f"  {method:40s} "
                f"top1={row['full_top1']:.3f} top5={row['full_top5']:.3f} "
                f"opp1={row['opp_top1']:.3f}"
            )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    args.out.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\n=== SUMMARY ===")
    print(
        out[["dataset", "method", "full_top1", "full_top5", "opp_top1"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
