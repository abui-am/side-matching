#!/usr/bin/env python3
"""Compare MD→XFeat N shortlist with square resize vs native (no resize).

Default: ReunionGreen, N=10, 512×512 vs no resize.
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
    get_transform,
    reunion_green,
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
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/xfeat_resize_compare"))
    p.add_argument("--datasets", nargs="+", default=["ReunionGreen"], choices=sorted(DATASETS))
    p.add_argument("--device", default=default_device())
    p.add_argument("--shortlist-n", type=int, default=10)
    p.add_argument("--square-size", type=int, default=512)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/results/reunion_md_xfeat_resize_N10.csv",
    )
    return p.parse_args()


def evaluate(
    *,
    name: str,
    data_root: Path,
    features_dir: Path,
    cache_dir: Path,
    matcher: XFeatMatcher,
    shortlist_n: int,
    img_size: int,
    label: str,
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
    print(f"\n=== {name} {label}: n={n} N={shortlist_n} resize={matcher.resize_tag} ===")

    rows: list[dict] = []
    for flip in (False, True):
        md_q = features_dir / f"MegaDescriptor_{name}_flip={flip}_grayscale=False.pickle"
        md_db = features_dir / f"MegaDescriptor_{name}_flip=False_grayscale=False.pickle"
        md_sim = MegaDescriptor(str(md_q), str(md_db)).compute_similarity(ignore="diagonal")

        xfeat_sim = load_or_compute_xfeat_shortlist_similarity(
            name=name,
            data_root=root,
            paths=paths,
            flip_query=flip,
            md_sim=md_sim,
            top_n=shortlist_n,
            cache_dir=cache_dir,
            matcher=matcher,
        )
        methods = {
            "MegaDescriptor": Prediction(df, md_sim, k=n - 1),
            f"MD→XFeat N={shortlist_n} ({label})": Prediction(
                df, cascade_similarity(md_sim, xfeat_sim, shortlist_n), k=n - 1
            ),
        }
        for method, pred in methods.items():
            pred.compute_accuracy(MODS)
            row = {
                "dataset": name,
                "method": method,
                "resize": matcher.resize_tag,
                "label": label,
                "flip": flip,
                "shortlist_n": shortlist_n,
                "n_queries": int(len(pred.true)),
                "full_top1": float(pred.accuracy["full"]["top 1"]),
                "full_top5": float(pred.accuracy["full"]["top 5"]),
                "opp_top1": float(pred.accuracy["different orientation"]["top 1"]),
            }
            rows.append(row)
            print(
                f"  flip={flip} {method:36s} "
                f"top1={row['full_top1']:.3f} top5={row['full_top5']:.3f} "
                f"opp1={row['opp_top1']:.3f}"
            )
    return rows


def main() -> None:
    args = parse_args()
    print(
        f"device={args.device} N={args.shortlist_n} "
        f"square={args.square_size} vs native datasets={args.datasets}"
    )

    configs = [
        (
            f"{args.square_size}x{args.square_size}",
            XFeatMatcher(
                device=args.device, max_size=None, square_size=args.square_size
            ),
        ),
        ("no_resize", XFeatMatcher(device=args.device, max_size=None, square_size=None)),
    ]

    rows: list[dict] = []
    for label, matcher in configs:
        t0 = time.time()
        print(f"\nXFeat {label} ({matcher.resize_tag}) ready in {time.time() - t0:.1f}s")
        for name in args.datasets:
            rows.extend(
                evaluate(
                    name=name,
                    data_root=args.data_dir,
                    features_dir=args.features_dir,
                    cache_dir=args.cache_dir,
                    matcher=matcher,
                    shortlist_n=args.shortlist_n,
                    img_size=args.img_size,
                    label=label,
                )
            )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    args.out.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2))

    print(f"\n=== SUMMARY (flip=True, N={args.shortlist_n}) ===")
    sub = out[(out["flip"] == True) & out["method"].str.startswith("MD→XFeat")][
        ["dataset", "label", "resize", "full_top1", "full_top5", "opp_top1"]
    ]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
