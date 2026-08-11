#!/usr/bin/env python3
"""MD→local shortlist on ReunionGreen: ALIKED / XFeat / DeDoDe-v2+LightGlue.

Example:
  .venv/bin/python scripts/compare_md_local_shortlist.py \\
    --datasets ReunionGreen \\
    --shortlist-n 50 \\
    --out docs/results/reunion_md_local_shortlist_N50.csv
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

from scripts.compare_base_models import (  # noqa: E402
    DATASETS,
    MODS,
    cascade_similarity,
    default_device,
    load_or_compute_aliked,
)
from sides_matching import MegaDescriptor, Prediction, get_transform  # noqa: E402
from sides_matching.dedode_lg_matching import (  # noqa: E402
    DeDoDeLightGlueMatcher,
    load_or_compute_dedode_lg_shortlist_similarity,
)
from sides_matching.xfeat_matching import (  # noqa: E402
    XFeatMatcher,
    load_or_compute_xfeat_shortlist_similarity,
)


def evaluate_dataset(
    *,
    name: str,
    data_root: Path,
    features_dir: Path,
    cache_dir: Path,
    device: str,
    dedode_matcher: DeDoDeLightGlueMatcher,
    xfeat_matcher: XFeatMatcher,
    grayscale: bool,
    img_size: int,
    shortlist_n: int,
    target_top1: float,
    reuse_reunion_cache: Path | None,
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
    print(f"\n=== {name}: n={n} ids={df['identity'].nunique()} shortlist={shortlist_n} ===")

    rows: list[dict] = []
    curves: dict[str, list[float]] = {}

    for flip in (False, True):
        md_q = features_dir / f"MegaDescriptor_{name}_flip={flip}_grayscale={grayscale}.pickle"
        md_db = features_dir / f"MegaDescriptor_{name}_flip=False_grayscale={grayscale}.pickle"
        al_q = (
            features_dir
            / f"Aliked_{name}_flip={flip}_grayscale={grayscale}_{img_size}.pickle"
        )
        al_db = (
            features_dir
            / f"Aliked_{name}_flip=False_grayscale={grayscale}_{img_size}.pickle"
        )

        md_sim = MegaDescriptor(str(md_q), str(md_db)).compute_similarity(ignore="diagonal")
        al_sim = load_or_compute_aliked(
            name=name,
            flip=flip,
            path_query=str(al_q),
            path_database=str(al_db),
            cache_dir=cache_dir,
            device=device,
            reuse_reunion_cache=reuse_reunion_cache,
        )
        xfeat_short = load_or_compute_xfeat_shortlist_similarity(
            name=name,
            data_root=root,
            paths=image_paths,
            flip_query=flip,
            md_sim=md_sim,
            top_n=shortlist_n,
            cache_dir=cache_dir,
            matcher=xfeat_matcher,
        )
        dedode_short = load_or_compute_dedode_lg_shortlist_similarity(
            name=name,
            data_root=root,
            paths=image_paths,
            flip_query=flip,
            md_sim=md_sim,
            top_n=shortlist_n,
            cache_dir=cache_dir,
            matcher=dedode_matcher,
        )

        methods: dict[str, Prediction] = {
            f"MD→ALIKED N={shortlist_n}": Prediction(
                df, cascade_similarity(md_sim, al_sim, shortlist_n), k=n - 1
            ),
            f"MD→XFeat N={shortlist_n}": Prediction(
                df, cascade_similarity(md_sim, xfeat_short, shortlist_n), k=n - 1
            ),
            f"MD→DeDoDe-v2+LG N={shortlist_n}": Prediction(
                df, cascade_similarity(md_sim, dedode_short, shortlist_n), k=n - 1
            ),
        }

        for method, pred in methods.items():
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
                "opp_top5": float(pred.accuracy["different orientation"]["top 5"]),
                "gap_to_target_top1": float(target_top1 - pred.accuracy["full"]["top 1"]),
            }
            rows.append(row)
            curves[f"{name}|{method}|flip={flip}|full"] = [
                float(pred.accuracy["full"][f"top {i}"]) for i in range(1, 11)
            ]
            print(
                f"  flip={flip} {method:30s} "
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
        "--reunion-aliked-cache",
        type=Path,
        default=Path("/tmp/reunion_green_sims"),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["ReunionGreen"],
        choices=sorted(DATASETS),
    )
    p.add_argument("--device", default=default_device())
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--grayscale", action="store_true", default=False)
    p.add_argument("--shortlist-n", type=int, default=50)
    p.add_argument("--target-top1", type=float, default=0.80)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/results/reunion_md_local_shortlist_N50.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.features_dir.is_dir():
        raise SystemExit(f"features dir not found: {args.features_dir}")
    if not args.data_dir.is_dir():
        raise SystemExit(f"data dir not found: {args.data_dir}")

    print(f"device={args.device}")
    print(f"shortlist_n={args.shortlist_n}")
    print(f"datasets={args.datasets}")

    t0 = time.time()
    dedode_matcher = DeDoDeLightGlueMatcher(device=args.device)
    print(f"DeDoDe-v2+LG loaded in {time.time() - t0:.1f}s")
    t0 = time.time()
    xfeat_matcher = XFeatMatcher(device=args.device)
    print(f"XFeat loaded in {time.time() - t0:.1f}s")

    all_rows: list[dict] = []
    all_curves: dict[str, list[float]] = {}
    reuse = args.reunion_aliked_cache if args.reunion_aliked_cache.is_dir() else None

    for name in args.datasets:
        rows, curves = evaluate_dataset(
            name=name,
            data_root=args.data_dir,
            features_dir=args.features_dir,
            cache_dir=args.cache_dir,
            device=args.device,
            dedode_matcher=dedode_matcher,
            xfeat_matcher=xfeat_matcher,
            grayscale=args.grayscale,
            img_size=args.img_size,
            shortlist_n=args.shortlist_n,
            target_top1=args.target_top1,
            reuse_reunion_cache=reuse,
        )
        all_rows.extend(rows)
        all_curves.update(curves)

    out = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps({"rows": all_rows, "curves": all_curves}, indent=2))

    print(f"\n=== SUMMARY (flip=True, N={args.shortlist_n}) ===")
    sub = out[out["flip"] == True][
        ["dataset", "method", "full_top1", "full_top5", "opp_top1", "gap_to_target_top1"]
    ]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
