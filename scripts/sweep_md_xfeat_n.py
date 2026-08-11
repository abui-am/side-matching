#!/usr/bin/env python3
"""Sweep shortlist N for MD→XFeat on Amvrakikos / ReunionGreen / ReunionHawksbill.

Builds one XFeat shortlist matrix at --max-n per dataset×flip, then evaluates
cascade for each N ≤ max-n. Also reports MD recall@N.
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
    MegaDescriptor,
    Prediction,
    get_transform,
    amvrakikos,
    reunion_green,
    reunion_hawksbill,
)
from sides_matching.xfeat_matching import (  # noqa: E402
    XFeatMatcher,
    load_or_compute_xfeat_shortlist_similarity,
)

DATASETS = {
    "Amvrakikos": ("AmvrakikosTurtles", amvrakikos),
    "ReunionGreen": ("ReunionTurtles", reunion_green),
    "ReunionHawksbill": ("ReunionTurtles", reunion_hawksbill),
}


def md_recall_at_n(md_sim: np.ndarray, identity: np.ndarray, top_n: int) -> float:
    """Fraction of queries whose correct identity appears in MD top-N gallery images."""
    hits = 0
    n = md_sim.shape[0]
    for i in range(n):
        finite = np.flatnonzero(np.isfinite(md_sim[i]))
        order = finite[np.argsort(-md_sim[i, finite])][:top_n]
        if np.any(identity[order] == identity[i]):
            hits += 1
    return hits / n


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
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--max-n", type=int, default=100)
    p.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 30, 40, 50, 75, 100],
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/results/md_xfeat_n_sweep.csv",
    )
    return p.parse_args()


def evaluate_dataset(
    *,
    name: str,
    data_root: Path,
    features_dir: Path,
    cache_dir: Path,
    matcher: XFeatMatcher,
    img_size: int,
    max_n: int,
    ns: list[int],
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
    identity = df["identity"].to_numpy()
    paths = df["path"].tolist()
    # Cap N by gallery size minus self.
    usable = [x for x in ns if x < n]
    print(f"\n=== {name}: n={n} ids={df['identity'].nunique()} max_n={max_n} ===")

    rows: list[dict] = []
    for flip in (False, True):
        md_q = features_dir / f"MegaDescriptor_{name}_flip={flip}_grayscale=False.pickle"
        md_db = features_dir / f"MegaDescriptor_{name}_flip=False_grayscale=False.pickle"
        if not md_q.is_file() or not md_db.is_file():
            raise SystemExit(f"missing MegaDescriptor pickles for {name} flip={flip}")

        md_sim = MegaDescriptor(str(md_q), str(md_db)).compute_similarity(ignore="diagonal")
        top_n_build = min(max_n, n - 1)
        xfeat = load_or_compute_xfeat_shortlist_similarity(
            name=name,
            data_root=root,
            paths=paths,
            flip_query=flip,
            md_sim=md_sim,
            top_n=top_n_build,
            cache_dir=cache_dir,
            matcher=matcher,
        )

        md_pred = Prediction(df, md_sim, k=n - 1)
        md_pred.compute_accuracy(MODS)
        md_top1 = float(md_pred.accuracy["full"]["top 1"])
        print(f"flip={flip} MegaDescriptor top1={md_top1:.3f}")

        for top_n in usable:
            if top_n > top_n_build:
                continue
            recall = md_recall_at_n(md_sim, identity, top_n)
            casc = cascade_similarity(md_sim, xfeat, top_n)
            pred = Prediction(df, casc, k=n - 1)
            pred.compute_accuracy(MODS)
            row = {
                "dataset": name,
                "method": f"MD→XFeat N={top_n}",
                "flip": flip,
                "shortlist_n": top_n,
                "md_recall_at_n": recall,
                "full_top1": float(pred.accuracy["full"]["top 1"]),
                "full_top5": float(pred.accuracy["full"]["top 5"]),
                "opp_top1": float(pred.accuracy["different orientation"]["top 1"]),
                "md_top1": md_top1,
            }
            rows.append(row)
            print(
                f"  N={top_n:3d} recall@N={recall:.3f} "
                f"top1={row['full_top1']:.3f} top5={row['full_top5']:.3f} "
                f"opp1={row['opp_top1']:.3f}"
            )
    return rows


def main() -> None:
    args = parse_args()
    ns = sorted({n for n in args.ns if n >= 1})
    if not ns:
        raise SystemExit("no valid N values")

    print(f"device={args.device}")
    print(f"datasets={args.datasets}")
    print(f"max_n={args.max_n} ns={ns}")

    matcher = XFeatMatcher(device=args.device)
    rows: list[dict] = []
    for name in args.datasets:
        rows.extend(
            evaluate_dataset(
                name=name,
                data_root=args.data_dir,
                features_dir=args.features_dir,
                cache_dir=args.cache_dir,
                matcher=matcher,
                img_size=args.img_size,
                max_n=args.max_n,
                ns=ns,
            )
        )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    args.out.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2))

    print("\n=== BEST per dataset (flip=True, by full top-1) ===")
    sub = out[out["flip"] == True]
    for name, g in sub.groupby("dataset"):
        best = g.sort_values("full_top1", ascending=False).iloc[0]
        print(
            f"{name}: best N={int(best['shortlist_n'])} "
            f"top1={best['full_top1']:.4f} "
            f"(MD recall@N={best['md_recall_at_n']:.4f}, MD alone={best['md_top1']:.4f})"
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
