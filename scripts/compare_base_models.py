#!/usr/bin/env python3
"""Compare base MegaDescriptor, ALIKED, calibrated fusion, and MD→ALIKED cascade.

Example:
  .venv/bin/python scripts/compare_base_models.py \\
    --features-dir /Users/abui/@code/tes/model/features \\
    --data-dir data \\
    --cache-dir /tmp/base_model_sims \\
    --out /tmp/base_model_comparison.csv
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
from wildlife_tools.similarity import MatchLightGlue

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sides_matching import (  # noqa: E402
    Combined,
    MegaDescriptor,
    Prediction,
    get_transform,
    reunion_green,
    zakynthos,
)
from sides_matching.predictions import Data  # noqa: E402


MODS = ["full", "different orientation", "same orientation"]
DATASETS = {
    "Zakynthos": ("ZakynthosTurtles", zakynthos),
    "ReunionGreen": ("ReunionTurtles", reunion_green),
}


class AlikedDevice(Data):
    """ALIKED + LightGlue with an explicit torch device (mps/cuda/cpu)."""

    def __init__(self, path_features_query, path_features_database, *, device: str):
        super().__init__(path_features_query, path_features_database)
        self.matcher = MatchLightGlue("aliked", device=device)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cascade_similarity(md_sim: np.ndarray, al_sim: np.ndarray, top_n: int) -> np.ndarray:
    """Rerank MegaDescriptor top-n with ALIKED; append MD-ordered tail for k > n."""
    n = md_sim.shape[0]
    out = np.full_like(md_sim, -np.inf, dtype=np.float64)
    for i in range(n):
        finite = np.flatnonzero(np.isfinite(md_sim[i]))
        order = finite[np.argsort(-md_sim[i, finite])]
        short, rest = order[:top_n], order[top_n:]
        al_key = np.where(np.isfinite(al_sim[i, short]), al_sim[i, short], -np.inf)
        rerank = short[np.lexsort((-md_sim[i, short], -al_key))]
        ranked = np.concatenate([rerank, rest])
        out[i, ranked] = np.arange(len(ranked), 0, -1, dtype=np.float64)
    return out


def load_or_compute_aliked(
    *,
    name: str,
    flip: bool,
    path_query: str,
    path_database: str,
    cache_dir: Path,
    device: str,
    reuse_reunion_cache: Path | None,
) -> np.ndarray:
    if name == "ReunionGreen" and reuse_reunion_cache is not None:
        legacy = reuse_reunion_cache / f"aliked_flip{flip}.npy"
        if legacy.is_file():
            print(f"  ALIKED legacy cache {legacy}")
            return np.load(legacy)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}_aliked_flip{flip}.npy"
    if cache.is_file():
        print(f"  ALIKED cache hit {cache}")
        return np.load(cache)

    t0 = time.time()
    sim = AlikedDevice(path_query, path_database, device=device).compute_similarity(
        ignore="diagonal"
    )
    np.save(cache, sim)
    print(f"  ALIKED flip={flip} done in {time.time() - t0:.1f}s -> {cache}")
    return sim


def evaluate_dataset(
    *,
    name: str,
    data_root: Path,
    features_dir: Path,
    cache_dir: Path,
    device: str,
    grayscale: bool,
    img_size: int,
    shortlist_n: int,
    val_fraction: float,
    seed: int,
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
    print(f"\n=== {name}: n={n} ids={df['identity'].nunique()} ===")

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

        methods: dict[str, Prediction] = {
            "MegaDescriptor": Prediction(df, md_sim, k=n - 1),
            "ALIKED": Prediction(df, al_sim, k=n - 1),
        }

        fused = Combined(
            methods["MegaDescriptor"],
            methods["ALIKED"],
            method="calibrated",
            val_fraction=val_fraction,
            seed=seed,
        )
        fus_sim = fused.compute_similarity(ignore="diagonal")
        methods["FusionCalibrated"] = Prediction(
            df, fus_sim, k=n - 1, query_indices=fused.test_indices
        )

        casc = cascade_similarity(md_sim, al_sim, shortlist_n)
        methods[f"MD→ALIKED N={shortlist_n}"] = Prediction(df, casc, k=n - 1)

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
                f"  flip={flip} {method:22s} "
                f"top1={row['full_top1']:.3f} top5={row['full_top5']:.3f} "
                f"opp1={row['opp_top1']:.3f} gap={row['gap_to_target_top1']:+.3f}"
            )

    return rows, curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing ZakynthosTurtles / ReunionTurtles",
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=Path("/Users/abui/@code/tes/model/features"),
        help="Directory with MegaDescriptor_* and Aliked_* pickles",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/base_model_sims"),
        help="Cache directory for ALIKED similarity matrices",
    )
    p.add_argument(
        "--reunion-aliked-cache",
        type=Path,
        default=Path("/tmp/reunion_green_sims"),
        help="Optional legacy Reunion ALIKED .npy cache dir",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["Zakynthos", "ReunionGreen"],
        choices=sorted(DATASETS),
    )
    p.add_argument("--device", default=default_device(), help="Torch device for LightGlue")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--grayscale", action="store_true", default=False)
    p.add_argument("--shortlist-n", type=int, default=30)
    p.add_argument("--val-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-top1", type=float, default=0.80)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/base_model_comparison.csv"),
        help="CSV output path (JSON written beside it with .json suffix)",
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
            grayscale=args.grayscale,
            img_size=args.img_size,
            shortlist_n=args.shortlist_n,
            val_fraction=args.val_fraction,
            seed=args.seed,
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
