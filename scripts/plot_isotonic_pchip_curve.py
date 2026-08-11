#!/usr/bin/env python3
"""Plot isotonic steps vs PCHIP-smoothed calibration curve for documentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wildlife_tools.similarity.calibration import IsotonicCalibration

REPO_ROOT = Path(__file__).resolve().parents[1]


def synthetic_xfeat_pairs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate shortlist (score, same-identity?) pairs like XFeat match counts."""
    scores: list[float] = []
    hits: list[float] = []
    for raw in range(0, 26):
        n_pos = max(8, 40 - raw * 2)
        n_neg = max(8, 20 + raw)
        scores.extend([float(raw)] * n_pos)
        hits.extend([1.0] * n_pos)
        scores.extend([float(raw)] * n_neg)
        hits.extend([0.0] * n_neg)
        scores.extend([raw + rng.uniform(-0.35, 0.35) for _ in range(6)])
        hits.extend(rng.integers(0, 2, size=6).astype(float))
    return np.asarray(scores, dtype=np.float64), np.asarray(hits, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/figures/isotonic_pchip_curve.png",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    scores, hits = synthetic_xfeat_pairs(rng)

    iso_only = IsotonicCalibration(interpolate=False, strict=False)
    iso_only.fit(scores, hits)

    iso_pchip = IsotonicCalibration(interpolate=True, strict=True)
    iso_pchip.fit(scores, hits)

    x_line = np.linspace(scores.min(), scores.max(), 400)
    y_step = iso_only.predict(x_line)
    y_smooth = iso_pchip.predict(x_line)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.scatter(
        scores,
        hits,
        s=6,
        alpha=0.08,
        color="#94a3b8",
        label="Calibration pairs (score, hit)",
        rasterized=True,
    )

    ax.plot(
        x_line,
        y_step,
        color="#f97316",
        linewidth=2.2,
        drawstyle="steps-post",
        label="Isotonic (stepwise)",
    )
    ax.plot(
        x_line,
        y_smooth,
        color="#2563eb",
        linewidth=2.2,
        label="Isotonic + PCHIP (smooth, strict)",
    )

    demo_x = np.array([11.0, 12.0, 13.0])
    demo_step = iso_only.predict(demo_x)
    demo_smooth = iso_pchip.predict(demo_x)
    ax.scatter(demo_x, demo_step, s=70, color="#f97316", zorder=5, edgecolors="white")
    ax.scatter(demo_x, demo_smooth, s=70, color="#2563eb", zorder=5, edgecolors="white")
    ax.annotate(
        "11 & 12 tie on steps",
        xy=(11.5, demo_step[1]),
        xytext=(14.5, demo_step[1] - 0.12),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#64748b"},
        color="#64748b",
    )
    ax.annotate(
        "PCHIP separates nearby scores",
        xy=(12.0, demo_smooth[1]),
        xytext=(14.8, demo_smooth[1] + 0.1),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#64748b"},
        color="#64748b",
    )

    ax.set_xlabel("Raw score (e.g. XFeat match count)")
    ax.set_ylabel("Calibrated hit rate")
    ax.set_title("Isotonic regression vs isotonic + PCHIP calibration")
    ax.set_xlim(-0.5, 25.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
