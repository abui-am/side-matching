# Base model comparison

How we compare **base MegaDescriptor**, **ALIKED**, **calibrated fusion**, and **MD→ALIKED shortlist rerank** on opposite-side sea turtle re-ID — and the numbers from the latest run.

Script: [`scripts/compare_base_models.py`](../scripts/compare_base_models.py)  
Results snapshot: [`docs/results/base_model_comparison.csv`](results/base_model_comparison.csv)

## Goal

Measure **full top-1** identity accuracy against an **80%** target, with top-5 and opposite-side top-1 as supporting metrics. No LoRA in this comparison (frozen MegaDescriptor-L-384 only).

## Datasets

| Dataset | Images | Identities | Role |
|---|---:|---:|---|
| Zakynthos | 160 | 40 | Zero-shot holdout (typical LoRA eval set) |
| Reunion green | 200 | 50 | In-domain hard set |

Images are balanced left/right. Self-matches are removed (`ignore='diagonal'`).

## Methods

| Method | Similarity | Notes |
|---|---|---|
| **MegaDescriptor** | Cosine on global embeddings | Query features may be horizontally flipped; gallery is always `flip=False` |
| **ALIKED** | ALIKED keypoints + LightGlue match counts | Same flip protocol; expensive all-pairs matching (cached to `.npy`) |
| **FusionCalibrated** | WildFusion-style isotonic+PCHIP per stream, then equal-weight average | Fits calibrators on a held-out **50% of queries** (`seed=0`); accuracy reported on the other 50% only |
| **MD→ALIKED N=30** | Cascade | Rank gallery by MegaDescriptor, keep top-30, rerank those with ALIKED, append remaining MD order for \(k>30\) |

Default settings: `grayscale=False`, `img_size=384`, shortlist \(N=30\), target top-1 \(=0.80\).

### Query flip

- **`flip=False`**: query embedding/keypoints as-is vs unflipped gallery.
- **`flip=True`**: horizontally flip the **query** features (left↔right proxy) vs unflipped gallery. This is the usual opposite-side protocol.

### Metrics

Computed with `Prediction.compute_accuracy`:

- **full top-1 / top-5** — correct identity anywhere in the top-\(k\) unique identity ranking.
- **opp. top-1 / top-5** (`different orientation`) — same, but same-orientation same-identity gallery hits are ignored (opposite-side emphasis).
- **gap to target** — `0.80 - full_top1` (negative means already above 80%).

## How to reproduce

```bash
.venv/bin/python scripts/compare_base_models.py \
  --features-dir /path/to/features \
  --data-dir data \
  --cache-dir /tmp/base_model_sims \
  --out docs/results/base_model_comparison.csv
```

Required feature pickles per dataset `NAME` (under `--features-dir`):

- `MegaDescriptor_{NAME}_flip={True,False}_grayscale=False.pickle`
- `Aliked_{NAME}_flip={True,False}_grayscale=False_384.pickle`

ALIKED similarity matrices are cached as `{NAME}_aliked_flip{True|False}.npy` under `--cache-dir` (and optionally reuse `/tmp/reunion_green_sims` for Reunion).

Outputs:

- CSV table of metrics
- JSON with the same rows plus top-\(k\) curves (`k=1..10`) beside the CSV (`.json`)

## Results (2026-08-10)

Features from local pickles under `tes/model/features`. Device: Apple MPS for LightGlue. Fusion uses `val_fraction=0.5`, `seed=0`.

### Query flip = True (primary opposite-side protocol)

| Dataset | Method | N queries | Full top-1 | Full top-5 | Opp. top-1 | Gap to 80% |
|---|---|---:|---:|---:|---:|---:|
| Zakynthos | MegaDescriptor | 160 | **76.2%** | 97.5% | 61.2% | +3.8pp |
| Zakynthos | ALIKED | 160 | 31.9% | 59.4% | 31.9% | +48.1pp |
| Zakynthos | FusionCalibrated | 80* | 77.5% | 98.7% | 67.5% | +2.5pp |
| Zakynthos | MD→ALIKED N=30 | 160 | 53.1% | 80.6% | 53.1% | +26.9pp |
| ReunionGreen | MegaDescriptor | 200 | 63.0% | 84.0% | 60.0% | +17.0pp |
| ReunionGreen | ALIKED | 200 | 62.5% | 83.5% | 62.5% | +17.5pp |
| ReunionGreen | FusionCalibrated | 100* | **74.0%** | 91.0% | 74.0% | +6.0pp |
| ReunionGreen | MD→ALIKED N=30 | 200 | 69.5% | 85.0% | 69.5% | +10.5pp |

\*Fusion accuracy is on the held-out half of queries after calibration fit.

### Query flip = False

| Dataset | Method | N queries | Full top-1 | Full top-5 | Opp. top-1 | Gap to 80% |
|---|---|---:|---:|---:|---:|---:|
| Zakynthos | MegaDescriptor | 160 | 75.6% | 98.7% | 62.5% | +4.4pp |
| Zakynthos | ALIKED | 160 | 63.1% | 71.9% | 0.0% | +16.9pp |
| Zakynthos | FusionCalibrated | 80* | **83.7%** | 100% | 61.2% | **−3.7pp** |
| Zakynthos | MD→ALIKED N=30 | 160 | 69.4% | 82.5% | 1.9% | +10.6pp |
| ReunionGreen | MegaDescriptor | 200 | 62.0% | 83.0% | 60.5% | +18.0pp |
| ReunionGreen | ALIKED | 200 | 65.5% | 75.0% | 0.0% | +14.5pp |
| ReunionGreen | FusionCalibrated | 100* | 73.0% | 87.0% | 50.0% | +7.0pp |
| ReunionGreen | MD→ALIKED N=30 | 200 | 41.0% | 51.0% | 0.0% | +39.0pp |

## Takeaways

1. **Zakynthos base MegaDescriptor is close to 80%** (76.2% top-1 with flip; −3.8pp).
2. **Calibrated fusion without flip already clears 80% on Zakynthos** (83.7% on the held-out half).
3. **Reunion green is harder**: best base fusion is 74% top-1 with flip (−6pp to target).
4. **ALIKED-only rerank is not universal**: with flip it helps Reunion (69.5%) but hurts Zakynthos (53.1%); without flip it collapses opposite-side on both.
5. Closing the remaining gap (especially Reunion) likely needs a stronger first stage (e.g. opposite-side LoRA) plus calibrated shortlist fusion — not larger ALIKED shortlists alone.

## Related code

- Similarity / fusion: `sides_matching/predictions.py` (`MegaDescriptor`, `Aliked`, `Combined`, `Prediction`)
- Feature extraction notebook: `notebooks/compute_features.ipynb`
- Matching notebook: `notebooks/matching.ipynb`
