"""Tests for calibrated Combined similarity fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sides_matching.predictions import Combined, Prediction


def _toy_predictions():
    """Two streams on different scales with known same-identity structure."""
    identity = np.array(["a", "a", "b", "b", "c", "c"])
    df = pd.DataFrame(
        {
            "identity": identity,
            "orientation": ["left", "right"] * 3,
            "year": [2020] * 6,
        }
    )
    # Stream 0: cosine-like scores in [-1, 1]
    sim0 = np.array(
        [
            [1.0, 0.9, 0.1, 0.0, -0.2, -0.1],
            [0.9, 1.0, 0.05, 0.1, -0.15, -0.05],
            [0.1, 0.05, 1.0, 0.85, 0.0, -0.1],
            [0.0, 0.1, 0.85, 1.0, -0.05, 0.05],
            [-0.2, -0.15, 0.0, -0.05, 1.0, 0.8],
            [-0.1, -0.05, -0.1, 0.05, 0.8, 1.0],
        ]
    )
    # Stream 1: match-count-like scores in [0, 50]
    sim1 = np.array(
        [
            [50.0, 40.0, 2.0, 1.0, 0.0, 1.0],
            [40.0, 50.0, 1.0, 3.0, 1.0, 0.0],
            [2.0, 1.0, 50.0, 35.0, 2.0, 1.0],
            [1.0, 3.0, 35.0, 50.0, 1.0, 2.0],
            [0.0, 1.0, 2.0, 1.0, 50.0, 30.0],
            [1.0, 0.0, 1.0, 2.0, 30.0, 50.0],
        ]
    )
    np.fill_diagonal(sim0, -np.inf)
    np.fill_diagonal(sim1, -np.inf)
    p0 = Prediction(df, sim0, k=5)
    p1 = Prediction(df, sim1, k=5)
    return p0, p1


def test_max_method_preserves_legacy_behavior():
    p0, p1 = _toy_predictions()
    combined = Combined(p0, p1, method="max")
    out = combined.compute_similarity()
    expected = np.maximum(p0.similarity, p1.similarity)
    np.testing.assert_allclose(out, expected)
    assert combined.test_indices.tolist() == list(range(len(p0.identity)))


def test_calibrated_fit_uses_held_out_query_split():
    p0, p1 = _toy_predictions()
    combined = Combined(p0, p1, method="calibrated", val_fraction=0.5, seed=0)
    n = len(p0.identity)
    assert len(combined.val_indices) + len(combined.test_indices) == n
    assert set(combined.val_indices).isdisjoint(combined.test_indices)
    assert set(combined.val_indices) | set(combined.test_indices) == set(range(n))
    assert combined.calibrators is not None
    assert len(combined.calibrators) == 2


def test_calibrated_average_is_in_unit_interval_for_finite_entries():
    p0, p1 = _toy_predictions()
    combined = Combined(p0, p1, method="calibrated", val_fraction=0.5, seed=1)
    out = combined.compute_similarity()
    finite = np.isfinite(out)
    assert finite.any()
    assert np.all(out[finite] >= -1e-6)
    assert np.all(out[finite] <= 1.0 + 1e-6)
    assert np.all(~np.isfinite(np.diag(out)))


def test_explicit_val_indices():
    p0, p1 = _toy_predictions()
    val_indices = np.array([0, 1, 2])
    combined = Combined(p0, p1, method="calibrated", val_indices=val_indices)
    np.testing.assert_array_equal(combined.val_indices, val_indices)
    np.testing.assert_array_equal(combined.test_indices, np.array([3, 4, 5]))


def test_prediction_accuracy_respects_query_indices():
    p0, p1 = _toy_predictions()
    combined = Combined(p0, p1, method="calibrated", val_fraction=0.5, seed=0)
    similarity = combined.compute_similarity()
    pred = Prediction(p0.df, similarity, k=5, query_indices=combined.test_indices)
    pred.compute_accuracy(["full"])
    assert len(pred.true) == len(combined.test_indices)
    assert "full" in pred.accuracy


def test_unknown_method_raises():
    p0, p1 = _toy_predictions()
    with pytest.raises(ValueError, match="Unknown method"):
        Combined(p0, p1, method="median")
