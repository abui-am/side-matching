"""Tests for bilateral calibration split helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sides_matching.evaluation import (
    filter_bilateral_df,
    identities_with_both_sides,
    split_calibration_one_per_side,
    subset_identities_df,
)


def _toy_df() -> pd.DataFrame:
    """Three bilateral identities with an extra photo each; one unilateral identity."""
    return pd.DataFrame(
        {
            "identity": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d"],
            "orientation": [
                "left",
                "right",
                "left",
                "left",
                "right",
                "top",
                "left",
                "right",
                "right",
                "left",
                "left",
            ],
        }
    )


def test_identities_with_both_sides():
    df = _toy_df()
    bilateral = identities_with_both_sides(df)
    assert bilateral == {"a", "b", "c"}
    assert "d" not in bilateral


def test_filter_bilateral_df():
    df = _toy_df()
    filtered, keep_idx = filter_bilateral_df(df)
    assert len(filtered) == 9
    assert len(keep_idx) == 9
    assert set(filtered["identity"].unique()) == {"a", "b", "c"}
    assert "d" not in filtered["identity"].to_numpy()


def test_split_calibration_one_per_side_counts():
    df, _ = filter_bilateral_df(_toy_df())
    val_idx, test_idx = split_calibration_one_per_side(df, seed=0)
    assert len(val_idx) == 6  # 3 identities × (1 left + 1 right)
    assert len(test_idx) == 3
    assert set(val_idx).isdisjoint(test_idx)
    assert set(val_idx) | set(test_idx) == set(range(len(df)))


def test_split_never_picks_unknown_orientation_for_val():
    df, _ = filter_bilateral_df(_toy_df())
    val_idx, _ = split_calibration_one_per_side(df, seed=0)
    orientations = df["orientation"].to_numpy()
    for idx in val_idx:
        assert orientations[idx] in {"left", "right"}


def test_split_is_seed_stable():
    df, _ = filter_bilateral_df(_toy_df())
    val_a, test_a = split_calibration_one_per_side(df, seed=42)
    val_b, test_b = split_calibration_one_per_side(df, seed=42)
    np.testing.assert_array_equal(val_a, val_b)
    np.testing.assert_array_equal(test_a, test_b)


def test_split_picks_one_left_when_multiple():
    df = pd.DataFrame(
        {
            "identity": ["x", "x", "x"],
            "orientation": ["left", "left", "right"],
        }
    )
    val_idx, test_idx = split_calibration_one_per_side(df, seed=7)
    assert len(val_idx) == 2
    assert len(test_idx) == 1
    left_positions = np.flatnonzero(df["orientation"].to_numpy() == "left")
    assert sum(idx in left_positions for idx in val_idx) == 1


def test_split_raises_when_identity_not_bilateral():
    df = pd.DataFrame({"identity": ["x", "x"], "orientation": ["left", "left"]})
    with pytest.raises(ValueError, match="lacks left or right"):
        split_calibration_one_per_side(df, seed=0)


def test_subset_identities_df():
    df, _ = filter_bilateral_df(_toy_df())
    sub = subset_identities_df(df, max_identities=2, seed=0)
    assert sub["identity"].nunique() == 2
    assert len(sub) < len(df)


def test_split_raises_when_no_test_queries():
    df = pd.DataFrame(
        {"identity": ["x", "x"], "orientation": ["left", "right"]}
    )
    with pytest.raises(ValueError, match="no held-out test"):
        split_calibration_one_per_side(df, seed=0)
