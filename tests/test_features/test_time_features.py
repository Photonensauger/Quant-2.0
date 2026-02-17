"""Tests for quant.features.time_features -- cyclical calendar features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.features.time_features import compute_time_features


class TestComputeTimeFeatures:
    """Tests for compute_time_features()."""

    def test_time_features_shape(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Output row count matches input and adds 10 sin/cos columns."""
        result = compute_time_features(sample_ohlcv_df)

        assert result.shape[0] == sample_ohlcv_df.shape[0]

        # 5 cycles x 2 (sin + cos) = 10 new columns
        n_new_cols = result.shape[1] - sample_ohlcv_df.shape[1]
        assert n_new_cols == 10, f"Expected 10 new columns, got {n_new_cols}"

    def test_no_nan(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Time features must never contain NaN values."""
        result = compute_time_features(sample_ohlcv_df)

        time_cols = [
            c for c in result.columns if c not in sample_ohlcv_df.columns
        ]
        nan_count = result[time_cols].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in time features"

    def test_sin_cos_range(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """All sin/cos values must be in [-1, +1]."""
        result = compute_time_features(sample_ohlcv_df)

        sin_cos_cols = [
            c
            for c in result.columns
            if c.endswith("_sin") or c.endswith("_cos")
        ]
        assert len(sin_cos_cols) == 10, (
            f"Expected 10 sin/cos columns, got {len(sin_cos_cols)}"
        )

        for col in sin_cos_cols:
            vals = result[col]
            assert vals.min() >= -1.0 - 1e-10, (
                f"{col} has value below -1: {vals.min()}"
            )
            assert vals.max() <= 1.0 + 1e-10, (
                f"{col} has value above +1: {vals.max()}"
            )

    def test_different_timestamps(self) -> None:
        """Features should differ for timestamps at different hours / days."""
        # Monday 09:00 vs Friday 17:30
        idx_a = pd.DatetimeIndex(["2024-01-01 09:00:00"])
        idx_b = pd.DatetimeIndex(["2024-01-05 17:30:00"])

        df_a = pd.DataFrame({"value": [1.0]}, index=idx_a)
        df_b = pd.DataFrame({"value": [1.0]}, index=idx_b)

        result_a = compute_time_features(df_a)
        result_b = compute_time_features(df_b)

        time_cols = [c for c in result_a.columns if c != "value"]

        # At least some of the sin/cos values should differ
        diffs = []
        for col in time_cols:
            diffs.append(abs(result_a[col].iloc[0] - result_b[col].iloc[0]))

        assert max(diffs) > 0.01, (
            "Time features identical for very different timestamps"
        )

    def test_non_datetime_index_raises(self) -> None:
        """A DataFrame without DatetimeIndex must raise TypeError."""
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        with pytest.raises(TypeError, match="Expected DatetimeIndex"):
            compute_time_features(df)
