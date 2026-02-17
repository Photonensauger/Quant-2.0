"""Tests for quant.features.technical -- technical indicator computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.features.technical import compute_technical_features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP = 50  # default warmup period before NaN-free rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeTechnicalFeatures:
    """Tests for compute_technical_features()."""

    def test_compute_technical_features_shape(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Output has same row count as input and adds ~40 feature columns."""
        result = compute_technical_features(sample_ohlcv_df)

        # Same number of rows
        assert result.shape[0] == sample_ohlcv_df.shape[0]

        # Original 5 OHLCV columns + ~40 new features
        n_new_cols = result.shape[1] - sample_ohlcv_df.shape[1]
        assert n_new_cols >= 35, f"Expected >=35 new columns, got {n_new_cols}"

    def test_no_nan_after_warmup(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """After the warmup period, no NaN values should remain in features."""
        result = compute_technical_features(sample_ohlcv_df)

        # Separate feature columns (exclude original OHLCV)
        feature_cols = [
            c for c in result.columns if c not in sample_ohlcv_df.columns
        ]
        features_after_warmup = result[feature_cols].iloc[WARMUP:]

        nan_counts = features_after_warmup.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        assert cols_with_nan.empty, (
            f"NaN values found after warmup row {WARMUP} in: "
            f"{cols_with_nan.to_dict()}"
        )

    def test_rsi_range(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """RSI values must be in [0, 100] wherever they are not NaN."""
        result = compute_technical_features(sample_ohlcv_df)

        for col in ("rsi_14", "rsi_28"):
            vals = result[col].dropna()
            assert vals.min() >= 0.0, f"{col} has value below 0: {vals.min()}"
            assert vals.max() <= 100.0, f"{col} has value above 100: {vals.max()}"

    def test_bollinger_bands_relationship(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Bollinger band width must be non-negative; bb_pct should be finite."""
        result = compute_technical_features(sample_ohlcv_df)

        bb_width = result["bb_width"].dropna()
        assert (bb_width >= 0).all(), "Bollinger band width has negative values"

        # bb_pct can exceed [0, 1] when price breaks out of bands, but it
        # should still be finite.
        bb_pct = result["bb_pct"].dropna()
        assert np.isfinite(bb_pct).all(), "bb_pct has non-finite values"

    def test_atr_positive(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """ATR (normalised) must be non-negative wherever it is not NaN."""
        result = compute_technical_features(sample_ohlcv_df)

        for col in ("atr_14", "atr_28"):
            vals = result[col].dropna()
            assert (vals >= 0).all(), f"{col} contains negative values"

    def test_volume_features_exist(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Volume-derived features must be present in the output."""
        result = compute_technical_features(sample_ohlcv_df)

        expected_volume_cols = {"volume_ratio", "volume_change", "obv_zscore"}
        actual_cols = set(result.columns)
        missing = expected_volume_cols - actual_cols
        assert not missing, f"Missing volume features: {missing}"

    def test_macd_components(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """MACD line, signal, and histogram must be present and consistent."""
        result = compute_technical_features(sample_ohlcv_df)

        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

        # Histogram = MACD line - signal line
        valid = result[["macd", "macd_signal", "macd_hist"]].dropna()
        np.testing.assert_allclose(
            valid["macd_hist"].values,
            (valid["macd"] - valid["macd_signal"]).values,
            atol=1e-10,
            err_msg="MACD histogram != macd - macd_signal",
        )

    def test_momentum_indicators(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Momentum and ROC features must be present and finite after warmup."""
        result = compute_technical_features(sample_ohlcv_df)

        momentum_cols = {
            "momentum_10", "momentum_20", "roc_5", "roc_10", "roc_20",
        }
        actual_cols = set(result.columns)
        missing = momentum_cols - actual_cols
        assert not missing, f"Missing momentum features: {missing}"

        # Values should be finite after warmup
        for col in momentum_cols:
            vals = result[col].iloc[WARMUP:].dropna()
            assert np.isfinite(vals).all(), (
                f"{col} has non-finite values after warmup"
            )

    def test_empty_dataframe(self) -> None:
        """An empty DataFrame with correct columns should return empty."""
        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
        result = compute_technical_features(empty_df)
        assert result.shape[0] == 0

    def test_missing_columns_raises(self) -> None:
        """A DataFrame missing required columns must raise ValueError."""
        bad_df = pd.DataFrame({"open": [1.0], "high": [2.0]})
        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            compute_technical_features(bad_df)
