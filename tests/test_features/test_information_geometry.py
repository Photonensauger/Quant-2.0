"""Tests for quant.features.information_geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import FeatureConfig
from quant.features.information_geometry import (
    _fisher_information,
    _histogram_density,
    _kl_divergence,
    _renyi_entropy,
    compute_information_geometry_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame."""
    np.random.seed(seed)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1.0)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "open": close * 1.001,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.randint(100, 10_000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture()
def config() -> FeatureConfig:
    return FeatureConfig(ig_rolling_window=60, ig_n_bins=50)


@pytest.fixture()
def ohlcv_df() -> pd.DataFrame:
    return _make_ohlcv(n=200)


# ---------------------------------------------------------------------------
# Shape and naming tests
# ---------------------------------------------------------------------------

class TestShapePreserved:
    def test_shape_preserved(self, ohlcv_df: pd.DataFrame, config: FeatureConfig) -> None:
        result = compute_information_geometry_features(ohlcv_df, config)
        assert len(result) == len(ohlcv_df), "Row count must be preserved"
        # 7 new IG columns
        new_cols = [c for c in result.columns if c.startswith("ig_")]
        assert len(new_cols) == 7


class TestColumnNamesPrefixed:
    def test_column_names_prefixed(self, ohlcv_df: pd.DataFrame, config: FeatureConfig) -> None:
        result = compute_information_geometry_features(ohlcv_df, config)
        ig_cols = [c for c in result.columns if c.startswith("ig_")]
        expected = {
            "ig_fisher_info",
            "ig_renyi_h_0.5",
            "ig_renyi_h_2.0",
            "ig_renyi_h_10.0",
            "ig_renyi_slope",
            "ig_kl_rate_5",
            "ig_kl_rate_20",
        }
        assert set(ig_cols) == expected


class TestNoNanAfterWarmup:
    def test_no_nan_after_warmup(self, config: FeatureConfig) -> None:
        df = _make_ohlcv(n=300)
        result = compute_information_geometry_features(df, config)
        # After row 80 (window=60 + max_kl_lag=20) there should be no NaN
        warmup = 80
        ig_cols = [c for c in result.columns if c.startswith("ig_")]
        subset = result.iloc[warmup:][ig_cols]
        nan_count = subset.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values after row {warmup}"


# ---------------------------------------------------------------------------
# Value constraint tests
# ---------------------------------------------------------------------------

class TestFisherInfoNonnegative:
    def test_fisher_info_nonnegative(self, ohlcv_df: pd.DataFrame, config: FeatureConfig) -> None:
        result = compute_information_geometry_features(ohlcv_df, config)
        valid = result["ig_fisher_info"].dropna()
        assert (valid >= 0).all(), "Fisher information must be >= 0"


class TestRenyiEntropyNonnegative:
    def test_renyi_entropy_nonnegative(self, ohlcv_df: pd.DataFrame, config: FeatureConfig) -> None:
        result = compute_information_geometry_features(ohlcv_df, config)
        for col in ["ig_renyi_h_0.5", "ig_renyi_h_2.0", "ig_renyi_h_10.0"]:
            valid = result[col].dropna()
            assert (valid >= 0).all(), f"{col} must be >= 0"


class TestKlRateNonnegative:
    def test_kl_rate_nonnegative(self, ohlcv_df: pd.DataFrame, config: FeatureConfig) -> None:
        result = compute_information_geometry_features(ohlcv_df, config)
        for col in ["ig_kl_rate_5", "ig_kl_rate_20"]:
            valid = result[col].dropna()
            assert (valid >= 0).all(), f"{col} must be >= 0"


class TestRenyiOrdering:
    def test_renyi_ordering(self, config: FeatureConfig) -> None:
        """H_0.5 >= H_2.0 >= H_10.0 (Renyi entropy decreases with alpha)."""
        df = _make_ohlcv(n=300)
        result = compute_information_geometry_features(df, config)
        valid_mask = (
            result["ig_renyi_h_0.5"].notna()
            & result["ig_renyi_h_2.0"].notna()
            & result["ig_renyi_h_10.0"].notna()
        )
        subset = result.loc[valid_mask]
        assert len(subset) > 0, "Need valid rows to test ordering"
        # Allow small numerical tolerance
        tol = 1e-6
        assert (subset["ig_renyi_h_0.5"] >= subset["ig_renyi_h_2.0"] - tol).all()
        assert (subset["ig_renyi_h_2.0"] >= subset["ig_renyi_h_10.0"] - tol).all()


class TestMissingCloseRaises:
    def test_missing_close_raises(self, config: FeatureConfig) -> None:
        df = pd.DataFrame({"open": [1, 2, 3], "volume": [100, 200, 300]})
        with pytest.raises(ValueError, match="close"):
            compute_information_geometry_features(df, config)


class TestValuesFinite:
    def test_values_finite(self, ohlcv_df: pd.DataFrame, config: FeatureConfig) -> None:
        result = compute_information_geometry_features(ohlcv_df, config)
        ig_cols = [c for c in result.columns if c.startswith("ig_")]
        for col in ig_cols:
            valid = result[col].dropna()
            assert np.all(np.isfinite(valid.values)), f"{col} has non-finite values"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_histogram_density_sums_to_one(self) -> None:
        values = np.random.randn(100)
        p, _ = _histogram_density(values, 20)
        assert abs(p.sum() - 1.0) < 1e-10

    def test_renyi_convergence_to_shannon(self) -> None:
        """Renyi entropy at alpha close to 1 should approximate Shannon."""
        p = np.array([0.2, 0.3, 0.1, 0.4])
        h_shannon = _renyi_entropy(p, 1.0)
        h_near = _renyi_entropy(p, 1.0001)
        assert abs(h_shannon - h_near) < 0.01

    def test_kl_zero_for_identical(self) -> None:
        p = np.array([0.25, 0.25, 0.25, 0.25])
        kl = _kl_divergence(p, p)
        assert abs(kl) < 1e-10

    def test_fisher_peaked_vs_flat(self) -> None:
        """A peaked distribution should have higher Fisher information."""
        np.random.seed(123)
        peaked = np.random.normal(0, 0.1, 500)
        flat = np.random.uniform(-3, 3, 500)
        fi_peaked = _fisher_information(peaked, 50, 0.01)
        fi_flat = _fisher_information(flat, 50, 0.01)
        assert fi_peaked > fi_flat
