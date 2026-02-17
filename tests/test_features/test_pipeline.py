"""Tests for quant.features.pipeline -- end-to-end feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import FeatureConfig
from quant.features.pipeline import FeaturePipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with a DatetimeIndex."""
    np.random.seed(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")

    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1.0)

    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    low = np.maximum(low, 0.5)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(100, 10_000, size=n).astype(float)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeaturePipeline:
    """Tests for FeaturePipeline."""

    def test_fit_transform_shapes(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """fit_transform returns (features, targets) with matching row counts."""
        pipe = FeaturePipeline()
        features, targets = pipe.fit_transform(sample_ohlcv_df)

        # 2-D feature matrix
        assert features.ndim == 2
        # 1-D target vector
        assert targets.ndim == 1
        # Same number of valid rows
        assert features.shape[0] == targets.shape[0]
        # At least some rows survived
        assert features.shape[0] > 0
        # Reasonable number of features
        assert features.shape[1] >= 10

    def test_no_nan_in_output(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Output arrays must be completely NaN-free."""
        pipe = FeaturePipeline()
        features, targets = pipe.fit_transform(sample_ohlcv_df)

        assert not np.any(np.isnan(features)), "Features contain NaN"
        assert not np.any(np.isnan(targets)), "Targets contain NaN"

    def test_targets_are_log_returns(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Targets should be forward log-returns with reasonable magnitude."""
        pipe = FeaturePipeline()
        _, targets = pipe.fit_transform(sample_ohlcv_df)

        # Log-returns of a random walk with small increments should be small
        assert np.all(np.isfinite(targets)), "Targets have non-finite values"
        # For 5-bar forward returns on small-step data, absolute values
        # should generally stay < 1.0
        assert np.abs(targets).max() < 1.0, (
            f"Target magnitude too large: {np.abs(targets).max():.4f}"
        )

    def test_correlation_filter(self) -> None:
        """Lowering the correlation threshold should drop more features."""
        df = _make_ohlcv(300)

        pipe_loose = FeaturePipeline(
            FeatureConfig(correlation_threshold=0.99)
        )
        features_loose, _ = pipe_loose.fit_transform(df)

        pipe_tight = FeaturePipeline(
            FeatureConfig(correlation_threshold=0.50)
        )
        features_tight, _ = pipe_tight.fit_transform(df)

        # Tighter threshold should yield fewer (or equal) features
        assert features_tight.shape[1] <= features_loose.shape[1], (
            f"Tight filter ({features_tight.shape[1]} cols) should have "
            f"<= loose filter ({features_loose.shape[1]} cols)"
        )

    def test_z_score_normalization(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Normalised features should have roughly bounded values."""
        pipe = FeaturePipeline()
        features, _ = pipe.fit_transform(sample_ohlcv_df)

        # After z-score normalisation the column means should be
        # reasonably close to zero (rolling z-score is not perfect centering)
        col_means = np.abs(features.mean(axis=0))
        assert col_means.mean() < 5.0, (
            f"Mean of absolute column means is {col_means.mean():.2f}, "
            f"expected < 5.0"
        )

    def test_transform_uses_saved_state(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """transform() on new data should work after fit_transform."""
        pipe = FeaturePipeline()
        features_fit, _ = pipe.fit_transform(sample_ohlcv_df)

        # Transform on same data should succeed and have same feature count
        features_tf, targets_tf = pipe.transform(sample_ohlcv_df)

        assert features_tf.shape[1] == features_fit.shape[1]
        assert features_tf.shape[0] > 0
        assert targets_tf.shape[0] == features_tf.shape[0]

    def test_state_roundtrip(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """get_state -> load_state must produce a working pipeline."""
        pipe = FeaturePipeline()
        pipe.fit_transform(sample_ohlcv_df)

        state = pipe.get_state()

        # Restore into a fresh pipeline
        pipe2 = FeaturePipeline()
        pipe2.load_state(state)

        assert pipe2.feature_names == pipe.feature_names
        assert pipe2._fitted is True

        # transform on new data using the restored pipeline
        df2 = _make_ohlcv(300, seed=99)
        features2, targets2 = pipe2.transform(df2)
        assert features2.shape[0] > 0
        assert features2.shape[1] == len(pipe.feature_names)

    def test_feature_names_tracked(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """After fit_transform the pipeline should expose feature names."""
        pipe = FeaturePipeline()
        features, _ = pipe.fit_transform(sample_ohlcv_df)

        names = pipe.feature_names
        assert isinstance(names, list)
        assert len(names) == features.shape[1]
        assert pipe.n_features == features.shape[1]

        # No OHLCV names should appear in feature names
        ohlcv_names = {"open", "high", "low", "close", "volume"}
        assert not ohlcv_names & set(names), (
            "OHLCV columns leaked into feature names"
        )

    def test_short_data_returns_empty(self) -> None:
        """Very short data (fewer rows than warmup) returns empty arrays."""
        np.random.seed(0)
        n = 5  # much shorter than warmup + forecast horizon
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        close = np.array([100.0, 100.5, 101.0, 100.8, 101.2])
        df = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": [1000.0] * n,
            },
            index=idx,
        )

        pipe = FeaturePipeline()
        features, targets = pipe.fit_transform(df)

        # With only 5 rows, NaN warmup + forecast horizon should leave
        # zero valid rows
        assert features.shape[0] == targets.shape[0]
        assert features.shape[0] == 0

    def test_transform_before_fit_raises(self) -> None:
        """Calling transform before fit_transform should raise RuntimeError."""
        pipe = FeaturePipeline()
        df = _make_ohlcv(100)
        with pytest.raises(RuntimeError, match="not been fitted"):
            pipe.transform(df)
