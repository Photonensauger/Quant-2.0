"""Tests for quant.data.dataset TimeSeriesDataset and WalkForwardSplitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quant.data.dataset import TimeSeriesDataset, WalkForwardSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_features_and_target(
    n: int = 200, n_features: int = 10
) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic feature DataFrame and target Series."""
    np.random.seed(99)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    features = pd.DataFrame(
        np.random.randn(n, n_features),
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    target = pd.Series(np.random.randn(n), index=dates, name="target")
    return features, target


# ---------------------------------------------------------------------------
# TimeSeriesDataset tests
# ---------------------------------------------------------------------------
class TestDatasetShapes:
    def test_dataset_shapes(self) -> None:
        """Dataset samples should have correct tensor shapes."""
        seq_len = 30
        forecast_horizon = 5
        n_features = 10
        n = 200

        features, target = _make_features_and_target(n, n_features)
        ds = TimeSeriesDataset(
            features, target, seq_len=seq_len, forecast_horizon=forecast_horizon
        )

        expected_n_samples = n - seq_len - forecast_horizon + 1
        assert len(ds) == expected_n_samples

        x, y = ds[0]
        assert x.shape == (seq_len, n_features)
        assert y.shape == (forecast_horizon,)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32


class TestDatasetSlidingWindow:
    def test_dataset_sliding_window(self) -> None:
        """Consecutive samples should slide by exactly 1 timestep."""
        seq_len = 10
        forecast_horizon = 3
        n_features = 5
        n = 50

        features, target = _make_features_and_target(n, n_features)
        ds = TimeSeriesDataset(
            features, target, seq_len=seq_len, forecast_horizon=forecast_horizon
        )

        # Sample 0 uses features[0:10], sample 1 uses features[1:11]
        x0, _ = ds[0]
        x1, _ = ds[1]

        # The last (seq_len - 1) rows of x0 should equal the first (seq_len - 1) rows of x1
        torch.testing.assert_close(x0[1:], x1[:-1])


class TestWalkForwardSplitNoOverlap:
    def test_walk_forward_split_no_overlap(self) -> None:
        """Train and test ranges within a split should not overlap."""
        splitter = WalkForwardSplitter(n_splits=5, test_ratio=0.2, gap_size=10)
        splits = splitter.split(1000)

        assert len(splits) > 0

        for train_range, test_range in splits:
            train_indices = set(train_range)
            test_indices = set(test_range)
            assert train_indices.isdisjoint(test_indices), (
                "Train and test indices must not overlap"
            )
            # Train should come before test
            assert max(train_range) < min(test_range)


class TestWalkForwardGap:
    def test_walk_forward_gap(self) -> None:
        """The gap between train end and test start should be >= gap_size."""
        gap_size = 15
        splitter = WalkForwardSplitter(n_splits=3, test_ratio=0.2, gap_size=gap_size)
        splits = splitter.split(500)

        assert len(splits) > 0

        for train_range, test_range in splits:
            train_end = max(train_range)  # last train index (exclusive would be +1)
            test_start = min(test_range)
            actual_gap = test_start - train_end
            assert actual_gap >= gap_size, (
                f"Gap {actual_gap} is less than required gap_size {gap_size}"
            )


class TestDatasetEmptyData:
    def test_dataset_empty_data(self) -> None:
        """Dataset with fewer rows than seq_len + forecast_horizon should have 0 samples."""
        seq_len = 60
        forecast_horizon = 5
        n_features = 10
        n = 20  # too few rows

        features, target = _make_features_and_target(n, n_features)
        ds = TimeSeriesDataset(
            features, target, seq_len=seq_len, forecast_horizon=forecast_horizon
        )

        assert len(ds) == 0
