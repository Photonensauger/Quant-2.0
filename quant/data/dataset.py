"""PyTorch datasets and walk-forward splitting for time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset over a feature DataFrame.

    Each sample returns:
        x: FloatTensor [seq_len, n_features]  -- input window
        y: FloatTensor [forecast_horizon]      -- target values
    """

    def __init__(
        self,
        features: pd.DataFrame,
        target: pd.Series | np.ndarray,
        seq_len: int = 60,
        forecast_horizon: int = 5,
    ) -> None:
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

        self.features = torch.tensor(
            features.values if isinstance(features, pd.DataFrame) else features,
            dtype=torch.float32,
        )
        self.target = torch.tensor(
            target.values if isinstance(target, pd.Series) else target,
            dtype=torch.float32,
        )

        self.n_samples = len(self.features) - self.seq_len - self.forecast_horizon + 1
        if self.n_samples <= 0:
            logger.warning(
                "Not enough data for windowing: {} rows, need >= {}",
                len(self.features),
                self.seq_len + self.forecast_horizon,
            )
            self.n_samples = 0

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_start = idx
        x_end = idx + self.seq_len
        y_start = x_end
        y_end = y_start + self.forecast_horizon

        x = self.features[x_start:x_end]       # [seq_len, n_features]
        y = self.target[y_start:y_end]           # [forecast_horizon]

        return x, y


@dataclass
class WalkForwardSplitter:
    """Expanding-window walk-forward cross-validator.

    Produces (train_range, test_range) pairs where each train window
    expands and a gap separates train from test to prevent leakage.

    Guarantee: for each (train, test) pair, max(train) + gap_size < min(test).
    """

    n_splits: int = 5
    test_ratio: float = 0.2
    gap_size: int = 10

    def split(self, n_total: int) -> list[tuple[range, range]]:
        """Generate walk-forward train/test index ranges.

        Args:
            n_total: Total number of samples.

        Returns:
            List of (train_range, test_range) tuples.
        """
        test_size = max(1, int(n_total * self.test_ratio / self.n_splits))
        splits: list[tuple[range, range]] = []

        for i in range(self.n_splits):
            test_end = n_total - (self.n_splits - 1 - i) * test_size
            test_start = test_end - test_size
            train_end = test_start - self.gap_size

            if train_end <= 0:
                logger.warning(
                    "Split {}/{}: train_end={} <= 0, skipping",
                    i + 1,
                    self.n_splits,
                    train_end,
                )
                continue

            train_range = range(0, train_end)
            test_range = range(test_start, test_end)

            splits.append((train_range, test_range))
            logger.debug(
                "Split {}/{}: train [0, {}), test [{}, {})",
                i + 1,
                self.n_splits,
                train_end,
                test_start,
                test_end,
            )

        if not splits:
            logger.error(
                "No valid splits generated for n_total={}, n_splits={}, gap={}",
                n_total,
                self.n_splits,
                self.gap_size,
            )

        return splits
