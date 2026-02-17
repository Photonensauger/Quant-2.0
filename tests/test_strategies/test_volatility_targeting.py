"""Tests for quant.strategies.volatility_targeting -- VolatilityTargetingStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import TradingConfig
from quant.strategies.volatility_targeting import VolatilityTargetingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(volatility: float = 0.01, n: int = 100, trend: float = 0.001) -> pd.DataFrame:
    """Create data with controlled volatility.

    Parameters
    ----------
    volatility : float
        Std dev of log-returns (per bar).
    n : int
        Number of bars.
    trend : float
        Mean log-return per bar (controls last-return direction).
    """
    np.random.seed(42)
    log_rets = np.random.normal(trend, volatility, n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"close": close, "symbol": "BTC-USD"},
        index=dates,
    )


@pytest.fixture()
def config() -> TradingConfig:
    return TradingConfig(
        min_confidence=0.0,
        vt_target_vol=0.10,
        vt_vol_lookback=20,
        vt_max_leverage=2.0,
    )


@pytest.fixture()
def strategy(config: TradingConfig) -> VolatilityTargetingStrategy:
    return VolatilityTargetingStrategy(config, name="test_vt")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLowVolHigherPosition:
    def test_low_vol_leads_to_higher_position(
        self, strategy: VolatilityTargetingStrategy,
    ) -> None:
        data_low = _make_data(volatility=0.002, trend=0.001)
        data_high = _make_data(volatility=0.02, trend=0.001)
        sig_low = strategy.generate_signals(data_low)
        sig_high = strategy.generate_signals(data_high)
        assert len(sig_low) >= 1 and len(sig_high) >= 1
        assert abs(sig_low[0].target_position) >= abs(sig_high[0].target_position)


class TestHighVolLowerPosition:
    def test_high_vol_leads_to_lower_position(
        self, strategy: VolatilityTargetingStrategy,
    ) -> None:
        data = _make_data(volatility=0.05, trend=0.001)
        signals = strategy.generate_signals(data)
        assert len(signals) >= 1
        # With high vol, scale should be small
        assert abs(signals[0].target_position) < 1.0


class TestDirectionFromPredictions:
    def test_direction_follows_model_predictions(
        self, strategy: VolatilityTargetingStrategy,
    ) -> None:
        data = _make_data(volatility=0.01, trend=0.001)
        neg_preds = np.array([-0.05, -0.03, -0.04])
        signals = strategy.generate_signals(data, model_predictions=neg_preds)
        assert len(signals) >= 1
        assert signals[0].direction == -1


class TestScaleCappedAtMaxLeverage:
    def test_scale_capped(self, config: TradingConfig) -> None:
        # Very low vol -> scale would exceed max_leverage without cap
        strategy = VolatilityTargetingStrategy(config, name="test_vt_cap")
        data = _make_data(volatility=0.0005, trend=0.001)
        signals = strategy.generate_signals(data)
        assert len(signals) >= 1
        assert abs(signals[0].target_position) <= 1.0
        assert signals[0].metadata["scale"] <= config.vt_max_leverage


class TestTooLittleData:
    def test_too_little_data_returns_empty(
        self, strategy: VolatilityTargetingStrategy,
    ) -> None:
        # Only 5 rows, but need lookback=20 + 1
        data = _make_data(n=5)
        signals = strategy.generate_signals(data)
        assert signals == []
