"""Tests for quant.strategies.trend_following -- TrendFollowingStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import TradingConfig
from quant.strategies.trend_following import TrendFollowingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(
    macd: float = 0.5,
    macd_signal: float = 0.0,
    adx: float = 30.0,
    n: int = 100,
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "close": np.linspace(100, 110, n),
            "macd": [0.0] * (n - 1) + [macd],
            "macd_signal": [0.0] * (n - 1) + [macd_signal],
            "adx": [20.0] * (n - 1) + [adx],
            "symbol": "BTC-USD",
        },
        index=dates,
    )


@pytest.fixture()
def config() -> TradingConfig:
    return TradingConfig(min_confidence=0.0)


@pytest.fixture()
def strategy(config: TradingConfig) -> TrendFollowingStrategy:
    return TrendFollowingStrategy(config, name="test_tf")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLongSignal:
    def test_long_when_macd_above_signal_and_adx_high(
        self, strategy: TrendFollowingStrategy,
    ) -> None:
        data = _make_data(macd=1.0, macd_signal=0.5, adx=40.0)
        signals = strategy.generate_signals(data)
        assert len(signals) == 1
        assert signals[0].direction == 1


class TestShortSignal:
    def test_short_when_macd_below_signal_and_adx_high(
        self, strategy: TrendFollowingStrategy,
    ) -> None:
        data = _make_data(macd=-1.0, macd_signal=0.5, adx=40.0)
        signals = strategy.generate_signals(data)
        assert len(signals) == 1
        assert signals[0].direction == -1


class TestNoSignalLowAdx:
    def test_no_signal_when_adx_below_threshold(
        self, strategy: TrendFollowingStrategy,
    ) -> None:
        data = _make_data(macd=1.0, macd_signal=0.5, adx=15.0)
        signals = strategy.generate_signals(data)
        assert len(signals) == 0


class TestConfidenceScalesWithAdx:
    def test_confidence_increases_with_adx(
        self, strategy: TrendFollowingStrategy,
    ) -> None:
        data_low = _make_data(macd=1.0, macd_signal=0.5, adx=30.0)
        data_high = _make_data(macd=1.0, macd_signal=0.5, adx=60.0)
        sig_low = strategy.generate_signals(data_low)
        sig_high = strategy.generate_signals(data_high)
        assert len(sig_low) == 1 and len(sig_high) == 1
        assert sig_high[0].confidence > sig_low[0].confidence


class TestMissingColumnsEmpty:
    def test_missing_columns_returns_empty(
        self, strategy: TrendFollowingStrategy,
    ) -> None:
        df = pd.DataFrame(
            {"close": [100, 101]},
            index=pd.date_range("2024-01-01", periods=2, freq="h"),
        )
        signals = strategy.generate_signals(df)
        assert signals == []
