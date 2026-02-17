"""Tests for quant.strategies.mean_reversion -- MeanReversionStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import TradingConfig
from quant.strategies.mean_reversion import MeanReversionStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(
    rsi: float = 50.0,
    bb_pct: float = 0.5,
    n: int = 100,
) -> pd.DataFrame:
    """Create a DataFrame with the required columns and given last-row values."""
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "close": np.linspace(100, 110, n),
            "rsi_14": [50.0] * (n - 1) + [rsi],
            "bb_pct": [0.5] * (n - 1) + [bb_pct],
            "symbol": "BTC-USD",
        },
        index=dates,
    )
    return df


@pytest.fixture()
def config() -> TradingConfig:
    return TradingConfig(min_confidence=0.0)


@pytest.fixture()
def strategy(config: TradingConfig) -> MeanReversionStrategy:
    return MeanReversionStrategy(config, name="test_mr")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLongSignal:
    def test_long_signal_at_oversold(self, strategy: MeanReversionStrategy) -> None:
        data = _make_data(rsi=20.0, bb_pct=0.05)
        signals = strategy.generate_signals(data)
        assert len(signals) == 1
        assert signals[0].direction == 1


class TestShortSignal:
    def test_short_signal_at_overbought(self, strategy: MeanReversionStrategy) -> None:
        data = _make_data(rsi=80.0, bb_pct=0.95)
        signals = strategy.generate_signals(data)
        assert len(signals) == 1
        assert signals[0].direction == -1


class TestNoSignalNeutral:
    def test_no_signal_at_neutral(self, strategy: MeanReversionStrategy) -> None:
        data = _make_data(rsi=50.0, bb_pct=0.5)
        signals = strategy.generate_signals(data)
        assert len(signals) == 0


class TestConfidenceBounded:
    def test_confidence_in_range(self, strategy: MeanReversionStrategy) -> None:
        for rsi, bb_pct in [(15.0, 0.02), (85.0, 0.98), (25.0, 0.08)]:
            data = _make_data(rsi=rsi, bb_pct=bb_pct)
            signals = strategy.generate_signals(data)
            for sig in signals:
                assert 0.0 <= sig.confidence <= 1.0


class TestMissingColumnsEmpty:
    def test_missing_columns_returns_empty(self, strategy: MeanReversionStrategy) -> None:
        df = pd.DataFrame(
            {"close": [100, 101]},
            index=pd.date_range("2024-01-01", periods=2, freq="h"),
        )
        signals = strategy.generate_signals(df)
        assert signals == []
