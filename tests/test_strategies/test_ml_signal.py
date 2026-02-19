"""Tests for quant.strategies.ml_signal – MLSignalStrategy."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import TradingConfig
from quant.strategies.ml_signal import MLSignalStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(
    n_bars: int = 300,
    base_price: float = 100.0,
    symbol: str = "BTC-USD",
) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with a DatetimeIndex."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")
    close = base_price + np.cumsum(np.random.randn(n_bars) * 0.5)
    close = np.maximum(close, 1.0)  # keep positive
    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n_bars) * 0.001),
            "high": close * (1 + np.abs(np.random.randn(n_bars) * 0.005)),
            "low": close * (1 - np.abs(np.random.randn(n_bars) * 0.005)),
            "close": close,
            "volume": np.random.randint(100, 10_000, n_bars).astype(float),
            "symbol": symbol,
        },
        index=dates,
    )
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> TradingConfig:
    return TradingConfig(
        signal_cooldown=2,
        confirmation_bars=2,
        min_confidence=0.3,
    )


@pytest.fixture()
def strategy(config: TradingConfig) -> MLSignalStrategy:
    return MLSignalStrategy(config, name="test_ml", lookback_std=50)


@pytest.fixture()
def data() -> pd.DataFrame:
    return _make_price_data(n_bars=300)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGeneratesSignalFromPredictions:
    """Given a strong prediction, the strategy should emit a signal."""

    def test_generates_signal_from_predictions(
        self, strategy: MLSignalStrategy, data: pd.DataFrame
    ) -> None:
        # Use a very large prediction to guarantee high confidence
        predictions = np.array([0.10, 0.08, 0.12, 0.09, 0.11])

        # The first call starts the confirmation counter.
        # We need `confirmation_bars` consecutive calls with the same direction.
        # config.confirmation_bars = 2, so 2 consecutive calls should suffice.
        strategy.generate_signals(data, predictions)
        signals = strategy.generate_signals(data, predictions)

        assert len(signals) >= 1, "Should emit at least one signal"
        sig = signals[0]
        assert sig.direction == 1
        assert sig.confidence > 0.0


class TestCooldownBlocksRapidSignals:
    """After emitting a signal, cooldown should suppress the next few calls."""

    def test_cooldown_blocks_rapid_signals(
        self, strategy: MLSignalStrategy, data: pd.DataFrame
    ) -> None:
        predictions = np.array([0.10, 0.10, 0.10, 0.10, 0.10])

        # Build confirmation and emit
        strategy.generate_signals(data, predictions)
        first = strategy.generate_signals(data, predictions)
        assert len(first) >= 1, "First signal should be emitted"

        # Immediately try again -- should be suppressed by cooldown
        suppressed = strategy.generate_signals(data, predictions)
        assert len(suppressed) == 0, (
            "Second signal should be blocked by cooldown"
        )


class TestConfirmationRequired:
    """A signal should not be emitted until the direction persists for
    ``confirmation_bars`` consecutive bars."""

    def test_confirmation_required(
        self, strategy: MLSignalStrategy, data: pd.DataFrame
    ) -> None:
        # Alternating signs prevent confirmation
        pos_pred = np.array([0.10, 0.10, 0.10])
        neg_pred = np.array([-0.10, -0.10, -0.10])

        strategy.reset()
        signals_a = strategy.generate_signals(data, pos_pred)
        signals_b = strategy.generate_signals(data, neg_pred)
        signals_c = strategy.generate_signals(data, pos_pred)

        # None of these should emit because direction keeps flipping,
        # so the confirmation counter never reaches confirmation_bars=2.
        assert len(signals_a) == 0, "First call should not yet emit (needs confirmation)"
        assert len(signals_b) == 0, "Direction flipped, reset confirmation"
        assert len(signals_c) == 0, "Direction flipped again, reset confirmation"


class TestMinConfidenceFilter:
    """Predictions with very low magnitude should be filtered by min_confidence."""

    def test_min_confidence_filter(
        self, strategy: MLSignalStrategy, data: pd.DataFrame
    ) -> None:
        # Extremely tiny prediction -> confidence ~ 0
        tiny_pred = np.array([1e-9, 1e-9, 1e-9])

        strategy.reset()
        # Even with confirmation, the confidence should be too low
        strategy.generate_signals(data, tiny_pred)
        signals = strategy.generate_signals(data, tiny_pred)

        assert len(signals) == 0, (
            "Tiny prediction should produce confidence below min_confidence"
        )


class TestNanPredictionsReturnsEmpty:
    """NaN/Inf predictions must not crash but return an empty signal list."""

    def test_nan_predictions_returns_empty(
        self, strategy: MLSignalStrategy, data: pd.DataFrame
    ) -> None:
        nan_pred = np.array([np.nan, 0.05, np.inf])
        signals = strategy.generate_signals(data, nan_pred)
        assert signals == [], "NaN/Inf predictions must produce no signals"


class TestSignalDirectionFromPredictionSign:
    """Negative mean prediction should yield direction == -1."""

    def test_signal_direction_from_prediction_sign(
        self, config: TradingConfig, data: pd.DataFrame
    ) -> None:
        # Lower min_confidence and set confirmation_bars=1 to make it easy to emit
        config_easy = TradingConfig(
            signal_cooldown=0,
            confirmation_bars=1,
            min_confidence=0.0,
        )
        strategy = MLSignalStrategy(config_easy, name="test_dir", lookback_std=50)

        neg_pred = np.array([-0.10, -0.08, -0.12])
        signals = strategy.generate_signals(data, neg_pred)

        assert len(signals) >= 1
        assert signals[0].direction == -1, (
            f"Expected direction -1 for negative prediction, got {signals[0].direction}"
        )
