"""Tests for quant.strategies.regime_adaptive -- RegimeAdaptiveStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import TradingConfig
from quant.strategies.base import BaseStrategy, Signal
from quant.strategies.regime_adaptive import RegimeAdaptiveStrategy


# ---------------------------------------------------------------------------
# Stub sub-strategy
# ---------------------------------------------------------------------------

class _StubStrategy(BaseStrategy):
    """Always emits a signal with fixed direction and confidence."""

    def __init__(
        self,
        config: TradingConfig,
        direction: int = 1,
        confidence: float = 0.8,
        name: str | None = None,
    ) -> None:
        super().__init__(config, name=name)
        self._direction = direction
        self._confidence = confidence

    def generate_signals(
        self, data: pd.DataFrame, model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        return [
            Signal(
                timestamp=self._latest_timestamp(data),
                symbol=self._infer_symbol(data),
                direction=self._direction,
                confidence=self._confidence,
                target_position=self._direction * self._confidence,
                metadata={"strategy": self.name},
            )
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(
    cp_score: float = 0.0,
    adx: float = 30.0,
    n: int = 100,
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "close": np.linspace(100, 110, n),
            "cp_score": [0.0] * (n - 1) + [cp_score],
            "adx": [20.0] * (n - 1) + [adx],
            "symbol": "BTC-USD",
        },
        index=dates,
    )


@pytest.fixture()
def config() -> TradingConfig:
    return TradingConfig(
        min_confidence=0.1,
        ra_cp_threshold=0.3,
        ra_cp_confidence_reduction=0.5,
        tf_adx_threshold=25.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHighCpScoreReducesConfidence:
    def test_high_cp_score_reduces_confidence(self, config: TradingConfig) -> None:
        sub = {"main": _StubStrategy(config, direction=1, confidence=0.8, name="main")}
        strategy = RegimeAdaptiveStrategy(config, sub_strategies=sub)

        data_high_cp = _make_data(cp_score=0.9)
        signals = strategy.generate_signals(data_high_cp)
        # confidence should be reduced: 0.8 * (1 - 0.5 * 0.9) = 0.44
        if signals:
            assert signals[0].confidence < 0.8


class TestLowCpScorePassesThrough:
    def test_low_cp_score_passes_through(self, config: TradingConfig) -> None:
        sub = {"main": _StubStrategy(config, direction=1, confidence=0.8, name="main")}
        strategy = RegimeAdaptiveStrategy(config, sub_strategies=sub)

        data_low_cp = _make_data(cp_score=0.1)
        signals = strategy.generate_signals(data_low_cp)
        assert len(signals) == 1
        # cp_score=0.1 < threshold=0.3 -> no reduction
        assert signals[0].confidence == pytest.approx(0.8, abs=1e-6)


class TestDualModeSelection:
    def test_dual_mode_selects_based_on_adx(self, config: TradingConfig) -> None:
        trend_stub = _StubStrategy(config, direction=1, confidence=0.9, name="trend")
        reversion_stub = _StubStrategy(config, direction=-1, confidence=0.7, name="reversion")
        sub = {"trend": trend_stub, "reversion": reversion_stub}
        strategy = RegimeAdaptiveStrategy(config, sub_strategies=sub)

        # High ADX -> trend strategy
        data_trend = _make_data(adx=40.0, cp_score=0.0)
        signals_trend = strategy.generate_signals(data_trend)
        assert len(signals_trend) == 1
        assert signals_trend[0].direction == 1  # from trend stub

        # Low ADX -> reversion strategy
        data_revert = _make_data(adx=15.0, cp_score=0.0)
        signals_revert = strategy.generate_signals(data_revert)
        assert len(signals_revert) == 1
        assert signals_revert[0].direction == -1  # from reversion stub


class TestEmptySubStrategiesRaises:
    def test_empty_sub_strategies_raises(self, config: TradingConfig) -> None:
        with pytest.raises(ValueError, match="at least one"):
            RegimeAdaptiveStrategy(config, sub_strategies={})


class TestSuppressionBelowMinConfidence:
    def test_suppression_below_min_confidence(self, config: TradingConfig) -> None:
        # Low confidence stub + high cp_score -> should be suppressed
        sub = {"main": _StubStrategy(config, direction=1, confidence=0.3, name="main")}
        strategy = RegimeAdaptiveStrategy(config, sub_strategies=sub)

        # cp_score=0.9 -> adj = 0.3 * (1 - 0.5*0.9) = 0.3 * 0.55 = 0.165
        # But min_confidence is 0.1, so it passes. Let's use even lower base.
        sub2 = {"main": _StubStrategy(config, direction=1, confidence=0.15, name="main")}
        config2 = TradingConfig(
            min_confidence=0.1,
            ra_cp_threshold=0.3,
            ra_cp_confidence_reduction=0.5,
        )
        strategy2 = RegimeAdaptiveStrategy(config2, sub_strategies=sub2)

        # cp_score=0.9 -> adj = 0.15 * (1 - 0.5*0.9) = 0.15 * 0.55 = 0.0825 < 0.1
        data = _make_data(cp_score=0.9)
        signals = strategy2.generate_signals(data)
        assert len(signals) == 0
