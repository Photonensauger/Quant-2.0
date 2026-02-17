"""Tests for quant.strategies.ensemble – EnsembleStrategy."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import TradingConfig
from quant.strategies.base import BaseStrategy, Signal
from quant.strategies.ensemble import EnsembleStrategy


# ---------------------------------------------------------------------------
# Stub strategy for testing
# ---------------------------------------------------------------------------

class _StubStrategy(BaseStrategy):
    """Returns a pre-configured signal on every call."""

    def __init__(
        self,
        config: TradingConfig,
        direction: int,
        confidence: float,
        symbol: str = "BTC-USD",
    ) -> None:
        super().__init__(config, name=f"stub_{direction}_{confidence}")
        self._direction = direction
        self._confidence = confidence
        self._symbol = symbol

    def generate_signals(
        self,
        data: pd.DataFrame,
        model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        return [
            Signal(
                timestamp=self._latest_timestamp(data),
                symbol=self._symbol,
                direction=self._direction,
                confidence=self._confidence,
                target_position=self._direction * self._confidence,
            )
        ]


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_data(n_bars: int = 50) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h")
    return pd.DataFrame(
        {"close": np.linspace(100, 110, n_bars)},
        index=dates,
    )


@pytest.fixture()
def config() -> TradingConfig:
    return TradingConfig(min_confidence=0.0, signal_cooldown=0, confirmation_bars=1)


@pytest.fixture()
def data() -> pd.DataFrame:
    return _make_data()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWeightedAggregation:
    """Ensemble should produce a signal whose direction matches the weighted
    majority vote of sub-strategies."""

    def test_weighted_aggregation(self, config: TradingConfig, data: pd.DataFrame) -> None:
        # Three strategies: two long (weight 0.4 each), one short (weight 0.2)
        strategies = {
            "long_a": _StubStrategy(config, direction=1, confidence=0.9),
            "long_b": _StubStrategy(config, direction=1, confidence=0.8),
            "short_c": _StubStrategy(config, direction=-1, confidence=0.7),
        }
        ensemble = EnsembleStrategy(
            config,
            strategies=strategies,
            initial_weights={"long_a": 0.4, "long_b": 0.4, "short_c": 0.2},
        )

        signals = ensemble.generate_signals(data)
        assert len(signals) == 1
        # Weighted vote: 0.4*1 + 0.4*1 + 0.2*(-1) = 0.6 => direction = +1
        assert signals[0].direction == 1

    def test_weighted_aggregation_short(self, config: TradingConfig, data: pd.DataFrame) -> None:
        strategies = {
            "short_a": _StubStrategy(config, direction=-1, confidence=0.9),
            "short_b": _StubStrategy(config, direction=-1, confidence=0.8),
            "long_c": _StubStrategy(config, direction=1, confidence=0.5),
        }
        ensemble = EnsembleStrategy(
            config,
            strategies=strategies,
            initial_weights={"short_a": 0.5, "short_b": 0.4, "long_c": 0.1},
        )

        signals = ensemble.generate_signals(data)
        assert len(signals) >= 1
        assert signals[0].direction == -1


class TestUpdateWeightsFromSharpe:
    """update_weights should adjust model weights based on Sharpe ratios."""

    def test_update_weights_from_sharpe(self, config: TradingConfig, data: pd.DataFrame) -> None:
        strategies = {
            "model_a": _StubStrategy(config, direction=1, confidence=0.8),
            "model_b": _StubStrategy(config, direction=1, confidence=0.6),
        }
        ensemble = EnsembleStrategy(config, strategies=strategies)

        # Initially equal weights
        assert abs(ensemble.weights["model_a"] - 0.5) < 0.01
        assert abs(ensemble.weights["model_b"] - 0.5) < 0.01

        # model_a has a much higher Sharpe
        ensemble.update_weights({"model_a": 2.0, "model_b": 0.1})

        assert ensemble.weights["model_a"] > ensemble.weights["model_b"], (
            f"model_a weight ({ensemble.weights['model_a']}) should exceed "
            f"model_b weight ({ensemble.weights['model_b']})"
        )

        # Weights still sum to 1
        total = sum(ensemble.weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights should sum to 1.0, got {total}"

    def test_negative_sharpe_gets_min_weight(self, config: TradingConfig) -> None:
        strategies = {
            "good": _StubStrategy(config, direction=1, confidence=0.8),
            "bad": _StubStrategy(config, direction=1, confidence=0.6),
        }
        ensemble = EnsembleStrategy(config, strategies=strategies, min_weight=0.05)

        ensemble.update_weights({"good": 3.0, "bad": -1.0})
        # After normalisation the bad model should still contribute (not zero),
        # and should be less than the good model.  The floor is applied before
        # normalisation, so the normalised weight will be less than min_weight
        # but strictly positive.
        assert ensemble.weights["bad"] > 0.0, "Negative-Sharpe model should still have positive weight"
        assert ensemble.weights["bad"] < ensemble.weights["good"], (
            "Negative-Sharpe model should have lower weight than positive-Sharpe model"
        )


class TestStateRoundtrip:
    """get_state / load_state should preserve the ensemble's weights and settings."""

    def test_state_roundtrip(self, config: TradingConfig) -> None:
        strategies = {
            "a": _StubStrategy(config, direction=1, confidence=0.8),
            "b": _StubStrategy(config, direction=-1, confidence=0.5),
        }
        ensemble = EnsembleStrategy(
            config,
            strategies=strategies,
            name="ens_test",
            min_weight=0.1,
        )
        ensemble.update_weights({"a": 2.0, "b": 0.5})

        state = ensemble.get_state()

        # Build a second ensemble and load state
        ensemble2 = EnsembleStrategy(
            config,
            strategies=strategies,
            name="fresh",
        )
        ensemble2.load_state(state)

        assert ensemble2.name == "ens_test"
        assert ensemble2.min_weight == pytest.approx(0.1)
        for key in ("a", "b"):
            assert ensemble2.weights[key] == pytest.approx(
                ensemble.weights[key], abs=1e-6
            ), f"Weight mismatch for {key}"
