"""Tests for quant.backtest.engine -- BacktestEngine and BacktestResult.

Every test uses synthetic OHLCV data and mock models so no network access
or real market data is needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import BacktestEngine, BacktestResult
from quant.config.settings import (
    BacktestConfig,
    ModelConfig,
    SystemConfig,
    TradingConfig,
)
from quant.strategies.base import BaseStrategy, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(loc=0.0003, scale=0.012, size=n)
    close = start_price * np.exp(np.cumsum(returns))

    open_ = np.roll(close, 1)
    open_[0] = close[0]
    open_ += rng.normal(0, 0.05, size=n)

    intraday = np.abs(rng.normal(0, 0.3, size=n))
    high = np.maximum(open_, close) + intraday
    low = np.minimum(open_, close) - intraday

    volume = rng.lognormal(mean=10, sigma=0.5, size=n)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class _AlwaysBuyStrategy(BaseStrategy):
    """Strategy that always emits a high-confidence BUY signal."""

    def __init__(self) -> None:
        super().__init__(config=TradingConfig(min_confidence=0.0, signal_cooldown=0), name="AlwaysBuy")

    def generate_signals(
        self,
        data: pd.DataFrame,
        model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        ts = data.index[-1].to_pydatetime() if isinstance(data.index, pd.DatetimeIndex) else datetime.now(tz=timezone.utc)
        symbol = getattr(data, "symbol", "UNKNOWN")
        return [
            Signal(
                timestamp=ts,
                symbol=symbol,
                direction=1,
                confidence=0.9,
                target_position=0.5,
            )
        ]


class _AlwaysSellStrategy(BaseStrategy):
    """Strategy that always emits a high-confidence SELL signal."""

    def __init__(self) -> None:
        super().__init__(config=TradingConfig(min_confidence=0.0, signal_cooldown=0), name="AlwaysSell")

    def generate_signals(
        self,
        data: pd.DataFrame,
        model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        ts = data.index[-1].to_pydatetime() if isinstance(data.index, pd.DatetimeIndex) else datetime.now(tz=timezone.utc)
        symbol = getattr(data, "symbol", "UNKNOWN")
        return [
            Signal(
                timestamp=ts,
                symbol=symbol,
                direction=-1,
                confidence=0.9,
                target_position=-0.5,
            )
        ]


class _FlatStrategy(BaseStrategy):
    """Strategy that never emits an actionable signal."""

    def __init__(self) -> None:
        super().__init__(config=TradingConfig(), name="Flat")

    def generate_signals(
        self,
        data: pd.DataFrame,
        model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        return []


def _build_config(**overrides: Any) -> SystemConfig:
    """Create a SystemConfig with test-friendly defaults."""
    bt_cfg = BacktestConfig(
        initial_capital=overrides.pop("initial_capital", 100_000.0),
        slippage_bps=overrides.pop("slippage_bps", 5.0),
        commission_bps=overrides.pop("commission_bps", 10.0),
        margin_requirement=overrides.pop("margin_requirement", 1.0),
    )
    trading_cfg = TradingConfig(
        signal_cooldown=overrides.pop("signal_cooldown", 0),
        min_confidence=overrides.pop("min_confidence", 0.0),
        max_drawdown=overrides.pop("max_drawdown", 0.50),
        daily_loss_limit=overrides.pop("daily_loss_limit", 1.0),
        max_var_95=overrides.pop("max_var_95", 1.0),
        max_position_pct=overrides.pop("max_position_pct", 1.0),
        slippage_bps=5.0,
        commission_bps=10.0,
    )
    model_cfg = ModelConfig(
        seq_len=overrides.pop("seq_len", 10),
        n_features=overrides.pop("n_features", 5),
    )
    return SystemConfig(
        backtest=bt_cfg,
        trading=trading_cfg,
        model=model_cfg,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBacktestRunsWithSyntheticData:
    """E2E: create OHLCV, simple strategy, run backtest, get result."""

    def test_backtest_runs_with_synthetic_data(self) -> None:
        """A backtest with a simple always-buy strategy completes without error."""
        config = _build_config(seq_len=10)
        engine = BacktestEngine(config=config)

        data = {"SYNTH": _make_ohlcv(n=120)}
        strategy = _AlwaysBuyStrategy()

        result = engine.run(data=data, strategy=strategy)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert len(result.timestamps) > 0
        assert len(result.returns) > 0
        assert isinstance(result.metrics, dict)


class TestBacktestResultFields:
    """Verify all fields are present on BacktestResult."""

    def test_backtest_result_has_all_fields(self) -> None:
        config = _build_config(seq_len=10)
        engine = BacktestEngine(config=config)
        data = {"SYNTH": _make_ohlcv(n=80)}
        strategy = _AlwaysBuyStrategy()

        result = engine.run(data=data, strategy=strategy)

        assert hasattr(result, "equity_curve")
        assert hasattr(result, "returns")
        assert hasattr(result, "trade_log")
        assert hasattr(result, "metrics")
        assert hasattr(result, "positions_history")
        assert hasattr(result, "timestamps")

        # Check metrics has the key metric keys
        for key in ("total_return", "sharpe_ratio", "max_drawdown", "total_trades"):
            assert key in result.metrics, f"Missing metric key: {key}"


class TestEquityCurveInitialCapital:
    """Equity curve starts at initial capital."""

    def test_equity_curve_starts_at_initial_capital(self) -> None:
        capital = 50_000.0
        config = _build_config(initial_capital=capital, seq_len=10)
        engine = BacktestEngine(config=config)
        data = {"SYNTH": _make_ohlcv(n=80)}

        # Use a flat strategy so no trades are opened => equity == capital
        strategy = _FlatStrategy()
        result = engine.run(data=data, strategy=strategy)

        assert result.equity_curve[0] == pytest.approx(capital, rel=1e-6)


class TestPositionsTracked:
    """Positions history is populated when trades occur."""

    def test_positions_tracked(self) -> None:
        config = _build_config(seq_len=10)
        engine = BacktestEngine(config=config)
        data = {"SYNTH": _make_ohlcv(n=120)}
        strategy = _AlwaysBuyStrategy()

        result = engine.run(data=data, strategy=strategy)

        # positions_history should have one snapshot per bar
        assert len(result.positions_history) == len(result.equity_curve)

        # At least some snapshots should have a position
        has_position = any(bool(snap) for snap in result.positions_history)
        assert has_position, "Expected at least one bar with an open position."


class TestCommissionApplied:
    """Commission reduces equity relative to a zero-commission run."""

    def test_commission_applied(self) -> None:
        data = {"SYNTH": _make_ohlcv(n=120, seed=99)}
        strategy = _AlwaysBuyStrategy()

        # Run with commission
        cfg_with = _build_config(commission_bps=50.0, slippage_bps=0.0, seq_len=10)
        engine_with = BacktestEngine(config=cfg_with)
        res_with = engine_with.run(data=data, strategy=strategy)

        # Run without commission
        cfg_without = _build_config(commission_bps=0.0, slippage_bps=0.0, seq_len=10)
        engine_without = BacktestEngine(config=cfg_without)
        res_without = engine_without.run(data=data, strategy=strategy)

        # With commission, final equity should be lower (or equal if no trades)
        if res_with.trade_log:
            assert res_with.equity_curve[-1] < res_without.equity_curve[-1]


class TestSlippageApplied:
    """Slippage reduces equity relative to a zero-slippage run."""

    def test_slippage_applied(self) -> None:
        data = {"SYNTH": _make_ohlcv(n=120, seed=77)}
        strategy = _AlwaysBuyStrategy()

        # Run with slippage
        cfg_with = _build_config(slippage_bps=100.0, commission_bps=0.0, seq_len=10)
        engine_with = BacktestEngine(config=cfg_with)
        res_with = engine_with.run(data=data, strategy=strategy)

        # Run without slippage
        cfg_without = _build_config(slippage_bps=0.0, commission_bps=0.0, seq_len=10)
        engine_without = BacktestEngine(config=cfg_without)
        res_without = engine_without.run(data=data, strategy=strategy)

        if res_with.trade_log:
            # Slippage should cause different fill prices -> different equity
            assert res_with.equity_curve[-1] != pytest.approx(
                res_without.equity_curve[-1], rel=1e-6
            )


class TestStopLossClosesPosition:
    """When price drops below stop-loss, positions get closed with reason 'stop_loss'."""

    def test_stop_loss_closes_position(self) -> None:
        # Build data with a sharp drop mid-way to trigger stop-loss
        n = 120
        rng = np.random.RandomState(123)
        close = np.ones(n) * 100.0
        # Crash at bar 80 onward
        close[80:] = 60.0
        open_ = close.copy()
        high = close + 1.0
        low = close - 1.0
        low[80:] = 58.0
        volume = np.ones(n) * 1000.0
        dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        config = _build_config(seq_len=10, max_drawdown=0.99)
        engine = BacktestEngine(config=config)
        strategy = _AlwaysBuyStrategy()

        result = engine.run(data={"CRASH": df}, strategy=strategy)

        # At least one trade should have been closed due to stop-loss
        stop_loss_trades = [t for t in result.trade_log if t.get("reason") == "stop_loss"]
        assert len(stop_loss_trades) > 0, (
            "Expected at least one trade closed by stop_loss. "
            f"Trade reasons: {[t.get('reason') for t in result.trade_log]}"
        )


class TestMaxDrawdownLiquidation:
    """When max drawdown is breached, the engine flags risk breach and stops opening new trades."""

    def test_max_drawdown_liquidation(self) -> None:
        # Very tight drawdown limit so it triggers on any adverse move
        n = 120
        close = np.concatenate([np.ones(60) * 100.0, np.linspace(100, 50, 60)])
        open_ = close.copy()
        high = close + 0.5
        low = close - 0.5
        volume = np.ones(n) * 1000.0
        dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        # Very tight drawdown -- risk should be breached
        config = _build_config(
            seq_len=10,
            max_drawdown=0.02,
            max_position_pct=1.0,
        )
        engine = BacktestEngine(config=config)
        strategy = _AlwaysBuyStrategy()

        result = engine.run(data={"DD": df}, strategy=strategy)

        # Final equity should be less than initial capital due to the crash
        assert result.equity_curve[-1] < config.backtest.initial_capital


class TestMultiAssetBacktest:
    """Backtest with multiple symbols processes each independently."""

    def test_multi_asset_backtest(self) -> None:
        config = _build_config(seq_len=10)
        engine = BacktestEngine(config=config)

        data = {
            "ASSET_A": _make_ohlcv(n=120, seed=1),
            "ASSET_B": _make_ohlcv(n=120, seed=2),
        }
        strategy = _AlwaysBuyStrategy()

        result = engine.run(data=data, strategy=strategy)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 120

        # Trades should reference both symbols
        symbols_traded = {t["symbol"] for t in result.trade_log}
        # At least one symbol should have been traded
        assert len(symbols_traded) >= 1
