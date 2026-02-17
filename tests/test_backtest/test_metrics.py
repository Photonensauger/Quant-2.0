"""Tests for quant.backtest.metrics -- compute_metrics and internal helpers.

All tests use synthetic equity curves and trade logs. No external data.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from quant.backtest.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade_log(
    pnls: list[float],
    base_time: datetime | None = None,
    duration_seconds: float = 3600.0,
) -> list[dict]:
    """Build a synthetic trade log from a list of P&L values."""
    base = base_time or datetime(2024, 6, 1, 10, 0, 0)
    log: list[dict] = []
    for i, pnl in enumerate(pnls):
        entry_time = base + timedelta(hours=i)
        exit_time = entry_time + timedelta(seconds=duration_seconds)
        log.append(
            {
                "pnl": pnl,
                "side": "long" if pnl >= 0 else "short",
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": 100.0,
                "exit_price": 100.0 + pnl / 10.0,
                "qty": 10.0,
            }
        )
    return log


def _make_equity_curve(
    initial: float,
    returns: list[float],
) -> list[float]:
    """Build an equity curve from an initial value and per-bar simple returns."""
    curve = [initial]
    for r in returns:
        curve.append(curve[-1] * (1.0 + r))
    return curve


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTotalReturn:

    def test_total_return(self) -> None:
        """Total return = (final - initial) / initial."""
        initial = 100_000.0
        # 10% total return
        equity = _make_equity_curve(initial, [0.01] * 10)
        trade_log = _make_trade_log([100.0, 200.0])

        metrics = compute_metrics(equity, trade_log, initial)

        expected = (equity[-1] - initial) / initial
        assert metrics["total_return"] == pytest.approx(expected, rel=1e-6)

    def test_total_return_negative(self) -> None:
        """Negative total return when final equity is below initial."""
        initial = 100_000.0
        equity = _make_equity_curve(initial, [-0.05] * 5)
        trade_log = _make_trade_log([-500.0])

        metrics = compute_metrics(equity, trade_log, initial)
        assert metrics["total_return"] < 0


class TestSharpeRatio:

    def test_sharpe_ratio(self) -> None:
        """Sharpe ratio is computed and annualised."""
        initial = 100_000.0
        # Positive drift with noise
        rng = np.random.RandomState(42)
        rets = (rng.normal(0.001, 0.01, size=252)).tolist()
        equity = _make_equity_curve(initial, rets)
        trade_log = _make_trade_log([100.0, 200.0, -50.0])

        metrics = compute_metrics(equity, trade_log, initial)

        # Sharpe should be a finite number and positive given positive drift
        assert np.isfinite(metrics["sharpe_ratio"])
        assert metrics["sharpe_ratio"] > 0

    def test_sharpe_ratio_zero_variance(self) -> None:
        """When returns are constant, Sharpe should be 0 (no std)."""
        initial = 100_000.0
        equity = _make_equity_curve(initial, [0.0] * 100)
        trade_log: list[dict] = []

        metrics = compute_metrics(equity, trade_log, initial)
        assert metrics["sharpe_ratio"] == 0.0


class TestSortinoRatio:

    def test_sortino_ratio(self) -> None:
        """Sortino ratio is computed using downside deviation."""
        initial = 100_000.0
        rng = np.random.RandomState(7)
        rets = (rng.normal(0.001, 0.01, size=252)).tolist()
        equity = _make_equity_curve(initial, rets)
        trade_log = _make_trade_log([100.0, -50.0])

        metrics = compute_metrics(equity, trade_log, initial)

        assert np.isfinite(metrics["sortino_ratio"])

    def test_sortino_ratio_all_positive_returns(self) -> None:
        """Sortino should be large (capped at 100) when there are no negative returns."""
        initial = 100_000.0
        equity = _make_equity_curve(initial, [0.01] * 50)
        trade_log: list[dict] = []

        metrics = compute_metrics(equity, trade_log, initial)
        # Should be 100 (capped) or positive
        assert metrics["sortino_ratio"] > 0


class TestMaxDrawdown:

    def test_max_drawdown(self) -> None:
        """Max drawdown is computed as the largest peak-to-trough fraction."""
        initial = 100_000.0
        # Up 10%, then crash 30% from peak, then recover
        rets = [0.10, -0.15, -0.15, 0.05, 0.05]
        equity = _make_equity_curve(initial, rets)
        trade_log: list[dict] = []

        metrics = compute_metrics(equity, trade_log, initial)

        # Max drawdown should be positive and less than 1
        assert 0 < metrics["max_drawdown"] < 1.0

        # Sanity-check: compute expected drawdown manually
        eq_arr = np.array(equity)
        running_max = np.maximum.accumulate(eq_arr)
        dd = (running_max - eq_arr) / running_max
        expected_dd = float(np.max(dd))
        assert metrics["max_drawdown"] == pytest.approx(expected_dd, rel=1e-6)

    def test_max_drawdown_monotonic_up(self) -> None:
        """When equity only goes up, drawdown should be 0."""
        initial = 100_000.0
        equity = _make_equity_curve(initial, [0.01] * 50)
        trade_log: list[dict] = []

        metrics = compute_metrics(equity, trade_log, initial)
        assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-10)


class TestWinRate:

    def test_win_rate(self) -> None:
        """Win rate = number of winning trades / total trades."""
        initial = 100_000.0
        equity = _make_equity_curve(initial, [0.01] * 10)
        pnls = [100.0, 200.0, -50.0, 300.0, -100.0]  # 3 wins, 2 losses
        trade_log = _make_trade_log(pnls)

        metrics = compute_metrics(equity, trade_log, initial)

        expected_win_rate = 3 / 5
        assert metrics["win_rate"] == pytest.approx(expected_win_rate, rel=1e-6)


class TestProfitFactor:

    def test_profit_factor(self) -> None:
        """Profit factor = gross profit / |gross loss|."""
        initial = 100_000.0
        equity = _make_equity_curve(initial, [0.01] * 10)
        pnls = [200.0, 300.0, -100.0, -50.0]
        trade_log = _make_trade_log(pnls)

        metrics = compute_metrics(equity, trade_log, initial)

        gross_profit = 200.0 + 300.0
        gross_loss = 100.0 + 50.0
        expected_pf = gross_profit / gross_loss
        assert metrics["profit_factor"] == pytest.approx(expected_pf, rel=1e-6)

    def test_profit_factor_no_losses(self) -> None:
        """When there are no losses, profit factor should be 0 (division guard)."""
        initial = 100_000.0
        equity = _make_equity_curve(initial, [0.01] * 10)
        trade_log = _make_trade_log([100.0, 200.0])

        metrics = compute_metrics(equity, trade_log, initial)
        # gross_loss is 0 => profit_factor = 0 per code's guard
        assert metrics["profit_factor"] == 0.0


class TestCalmarRatio:

    def test_calmar_ratio(self) -> None:
        """Calmar ratio = annualized return / max drawdown."""
        initial = 100_000.0
        rng = np.random.RandomState(10)
        rets = (rng.normal(0.002, 0.015, size=252)).tolist()
        equity = _make_equity_curve(initial, rets)
        trade_log = _make_trade_log([100.0, -50.0])

        metrics = compute_metrics(equity, trade_log, initial)

        assert np.isfinite(metrics["calmar_ratio"])
        # If there was a drawdown and positive return, Calmar should be positive
        if metrics["max_drawdown"] > 0 and metrics["annualized_return"] > 0:
            assert metrics["calmar_ratio"] > 0


class TestAllMetricsKeysPresent:

    def test_all_metrics_keys_present(self) -> None:
        """Verify all expected metric keys are present in the output."""
        initial = 100_000.0
        equity = _make_equity_curve(initial, [0.01] * 50)
        trade_log = _make_trade_log([100.0, -50.0, 200.0])

        metrics = compute_metrics(equity, trade_log, initial)

        expected_keys = {
            "initial_capital",
            "final_equity",
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "max_drawdown_duration",
            "total_trades",
            "win_rate",
            "profit_factor",
            "avg_win",
            "avg_loss",
            "expectancy",
            "avg_trade_duration",
        }
        assert expected_keys.issubset(set(metrics.keys()))


class TestZeroTrades:

    def test_zero_trades(self) -> None:
        """When there are no trades, trade-level metrics should be zero."""
        initial = 100_000.0
        equity = _make_equity_curve(initial, [0.01] * 20)
        trade_log: list[dict] = []

        metrics = compute_metrics(equity, trade_log, initial)

        assert metrics["total_trades"] == 0.0
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0
        assert metrics["avg_win"] == 0.0
        assert metrics["avg_loss"] == 0.0
        assert metrics["expectancy"] == 0.0


class TestFlatEquity:

    def test_flat_equity(self) -> None:
        """A perfectly flat equity curve should yield zero return and zero Sharpe."""
        initial = 100_000.0
        equity = [initial] * 100
        trade_log: list[dict] = []

        metrics = compute_metrics(equity, trade_log, initial)

        assert metrics["total_return"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-10)
