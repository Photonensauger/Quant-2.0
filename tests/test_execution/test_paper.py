"""Tests for quant.execution.paper -- PaperExecutor.

All tests use synthetic orders and prices. No broker connection needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from quant.config.settings import TradingConfig
from quant.execution.paper import PaperExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executor(
    initial_capital: float = 100_000.0,
    slippage_bps: float = 5.0,
    commission_bps: float = 10.0,
) -> PaperExecutor:
    """Create a PaperExecutor with explicit config."""
    cfg = TradingConfig(
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
    )
    return PaperExecutor(config=cfg, initial_capital=initial_capital)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSubmitBuyOrder:

    def test_submit_buy_order(self) -> None:
        """A buy order fills at a slippage-adjusted price and deducts cash."""
        ex = _make_executor(slippage_bps=10.0, commission_bps=0.0)
        initial_cash = ex.cash

        fill = ex.submit_order(symbol="AAPL", side="buy", qty=10.0, price=150.0)

        assert fill["side"] == "buy"
        assert fill["symbol"] == "AAPL"
        assert fill["fill_qty"] == 10.0
        # Fill price should be slightly above the reference (buy slippage)
        assert fill["fill_price"] > 150.0
        # Cash should have decreased
        assert ex.cash < initial_cash

    def test_submit_buy_creates_long_position(self) -> None:
        """After a buy fill, a long position exists for the symbol."""
        ex = _make_executor()
        ex.submit_order(symbol="GOOG", side="buy", qty=5.0, price=100.0)

        assert "GOOG" in ex.positions
        assert ex.positions["GOOG"].side == "long"
        assert ex.positions["GOOG"].qty == pytest.approx(5.0, rel=1e-6)


class TestSubmitSellOrder:

    def test_submit_sell_order(self) -> None:
        """A sell order fills at a slippage-adjusted price and credits cash."""
        ex = _make_executor(slippage_bps=10.0, commission_bps=0.0)

        fill = ex.submit_order(symbol="TSLA", side="sell", qty=5.0, price=200.0)

        assert fill["side"] == "sell"
        assert fill["symbol"] == "TSLA"
        assert fill["fill_qty"] == 5.0
        # Fill price should be slightly below the reference (sell slippage)
        assert fill["fill_price"] < 200.0

    def test_submit_sell_creates_short_position(self) -> None:
        """A sell without an existing position opens a short."""
        ex = _make_executor()
        ex.submit_order(symbol="SPY", side="sell", qty=3.0, price=400.0)

        assert "SPY" in ex.positions
        assert ex.positions["SPY"].side == "short"


class TestSlippageApplied:

    def test_slippage_applied(self) -> None:
        """Buy fills above reference; sell fills below reference."""
        # Large slippage for visibility
        ex = _make_executor(slippage_bps=100.0, commission_bps=0.0)

        buy_fill = ex.submit_order(symbol="A", side="buy", qty=1.0, price=100.0)
        assert buy_fill["fill_price"] > 100.0

        ex2 = _make_executor(slippage_bps=100.0, commission_bps=0.0)
        sell_fill = ex2.submit_order(symbol="A", side="sell", qty=1.0, price=100.0)
        assert sell_fill["fill_price"] < 100.0

    def test_slippage_proportional(self) -> None:
        """Slippage is proportional to the basis points."""
        ex = _make_executor(slippage_bps=50.0, commission_bps=0.0)
        fill = ex.submit_order(symbol="X", side="buy", qty=1.0, price=1000.0)

        expected = 1000.0 * (1.0 + 50.0 / 10_000.0)
        assert fill["fill_price"] == pytest.approx(expected, rel=1e-8)


class TestCommissionDeducted:

    def test_commission_deducted(self) -> None:
        """Commission is computed as notional * commission_bps / 10000."""
        ex = _make_executor(slippage_bps=0.0, commission_bps=100.0)  # 1% commission
        fill = ex.submit_order(symbol="BTC", side="buy", qty=1.0, price=50_000.0)

        expected_commission = 50_000.0 * 1.0 * (100.0 / 10_000.0)
        assert fill["commission"] == pytest.approx(expected_commission, rel=1e-6)

    def test_commission_reduces_cash(self) -> None:
        """Cash deduction includes notional + commission for buys."""
        ex = _make_executor(
            initial_capital=100_000.0,
            slippage_bps=0.0,
            commission_bps=50.0,
        )
        ex.submit_order(symbol="ETH", side="buy", qty=10.0, price=3000.0)

        # notional = 10 * 3000 = 30_000
        # commission = 30_000 * 50 / 10_000 = 150
        # cash should be: 100_000 - 30_000 - 150 = 69_850
        assert ex.cash == pytest.approx(69_850.0, rel=1e-6)


class TestClosePosition:

    def test_close_position(self) -> None:
        """Closing a position removes it from the positions dict."""
        ex = _make_executor(slippage_bps=0.0, commission_bps=0.0)
        ex.submit_order(symbol="AAPL", side="buy", qty=10.0, price=150.0)
        assert "AAPL" in ex.positions

        fill = ex.close_position("AAPL", price=160.0)

        assert fill["fill_qty"] > 0
        # Position should be removed (fully closed)
        assert "AAPL" not in ex.positions

    def test_close_position_no_existing(self) -> None:
        """Closing a non-existent position returns an empty fill."""
        ex = _make_executor()
        fill = ex.close_position("NOPE", price=100.0)
        assert fill["fill_qty"] == 0.0


class TestCloseAllPositions:

    def test_close_all_positions(self) -> None:
        """close_all_positions closes every open position."""
        ex = _make_executor(slippage_bps=0.0, commission_bps=0.0)
        ex.submit_order(symbol="AAPL", side="buy", qty=5.0, price=150.0)
        ex.submit_order(symbol="GOOG", side="buy", qty=3.0, price=2800.0)
        assert len(ex.positions) == 2

        fills = ex.close_all_positions({"AAPL": 155.0, "GOOG": 2850.0})

        assert len(fills) == 2
        assert len(ex.positions) == 0


class TestPortfolioState:

    def test_portfolio_state(self) -> None:
        """get_portfolio_state returns a dict with expected keys."""
        ex = _make_executor()
        ex.submit_order(symbol="SPY", side="buy", qty=10.0, price=400.0)

        state = ex.get_portfolio_state()

        assert "positions" in state
        assert "cash" in state
        assert "equity" in state
        assert "equity_curve" in state
        assert "daily_pnl" in state
        assert "initial_capital" in state

        # Equity should be approximately initial capital (just opened, no move)
        assert state["equity"] > 0


class TestInsufficientCash:

    def test_insufficient_cash(self) -> None:
        """When cash is insufficient, the buy order is reduced or rejected."""
        ex = _make_executor(initial_capital=100.0, slippage_bps=0.0, commission_bps=0.0)

        # Try to buy 10 shares at $50 = $500 notional, but only $100 cash
        fill = ex.submit_order(symbol="EXPEN", side="buy", qty=10.0, price=50.0)

        # Either qty is reduced to fit or the fill is zero
        if fill["fill_qty"] > 0:
            assert fill["fill_qty"] < 10.0
            # Total cost should not exceed initial cash
            cost = fill["fill_price"] * fill["fill_qty"] + fill["commission"]
            assert cost <= 100.0 + 1e-6
        else:
            # Rejected entirely
            assert fill["fill_qty"] == 0.0

    def test_insufficient_cash_complete_rejection(self) -> None:
        """With near-zero cash, order should be rejected."""
        ex = _make_executor(initial_capital=0.01, slippage_bps=0.0, commission_bps=0.0)
        fill = ex.submit_order(symbol="EXPEN", side="buy", qty=100.0, price=1000.0)

        # fill_qty should be nearly zero (or zero)
        assert fill["fill_qty"] * fill["fill_price"] <= 0.02


class TestTradeLog:

    def test_trade_log(self) -> None:
        """Each submitted order appends a record to trade_log."""
        ex = _make_executor()
        assert len(ex.trade_log) == 0

        ex.submit_order(symbol="A", side="buy", qty=5.0, price=100.0)
        assert len(ex.trade_log) == 1

        ex.submit_order(symbol="B", side="sell", qty=3.0, price=200.0)
        assert len(ex.trade_log) == 2

        # Each record should have standard keys
        for record in ex.trade_log:
            assert "symbol" in record
            assert "side" in record
            assert "fill_price" in record
            assert "fill_qty" in record
            assert "commission" in record
            assert "timestamp" in record

    def test_trade_log_after_close(self) -> None:
        """Closing a position also adds a record to the trade log."""
        ex = _make_executor(slippage_bps=0.0, commission_bps=0.0)
        ex.submit_order(symbol="MSFT", side="buy", qty=10.0, price=300.0)
        assert len(ex.trade_log) == 1

        ex.close_position("MSFT", price=310.0)
        assert len(ex.trade_log) == 2

    def test_invalid_side_raises(self) -> None:
        """An invalid side raises ValueError."""
        ex = _make_executor()
        with pytest.raises(ValueError, match="side must be"):
            ex.submit_order(symbol="X", side="hold", qty=1.0, price=100.0)

    def test_invalid_qty_raises(self) -> None:
        """Non-positive quantity raises ValueError."""
        ex = _make_executor()
        with pytest.raises(ValueError, match="qty must be positive"):
            ex.submit_order(symbol="X", side="buy", qty=-1.0, price=100.0)

    def test_invalid_price_raises(self) -> None:
        """Non-positive price raises ValueError."""
        ex = _make_executor()
        with pytest.raises(ValueError, match="price must be positive"):
            ex.submit_order(symbol="X", side="buy", qty=1.0, price=0.0)
