"""Tests for quant.portfolio.risk – RiskManager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from quant.config.settings import TradingConfig
from quant.portfolio.risk import RiskManager


# ---------------------------------------------------------------------------
# Position-like stub (avoids importing the real Position to stay self-contained)
# ---------------------------------------------------------------------------

@dataclass
class _FakePosition:
    symbol: str
    side: str
    qty: float
    entry_price: float
    current_price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> TradingConfig:
    return TradingConfig(
        max_drawdown=0.15,
        daily_loss_limit=0.03,
        max_var_95=0.05,
        max_position_pct=0.10,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=3.0,
    )


@pytest.fixture()
def risk_mgr(config: TradingConfig) -> RiskManager:
    return RiskManager(config)


def _healthy_state() -> dict[str, Any]:
    """Portfolio state that passes all checks."""
    return {
        "positions": {},
        "cash": 90_000.0,
        "equity": 100_000.0,
        "equity_curve": [100_000.0, 100_500.0, 101_000.0],
        "daily_pnl": 200.0,
        "returns_history": np.random.default_rng(0).normal(0.001, 0.005, 100),
    }


# ---------------------------------------------------------------------------
# Tests – portfolio level
# ---------------------------------------------------------------------------

class TestCheckAllPassesHealthy:
    """A healthy portfolio state should pass all risk checks."""

    def test_check_all_passes_healthy(self, risk_mgr: RiskManager) -> None:
        state = _healthy_state()
        is_ok, violations = risk_mgr.check_all(state)
        assert is_ok is True
        assert violations == []


class TestMaxDrawdownViolation:
    """Drawdown exceeding the limit should be flagged."""

    def test_max_drawdown_violation(self, risk_mgr: RiskManager) -> None:
        state = _healthy_state()
        # Simulate a drawdown from 100k peak to 80k = 20% drawdown > 15% limit
        state["equity_curve"] = [100_000.0, 100_000.0, 80_000.0]

        is_ok, violations = risk_mgr.check_all(state)
        assert is_ok is False
        assert any("drawdown" in v.lower() for v in violations)


class TestDailyLossViolation:
    """Daily loss exceeding the limit should be flagged."""

    def test_daily_loss_violation(self, risk_mgr: RiskManager) -> None:
        state = _healthy_state()
        # -5000 on 100k equity = 5% > 3% limit
        state["daily_pnl"] = -5000.0
        state["equity"] = 100_000.0

        is_ok, violations = risk_mgr.check_all(state)
        assert is_ok is False
        assert any("daily loss" in v.lower() for v in violations)


class TestVarViolation:
    """VaR exceeding the limit should be flagged."""

    def test_var_violation(self, risk_mgr: RiskManager) -> None:
        state = _healthy_state()
        # Returns with heavy left tail -- 5th percentile should be large loss
        rng = np.random.default_rng(42)
        bad_returns = rng.normal(-0.02, 0.04, 200)  # mean -2%, vol 4%
        state["returns_history"] = bad_returns

        is_ok, violations = risk_mgr.check_all(state)
        assert is_ok is False
        assert any("var" in v.lower() for v in violations)


class TestConcentrationViolation:
    """A single position consuming > max_position_pct of equity should be flagged."""

    def test_concentration_violation(self, risk_mgr: RiskManager) -> None:
        state = _healthy_state()
        # Position worth 15k out of 100k equity = 15% > 10% limit
        state["positions"] = {
            "BTC-USD": _FakePosition(
                symbol="BTC-USD",
                side="long",
                qty=1.5,
                entry_price=10_000.0,
                current_price=10_000.0,
            ),
        }
        state["equity"] = 100_000.0

        is_ok, violations = risk_mgr.check_all(state)
        assert is_ok is False
        assert any("concentration" in v.lower() for v in violations)


# ---------------------------------------------------------------------------
# Tests – position level
# ---------------------------------------------------------------------------

class TestStopLossTrigger:
    """check_stop_loss should trigger when price falls below the stop."""

    def test_stop_loss_trigger_long(self, risk_mgr: RiskManager) -> None:
        pos = _FakePosition(
            symbol="BTC-USD", side="long", qty=1.0,
            entry_price=100.0, current_price=90.0,
        )
        atr = 5.0
        # stop = 100 - 2*5 = 90 -> current_price (90) <= stop (90) -> triggered
        assert risk_mgr.check_stop_loss(pos, current_price=90.0, atr=atr) is True

    def test_stop_loss_no_trigger_long(self, risk_mgr: RiskManager) -> None:
        pos = _FakePosition(
            symbol="BTC-USD", side="long", qty=1.0,
            entry_price=100.0, current_price=95.0,
        )
        atr = 5.0
        # stop = 100 - 2*5 = 90, current_price=95 > 90 -> not triggered
        assert risk_mgr.check_stop_loss(pos, current_price=95.0, atr=atr) is False

    def test_stop_loss_trigger_short(self, risk_mgr: RiskManager) -> None:
        pos = _FakePosition(
            symbol="BTC-USD", side="short", qty=1.0,
            entry_price=100.0, current_price=110.0,
        )
        atr = 5.0
        # stop = 100 + 2*5 = 110, current_price=110 >= 110 -> triggered
        assert risk_mgr.check_stop_loss(pos, current_price=110.0, atr=atr) is True

    def test_stop_loss_explicit_level(self, risk_mgr: RiskManager) -> None:
        pos = _FakePosition(
            symbol="BTC-USD", side="long", qty=1.0,
            entry_price=100.0, current_price=94.0,
            stop_loss=95.0,
        )
        # Explicit stop_loss=95, current=94 <= 95 -> triggered
        assert risk_mgr.check_stop_loss(pos, current_price=94.0, atr=5.0) is True


class TestTakeProfitTrigger:
    """check_take_profit should trigger when price exceeds the target."""

    def test_take_profit_trigger_long(self, risk_mgr: RiskManager) -> None:
        pos = _FakePosition(
            symbol="BTC-USD", side="long", qty=1.0,
            entry_price=100.0, current_price=115.0,
        )
        atr = 5.0
        # tp = 100 + 3*5 = 115, current_price=115 >= 115 -> triggered
        assert risk_mgr.check_take_profit(pos, current_price=115.0, atr=atr) is True

    def test_take_profit_no_trigger_long(self, risk_mgr: RiskManager) -> None:
        pos = _FakePosition(
            symbol="BTC-USD", side="long", qty=1.0,
            entry_price=100.0, current_price=110.0,
        )
        atr = 5.0
        # tp = 100 + 3*5 = 115, current_price=110 < 115 -> not triggered
        assert risk_mgr.check_take_profit(pos, current_price=110.0, atr=atr) is False

    def test_take_profit_trigger_short(self, risk_mgr: RiskManager) -> None:
        pos = _FakePosition(
            symbol="BTC-USD", side="short", qty=1.0,
            entry_price=100.0, current_price=85.0,
        )
        atr = 5.0
        # tp = 100 - 3*5 = 85, current_price=85 <= 85 -> triggered
        assert risk_mgr.check_take_profit(pos, current_price=85.0, atr=atr) is True
