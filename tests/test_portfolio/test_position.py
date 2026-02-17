"""Tests for quant.portfolio.position – Position dataclass & PositionSizer."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import TradingConfig
from quant.portfolio.position import Position, PositionSizer, SizingMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market_data(n_bars: int = 30, base_price: float = 100.0) -> pd.DataFrame:
    """Create synthetic OHLCV data with an ATR column."""
    np.random.seed(7)
    dates = pd.date_range("2024-06-01", periods=n_bars, freq="h")
    close = base_price + np.cumsum(np.random.randn(n_bars) * 0.3)
    close = np.maximum(close, 1.0)
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.005))
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n_bars) * 0.1,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100, 5000, n_bars).astype(float),
            "atr": np.full(n_bars, 2.0),  # fixed ATR for predictability
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> TradingConfig:
    return TradingConfig(
        max_position_pct=0.10,
        kelly_fraction=0.25,
        stop_loss_atr_mult=2.0,
    )


@pytest.fixture()
def sizer(config: TradingConfig) -> PositionSizer:
    return PositionSizer(config=config, method=SizingMethod.VOLATILITY_ADJUSTED, fixed_fraction=0.02)


@pytest.fixture()
def market_data() -> pd.DataFrame:
    return _make_market_data()


# ---------------------------------------------------------------------------
# Tests – Position dataclass
# ---------------------------------------------------------------------------

class TestPositionCreation:
    """Position should accept valid parameters and reject invalid ones."""

    def test_position_creation(self) -> None:
        pos = Position(
            symbol="BTC-USD",
            side="long",
            qty=1.5,
            entry_price=50_000.0,
            entry_time=datetime(2025, 1, 1, 12, 0),
        )
        assert pos.symbol == "BTC-USD"
        assert pos.side == "long"
        assert pos.qty == 1.5
        assert pos.entry_price == 50_000.0

    def test_position_invalid_side(self) -> None:
        with pytest.raises(ValueError, match="side must be"):
            Position(
                symbol="X",
                side="flat",
                qty=1.0,
                entry_price=100.0,
                entry_time=datetime.now(),
            )

    def test_position_invalid_qty(self) -> None:
        with pytest.raises(ValueError, match="qty must be positive"):
            Position(
                symbol="X",
                side="long",
                qty=-1.0,
                entry_price=100.0,
                entry_time=datetime.now(),
            )


# ---------------------------------------------------------------------------
# Tests – PositionSizer
# ---------------------------------------------------------------------------

class TestPositionSizerKelly:
    """Kelly criterion sizing should produce a reasonable positive quantity."""

    def test_position_sizer_kelly(
        self, config: TradingConfig, market_data: pd.DataFrame
    ) -> None:
        sizer = PositionSizer(config=config, method=SizingMethod.KELLY_CRITERION)
        signal = {"direction": 1, "confidence": 0.7, "avg_win_loss_ratio": 1.5}
        qty = sizer.calculate(
            capital=100_000.0, price=100.0, signal=signal, market_data=market_data
        )
        assert qty > 0.0, f"Kelly should produce positive qty for long signal, got {qty}"
        # Notional should not exceed max_position_pct
        assert abs(qty * 100.0) <= 100_000.0 * config.max_position_pct + 1e-6


class TestPositionSizerVolatilityAdjusted:
    """Volatility-adjusted sizing should scale with ATR and confidence."""

    def test_position_sizer_volatility_adjusted(
        self, config: TradingConfig, market_data: pd.DataFrame
    ) -> None:
        # Use a large max_position_pct so the hard cap does not flatten the
        # difference between high-confidence and low-confidence sizes.
        uncapped_config = TradingConfig(
            max_position_pct=1.0,          # effectively no cap
            kelly_fraction=config.kelly_fraction,
            stop_loss_atr_mult=config.stop_loss_atr_mult,
        )
        sizer = PositionSizer(
            config=uncapped_config,
            method=SizingMethod.VOLATILITY_ADJUSTED,
            fixed_fraction=0.02,
        )

        signal_hi = {"direction": 1, "confidence": 0.9}
        signal_lo = {"direction": 1, "confidence": 0.3}

        qty_hi = sizer.calculate(
            capital=100_000.0, price=100.0, signal=signal_hi, market_data=market_data
        )
        qty_lo = sizer.calculate(
            capital=100_000.0, price=100.0, signal=signal_lo, market_data=market_data
        )

        assert qty_hi > qty_lo, (
            f"Higher confidence should yield larger qty: hi={qty_hi}, lo={qty_lo}"
        )


class TestPositionSizerMaxCap:
    """Position size must be capped at max_position_pct of capital."""

    def test_position_sizer_max_cap(
        self, sizer: PositionSizer, market_data: pd.DataFrame
    ) -> None:
        signal = {"direction": 1, "confidence": 1.0, "avg_win_loss_ratio": 10.0}
        capital = 100_000.0
        price = 10.0  # cheap asset -> large qty if uncapped

        # Try Kelly to get potentially large qty
        sizer_kelly = PositionSizer(
            config=sizer.config, method=SizingMethod.KELLY_CRITERION
        )
        qty = sizer_kelly.calculate(
            capital=capital, price=price, signal=signal, market_data=market_data
        )
        max_notional = capital * sizer.config.max_position_pct
        actual_notional = abs(qty * price)
        assert actual_notional <= max_notional + 1e-6, (
            f"Notional {actual_notional} exceeds cap {max_notional}"
        )


class TestPositionSizerNegativeSignal:
    """A short signal (direction=-1) should produce a negative quantity."""

    def test_position_sizer_negative_signal(
        self, sizer: PositionSizer, market_data: pd.DataFrame
    ) -> None:
        signal = {"direction": -1, "confidence": 0.7}
        qty = sizer.calculate(
            capital=100_000.0, price=100.0, signal=signal, market_data=market_data
        )
        assert qty < 0.0, f"Short signal should produce negative qty, got {qty}"


class TestFixedFraction:
    """Fixed-fraction sizing should be proportional to capital * fraction * confidence."""

    def test_fixed_fraction(
        self, config: TradingConfig, market_data: pd.DataFrame
    ) -> None:
        fixed_frac = 0.05
        sizer = PositionSizer(
            config=config, method=SizingMethod.FIXED_FRACTION, fixed_fraction=fixed_frac
        )
        signal = {"direction": 1, "confidence": 0.5}
        capital = 100_000.0
        price = 50.0

        qty = sizer.calculate(
            capital=capital, price=price, signal=signal, market_data=market_data
        )
        # Expected: capital * fixed_frac * confidence / price = 100000*0.05*0.5/50 = 50
        expected = capital * fixed_frac * 0.5 / price
        # May be capped by max_position_pct
        max_qty = capital * config.max_position_pct / price
        expected_capped = min(expected, max_qty)

        assert abs(qty - expected_capped) < 1.0, (
            f"Fixed fraction qty={qty}, expected ~{expected_capped}"
        )


class TestZeroQtyForFlatSignal:
    """A signal with zero confidence should produce zero quantity."""

    def test_zero_qty_for_flat_signal(
        self, sizer: PositionSizer, market_data: pd.DataFrame
    ) -> None:
        signal = {"direction": 1, "confidence": 0.0}
        qty = sizer.calculate(
            capital=100_000.0, price=100.0, signal=signal, market_data=market_data
        )
        assert qty == 0.0, f"Zero confidence should yield zero qty, got {qty}"
