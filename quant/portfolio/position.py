"""Position sizing and position state management.

Provides the ``Position`` dataclass for tracking open trades and the
``PositionSizer`` class for computing order quantities using Kelly-criterion,
volatility-adjusted (ATR), or fixed-fraction methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quant.config.settings import TradingConfig


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Immutable record of a single open position.

    Attributes
    ----------
    symbol : str
        Ticker / instrument identifier.
    side : str
        ``"long"`` or ``"short"``.
    qty : float
        Signed quantity (positive for long, positive for short -- the *side*
        field disambiguates direction).
    entry_price : float
        Average fill price at entry.
    entry_time : datetime
        Timestamp of the entry fill.
    current_price : float
        Latest market price (updated externally).
    unrealized_pnl : float
        Mark-to-market P&L (updated externally).
    stop_loss : float
        Protective stop price (0.0 = not yet set).
    take_profit : float
        Take-profit target price (0.0 = not yet set).
    """

    symbol: str
    side: str  # "long" or "short"
    qty: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    def __post_init__(self) -> None:
        if self.side not in ("long", "short"):
            raise ValueError(f"side must be 'long' or 'short', got '{self.side}'")
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")

    @property
    def notional(self) -> float:
        """Absolute notional value at the current market price."""
        price = self.current_price if self.current_price > 0 else self.entry_price
        return abs(self.qty * price)

    def update_mark(self, price: float) -> None:
        """Update the mark-to-market price and unrealised P&L."""
        self.current_price = price
        if self.side == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.qty
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.qty


# ---------------------------------------------------------------------------
# Sizing methods
# ---------------------------------------------------------------------------

class SizingMethod(str, Enum):
    """Supported position-sizing algorithms."""

    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    FIXED_FRACTION = "fixed_fraction"


class PositionSizer:
    """Compute order quantities with risk-aware sizing.

    Parameters
    ----------
    config : TradingConfig
        Trading configuration providing ``max_position_pct`` and
        ``kelly_fraction``.
    method : SizingMethod
        Default sizing method (override per-call via *method* kwarg).
    fixed_fraction : float
        Fraction of capital risked per trade when using ``FIXED_FRACTION``.
    """

    def __init__(
        self,
        config: TradingConfig | None = None,
        method: SizingMethod = SizingMethod.VOLATILITY_ADJUSTED,
        fixed_fraction: float = 0.02,
    ) -> None:
        self.config = config or TradingConfig()
        self.method = method
        self.fixed_fraction = fixed_fraction
        logger.info(
            "PositionSizer initialised | method={} | max_pos_pct={} | kelly_frac={}",
            self.method.value,
            self.config.max_position_pct,
            self.config.kelly_fraction,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        capital: float,
        price: float,
        signal: dict[str, Any],
        market_data: pd.DataFrame,
        method: SizingMethod | None = None,
    ) -> float:
        """Compute the order quantity for a given signal.

        Parameters
        ----------
        capital : float
            Available trading capital.
        price : float
            Current asset price.
        signal : dict
            Must contain at least ``"direction"`` (``1`` for long, ``-1``
            for short) and ``"confidence"`` (float in [0, 1]).  May also
            contain ``"win_rate"`` and ``"avg_win_loss_ratio"`` for
            Kelly sizing.
        market_data : pd.DataFrame
            Recent OHLCV data.  Must include a ``"close"`` column; an
            ``"atr"`` column is used when available, otherwise ATR(14)
            is computed internally.

        Returns
        -------
        float
            Signed quantity (positive = buy, negative = sell).

        Guarantees
        ----------
        ``|qty * price| <= capital * max_position_pct``
        """
        if capital <= 0 or price <= 0:
            logger.warning("Invalid capital={} or price={}; returning 0.", capital, price)
            return 0.0

        active_method = method or self.method

        dispatch = {
            SizingMethod.KELLY_CRITERION: self._kelly_criterion,
            SizingMethod.VOLATILITY_ADJUSTED: self._volatility_adjusted,
            SizingMethod.FIXED_FRACTION: self._fixed_fraction,
        }

        raw_qty = dispatch[active_method](capital, price, signal, market_data)

        # Apply direction
        direction = signal.get("direction", 1)
        signed_qty = abs(raw_qty) * (1 if direction >= 0 else -1)

        # Enforce hard cap: |qty * price| <= capital * max_position_pct
        max_notional = capital * self.config.max_position_pct
        actual_notional = abs(signed_qty * price)
        if actual_notional > max_notional:
            capped_qty = max_notional / price
            logger.debug(
                "Capping qty from {:.4f} to {:.4f} (max_notional={:.2f})",
                abs(signed_qty),
                capped_qty,
                max_notional,
            )
            signed_qty = capped_qty * (1 if direction >= 0 else -1)

        # Floor to zero if negligibly small
        if abs(signed_qty) < 1e-10:
            return 0.0

        logger.debug(
            "Sized position: method={} qty={:.6f} notional={:.2f}",
            active_method.value,
            signed_qty,
            abs(signed_qty * price),
        )
        return signed_qty

    # ------------------------------------------------------------------
    # Sizing strategies (private)
    # ------------------------------------------------------------------

    def _kelly_criterion(
        self,
        capital: float,
        price: float,
        signal: dict[str, Any],
        market_data: pd.DataFrame,
    ) -> float:
        """Kelly criterion sizing using signal confidence as win probability.

        Kelly fraction ``f* = (p * b - q) / b`` where
        * ``p`` = win probability  (signal confidence)
        * ``q`` = 1 - p
        * ``b`` = average win / average loss ratio

        We apply a fractional Kelly (``kelly_fraction`` from config) for
        safety.
        """
        p = float(signal.get("confidence", 0.5))
        p = np.clip(p, 0.01, 0.99)
        q = 1.0 - p

        # Historical win/loss ratio -- provided by the signal or default
        b = float(signal.get("avg_win_loss_ratio", 1.5))
        if b <= 0:
            b = 1.0

        kelly_f = (p * b - q) / b
        kelly_f = max(kelly_f, 0.0)  # never bet negative

        # Apply fractional Kelly
        fraction = kelly_f * self.config.kelly_fraction
        alloc = capital * fraction
        qty = alloc / price

        logger.debug(
            "Kelly: p={:.2f} b={:.2f} f*={:.4f} frac_kelly={:.4f} qty={:.4f}",
            p,
            b,
            kelly_f,
            fraction,
            qty,
        )
        return qty

    def _volatility_adjusted(
        self,
        capital: float,
        price: float,
        signal: dict[str, Any],
        market_data: pd.DataFrame,
    ) -> float:
        """ATR-based sizing: risk a fixed fraction of capital per ATR unit.

        ``qty = (capital * risk_fraction) / (ATR * atr_multiplier * price)``

        where *risk_fraction* is scaled by signal confidence.
        """
        atr = self._get_atr(market_data)
        if atr <= 0:
            logger.warning("ATR is zero; falling back to fixed fraction.")
            return self._fixed_fraction(capital, price, signal, market_data)

        confidence = float(signal.get("confidence", 0.5))
        confidence = np.clip(confidence, 0.0, 1.0)

        # Risk budget: base fraction scaled by confidence
        risk_fraction = self.fixed_fraction * confidence
        stop_distance = atr * self.config.stop_loss_atr_mult

        # Dollar risk per share
        dollar_risk = stop_distance
        if dollar_risk <= 0:
            return 0.0

        risk_budget = capital * risk_fraction
        qty = risk_budget / dollar_risk

        logger.debug(
            "VolAdj: ATR={:.4f} conf={:.2f} risk_frac={:.4f} qty={:.4f}",
            atr,
            confidence,
            risk_fraction,
            qty,
        )
        return qty

    def _fixed_fraction(
        self,
        capital: float,
        price: float,
        signal: dict[str, Any],
        _market_data: pd.DataFrame,
    ) -> float:
        """Allocate a fixed fraction of capital to the trade."""
        confidence = float(signal.get("confidence", 0.5))
        confidence = np.clip(confidence, 0.0, 1.0)

        alloc = capital * self.fixed_fraction * confidence
        qty = alloc / price

        logger.debug(
            "FixedFrac: frac={:.4f} conf={:.2f} qty={:.4f}",
            self.fixed_fraction,
            confidence,
            qty,
        )
        return qty

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_atr(market_data: pd.DataFrame, period: int = 14) -> float:
        """Extract or compute ATR from market data.

        If an ``atr`` column exists, use the latest value.  Otherwise
        compute the Average True Range over *period* bars.
        """
        if "atr" in market_data.columns:
            atr_val = market_data["atr"].dropna()
            if len(atr_val) > 0:
                return float(atr_val.iloc[-1])

        # Compute ATR from OHLC
        required = {"high", "low", "close"}
        if not required.issubset(set(market_data.columns)):
            logger.warning("Cannot compute ATR: missing OHLC columns.")
            return 0.0

        high = market_data["high"]
        low = market_data["low"]
        close = market_data["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_series = true_range.rolling(window=period, min_periods=1).mean()
        if len(atr_series.dropna()) == 0:
            return 0.0

        return float(atr_series.iloc[-1])
