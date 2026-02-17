"""Portfolio-level and position-level risk management.

Provides real-time risk checks (drawdown, daily loss, VaR, concentration)
and per-position stop-loss / take-profit logic based on ATR multiples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from quant.config.settings import TradingConfig

# Re-use the Position dataclass defined in the position module.  We import
# it lazily (TYPE_CHECKING) to avoid circular imports at runtime, but the
# public helpers accept any object with the required attributes.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant.portfolio.position import Position


@dataclass
class RiskCheckResult:
    """Structured output of a full risk check."""

    is_ok: bool
    violations: list[str] = field(default_factory=list)


class RiskManager:
    """Portfolio risk manager enforcing hard limits from TradingConfig.

    Parameters
    ----------
    config : TradingConfig
        Trading configuration containing all risk thresholds.
    """

    def __init__(self, config: TradingConfig | None = None) -> None:
        self.config = config or TradingConfig()
        logger.info(
            "RiskManager initialised | max_dd={} | daily_limit={} | VaR95={} | max_pos_pct={}",
            self.config.max_drawdown,
            self.config.daily_loss_limit,
            self.config.max_var_95,
            self.config.max_position_pct,
        )

    # ------------------------------------------------------------------
    # Public API  -  portfolio-level
    # ------------------------------------------------------------------

    def check_all(self, portfolio_state: dict[str, Any]) -> tuple[bool, list[str]]:
        """Run every portfolio-level risk check.

        Parameters
        ----------
        portfolio_state : dict
            Expected keys:
                positions       - dict[str, Position-like]  (symbol -> position)
                cash            - float
                equity          - float (current portfolio value)
                equity_curve    - list[float] (historical equity values)
                daily_pnl       - float (realised + unrealised P&L today)
                returns_history - np.ndarray (daily return series)

        Returns
        -------
        (bool, list[str])
            ``(True, [])`` when all checks pass, or ``(False, violations)``
            where *violations* lists human-readable descriptions.
        """
        violations: list[str] = []

        violations.extend(self._check_max_drawdown(portfolio_state))
        violations.extend(self._check_daily_loss_limit(portfolio_state))
        violations.extend(self._check_var_95(portfolio_state))
        violations.extend(self._check_position_concentration(portfolio_state))

        is_ok = len(violations) == 0

        if not is_ok:
            logger.warning("Risk violations detected: {}", violations)
        else:
            logger.debug("All risk checks passed.")

        return is_ok, violations

    # ------------------------------------------------------------------
    # Public API  -  position-level
    # ------------------------------------------------------------------

    def check_stop_loss(
        self,
        position: "Position",
        current_price: float,
        atr: float,
    ) -> bool:
        """Return True if *position* should be stopped out.

        If ``position.stop_loss`` is already set (> 0), it is used directly.
        Otherwise the stop is computed as ``entry_price -/+ stop_loss_atr_mult * ATR``
        depending on the position side.
        """
        stop = position.stop_loss
        if stop <= 0.0:
            mult = self.config.stop_loss_atr_mult
            if position.side == "long":
                stop = position.entry_price - mult * atr
            else:
                stop = position.entry_price + mult * atr

        triggered = (
            (position.side == "long" and current_price <= stop)
            or (position.side == "short" and current_price >= stop)
        )

        if triggered:
            logger.info(
                "Stop-loss triggered for {} {} @ {:.4f} (stop={:.4f}, ATR={:.4f})",
                position.side,
                position.symbol,
                current_price,
                stop,
                atr,
            )
        return triggered

    def check_take_profit(
        self,
        position: "Position",
        current_price: float,
        atr: float,
    ) -> bool:
        """Return True if *position* has hit its take-profit target.

        If ``position.take_profit`` is already set (> 0), it is used directly.
        Otherwise the target is ``entry_price +/- take_profit_atr_mult * ATR``.
        """
        tp = position.take_profit
        if tp <= 0.0:
            mult = self.config.take_profit_atr_mult
            if position.side == "long":
                tp = position.entry_price + mult * atr
            else:
                tp = position.entry_price - mult * atr

        triggered = (
            (position.side == "long" and current_price >= tp)
            or (position.side == "short" and current_price <= tp)
        )

        if triggered:
            logger.info(
                "Take-profit triggered for {} {} @ {:.4f} (tp={:.4f}, ATR={:.4f})",
                position.side,
                position.symbol,
                current_price,
                tp,
                atr,
            )
        return triggered

    # ------------------------------------------------------------------
    # Individual risk checks (private)
    # ------------------------------------------------------------------

    def _check_max_drawdown(self, state: dict[str, Any]) -> list[str]:
        """Check if portfolio drawdown exceeds the configured limit."""
        equity_curve: list[float] = state.get("equity_curve", [])
        if len(equity_curve) < 2:
            return []

        curve = np.asarray(equity_curve, dtype=np.float64)
        running_max = np.maximum.accumulate(curve)
        drawdowns = (running_max - curve) / np.where(running_max > 0, running_max, 1.0)
        current_dd = float(drawdowns[-1])

        if current_dd > self.config.max_drawdown:
            msg = (
                f"Max drawdown breached: {current_dd:.2%} > "
                f"limit {self.config.max_drawdown:.2%}"
            )
            logger.error(msg)
            return [msg]
        return []

    def _check_daily_loss_limit(self, state: dict[str, Any]) -> list[str]:
        """Check if today's P&L loss exceeds the daily limit."""
        daily_pnl: float = state.get("daily_pnl", 0.0)
        equity: float = state.get("equity", 0.0)

        if equity <= 0.0:
            return ["Equity is zero or negative."]

        daily_loss_pct = -daily_pnl / equity if daily_pnl < 0 else 0.0

        if daily_loss_pct > self.config.daily_loss_limit:
            msg = (
                f"Daily loss limit breached: {daily_loss_pct:.2%} > "
                f"limit {self.config.daily_loss_limit:.2%}"
            )
            logger.error(msg)
            return [msg]
        return []

    def _check_var_95(self, state: dict[str, Any]) -> list[str]:
        """Check if the historical 95% VaR exceeds the configured limit.

        VaR is estimated as the 5th percentile of the *returns_history*.
        """
        returns_history: NDArray | None = state.get("returns_history")
        if returns_history is None or len(returns_history) < 20:
            # Not enough data to estimate VaR reliably.
            return []

        returns = np.asarray(returns_history, dtype=np.float64)
        var_95 = float(-np.percentile(returns, 5))  # positive number = loss

        if var_95 > self.config.max_var_95:
            msg = (
                f"VaR(95%) breached: {var_95:.2%} > "
                f"limit {self.config.max_var_95:.2%}"
            )
            logger.error(msg)
            return [msg]
        return []

    def _check_position_concentration(self, state: dict[str, Any]) -> list[str]:
        """Check if any single position exceeds the max allocation."""
        positions: dict = state.get("positions", {})
        equity: float = state.get("equity", 0.0)
        violations: list[str] = []

        if equity <= 0.0 or not positions:
            return violations

        for symbol, pos in positions.items():
            # Support both Position dataclass and plain dict
            if hasattr(pos, "qty") and hasattr(pos, "current_price"):
                notional = abs(pos.qty * pos.current_price)
            elif isinstance(pos, dict):
                notional = abs(pos.get("qty", 0) * pos.get("current_price", 0))
            else:
                continue

            concentration = notional / equity
            if concentration > self.config.max_position_pct:
                msg = (
                    f"Position concentration breached for {symbol}: "
                    f"{concentration:.2%} > limit {self.config.max_position_pct:.2%}"
                )
                logger.error(msg)
                violations.append(msg)

        return violations
