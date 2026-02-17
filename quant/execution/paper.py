"""Paper (simulated) order execution engine.

Provides ``PaperExecutor`` which simulates order fills locally, applying
configurable slippage and commission.  Intended for backtesting, paper
trading, and strategy development without a live broker connection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from quant.config.settings import TradingConfig
from quant.portfolio.position import Position


class PaperExecutor:
    """Simulated order executor that fills orders instantly with modelled
    slippage and commission costs.

    Parameters
    ----------
    config : TradingConfig | None
        Trading configuration.  Slippage and commission are read from
        ``slippage_bps`` and ``commission_bps`` respectively.
    initial_capital : float
        Starting cash balance (default 100 000).
    """

    def __init__(
        self,
        config: TradingConfig | None = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.config = config or TradingConfig()
        self.initial_capital = initial_capital
        self.cash: float = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_log: list[dict[str, Any]] = []
        self._equity_curve: list[float] = [initial_capital]
        self._daily_pnl: float = 0.0

        # Pre-compute multipliers from basis-point config
        self._slippage_mult = self.config.slippage_bps / 10_000.0
        self._commission_mult = self.config.commission_bps / 10_000.0

        logger.info(
            "PaperExecutor initialised | capital={:,.2f} | slippage={:.1f}bps | commission={:.1f}bps",
            initial_capital,
            self.config.slippage_bps,
            self.config.commission_bps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> dict[str, Any]:
        """Simulate filling an order at *price* with slippage and commission.

        Parameters
        ----------
        symbol : str
            Ticker / instrument identifier.
        side : str
            ``"buy"`` or ``"sell"``.
        qty : float
            Unsigned order quantity (always positive).
        price : float
            Reference market price at which the order is submitted.

        Returns
        -------
        dict
            ``{"fill_price": float, "fill_qty": float,
              "commission": float, "timestamp": datetime,
              "symbol": str, "side": str}``

        Raises
        ------
        ValueError
            If *side* is not ``"buy"`` / ``"sell"`` or *qty* / *price*
            are non-positive.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")

        # --- Apply slippage ---
        # Buys fill slightly above reference; sells slightly below.
        if side == "buy":
            fill_price = price * (1.0 + self._slippage_mult)
        else:
            fill_price = price * (1.0 - self._slippage_mult)

        fill_qty = qty

        # --- Commission ---
        notional = fill_price * fill_qty
        commission = notional * self._commission_mult

        # --- Check cash sufficiency for buys ---
        if side == "buy":
            total_cost = notional + commission
            if total_cost > self.cash:
                # Reduce quantity to what cash allows
                affordable_notional = self.cash / (1.0 + self._commission_mult)
                fill_qty = affordable_notional / fill_price
                if fill_qty <= 0:
                    logger.warning(
                        "Insufficient cash for {} {} @ {:.4f} (cash={:.2f})",
                        symbol,
                        side,
                        price,
                        self.cash,
                    )
                    return self._empty_fill(symbol, side)
                notional = fill_price * fill_qty
                commission = notional * self._commission_mult
                total_cost = notional + commission
                logger.warning(
                    "Order size reduced: requested {:.4f}, filled {:.4f} "
                    "(cash constraint)",
                    qty,
                    fill_qty,
                )

        timestamp = datetime.now(tz=timezone.utc)

        # --- Update internal state ---
        self._apply_fill(symbol, side, fill_qty, fill_price, commission)

        fill_record: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "fill_price": fill_price,
            "fill_qty": fill_qty,
            "commission": commission,
            "timestamp": timestamp,
        }
        self.trade_log.append(fill_record)

        logger.info(
            "PAPER FILL | {} {} {:.4f} @ {:.4f} | commission={:.4f} | cash={:,.2f}",
            side.upper(),
            symbol,
            fill_qty,
            fill_price,
            commission,
            self.cash,
        )

        return fill_record

    def close_position(
        self,
        symbol: str,
        price: float | None = None,
    ) -> dict[str, Any]:
        """Close an existing position for *symbol*.

        Parameters
        ----------
        symbol : str
            The instrument whose position should be closed.
        price : float | None
            Reference price for the closing fill.  If ``None``, the
            position's ``current_price`` is used (it must have been set
            via ``update_mark`` or an earlier trade).

        Returns
        -------
        dict
            Fill record identical in structure to ``submit_order`` output.
            Returns an empty-fill dict if no position exists.
        """
        if symbol not in self.positions:
            logger.warning("No open position for {} to close.", symbol)
            return self._empty_fill(symbol, "sell")

        pos = self.positions[symbol]
        close_price = price if price is not None else pos.current_price
        if close_price <= 0:
            close_price = pos.entry_price
            logger.debug(
                "current_price unavailable for {}; using entry_price={:.4f}",
                symbol,
                close_price,
            )

        # Opposite side to close
        close_side = "sell" if pos.side == "long" else "buy"
        return self.submit_order(symbol, close_side, pos.qty, close_price)

    def close_all_positions(
        self,
        current_prices: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Close every open position using the supplied prices.

        Parameters
        ----------
        current_prices : dict[str, float]
            Mapping of symbol to latest market price.

        Returns
        -------
        list[dict]
            List of fill records, one per closed position.
        """
        fills: list[dict[str, Any]] = []
        # Snapshot symbols to avoid mutating dict during iteration
        symbols = list(self.positions.keys())

        for symbol in symbols:
            price = current_prices.get(symbol)
            if price is None:
                logger.warning(
                    "No price provided for {} in close_all_positions; "
                    "using position's current_price.",
                    symbol,
                )
            fills.append(self.close_position(symbol, price))

        logger.info(
            "Closed all positions | {} fills | cash={:,.2f}",
            len(fills),
            self.cash,
        )
        return fills

    def get_portfolio_state(self) -> dict[str, Any]:
        """Build a portfolio-state dict compatible with ``RiskManager.check_all``.

        Returns
        -------
        dict
            Keys: ``positions``, ``cash``, ``equity``, ``equity_curve``,
            ``daily_pnl``, ``initial_capital``.
        """
        equity = self._compute_equity()

        return {
            "positions": dict(self.positions),
            "cash": self.cash,
            "equity": equity,
            "equity_curve": list(self._equity_curve),
            "daily_pnl": self._daily_pnl,
            "initial_capital": self.initial_capital,
        }

    def update_market_prices(self, prices: dict[str, float]) -> None:
        """Mark all open positions to the latest market prices.

        Parameters
        ----------
        prices : dict[str, float]
            Mapping of symbol to latest price.
        """
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_mark(prices[symbol])

        equity = self._compute_equity()
        self._equity_curve.append(equity)

    def reset_daily_pnl(self) -> None:
        """Reset the intraday P&L accumulator (call at start of each day)."""
        self._daily_pnl = 0.0
        logger.debug("Daily P&L reset.")

    def reset(self) -> None:
        """Reset the executor to its initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trade_log.clear()
        self._equity_curve = [self.initial_capital]
        self._daily_pnl = 0.0
        logger.info("PaperExecutor reset to initial state.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
        commission: float,
    ) -> None:
        """Update positions and cash for a completed fill."""
        if side == "buy":
            self.cash -= fill_price * qty + commission
            self._open_or_add(symbol, "long", qty, fill_price)
        else:  # sell
            self.cash += fill_price * qty - commission
            self._close_or_reduce(symbol, "short", qty, fill_price)

        # Track daily P&L (negative of commission is an immediate cost)
        self._daily_pnl -= commission

    def _open_or_add(
        self,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
    ) -> None:
        """Open a new long position or add to an existing one.

        If a short position exists for *symbol*, reduce/close it first
        (covering), then open a long with any residual quantity.
        """
        if symbol in self.positions:
            existing = self.positions[symbol]

            if existing.side == side:
                # Average into the existing position
                total_qty = existing.qty + qty
                avg_price = (
                    (existing.entry_price * existing.qty + fill_price * qty) / total_qty
                )
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    qty=total_qty,
                    entry_price=avg_price,
                    entry_time=existing.entry_time,
                    current_price=fill_price,
                    stop_loss=existing.stop_loss,
                    take_profit=existing.take_profit,
                )
                return

            # Opposite side: reduce or flip
            if qty < existing.qty:
                # Partial close of existing short
                realized = (existing.entry_price - fill_price) * qty
                self._daily_pnl += realized
                remaining = existing.qty - qty
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=existing.side,
                    qty=remaining,
                    entry_price=existing.entry_price,
                    entry_time=existing.entry_time,
                    current_price=fill_price,
                    stop_loss=existing.stop_loss,
                    take_profit=existing.take_profit,
                )
                return

            # Full close (and possibly flip)
            realized = (existing.entry_price - fill_price) * existing.qty
            self._daily_pnl += realized
            residual = qty - existing.qty
            del self.positions[symbol]

            if residual > 0:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    qty=residual,
                    entry_price=fill_price,
                    entry_time=datetime.now(tz=timezone.utc),
                    current_price=fill_price,
                )
            return

        # No existing position -- open new
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=fill_price,
            entry_time=datetime.now(tz=timezone.utc),
            current_price=fill_price,
        )

    def _close_or_reduce(
        self,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
    ) -> None:
        """Close or reduce a long position, or open/add to a short.

        A *sell* against an existing long reduces/closes it; if the sell
        quantity exceeds the long, the residual opens a short.
        """
        if symbol in self.positions:
            existing = self.positions[symbol]

            if existing.side == "long":
                # Selling against a long position
                if qty < existing.qty:
                    realized = (fill_price - existing.entry_price) * qty
                    self._daily_pnl += realized
                    remaining = existing.qty - qty
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side="long",
                        qty=remaining,
                        entry_price=existing.entry_price,
                        entry_time=existing.entry_time,
                        current_price=fill_price,
                        stop_loss=existing.stop_loss,
                        take_profit=existing.take_profit,
                    )
                    return

                # Full close (and possibly flip to short)
                realized = (fill_price - existing.entry_price) * existing.qty
                self._daily_pnl += realized
                residual = qty - existing.qty
                del self.positions[symbol]

                if residual > 0:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side="short",
                        qty=residual,
                        entry_price=fill_price,
                        entry_time=datetime.now(tz=timezone.utc),
                        current_price=fill_price,
                    )
                return

            # Existing position is also short -- add to it
            total_qty = existing.qty + qty
            avg_price = (
                (existing.entry_price * existing.qty + fill_price * qty) / total_qty
            )
            self.positions[symbol] = Position(
                symbol=symbol,
                side="short",
                qty=total_qty,
                entry_price=avg_price,
                entry_time=existing.entry_time,
                current_price=fill_price,
                stop_loss=existing.stop_loss,
                take_profit=existing.take_profit,
            )
            return

        # No existing position -- open new short
        self.positions[symbol] = Position(
            symbol=symbol,
            side="short",
            qty=qty,
            entry_price=fill_price,
            entry_time=datetime.now(tz=timezone.utc),
            current_price=fill_price,
        )

    def _compute_equity(self) -> float:
        """Compute total portfolio equity = cash + sum(position notional * direction)."""
        unrealized = 0.0
        for pos in self.positions.values():
            price = pos.current_price if pos.current_price > 0 else pos.entry_price
            if pos.side == "long":
                unrealized += (price - pos.entry_price) * pos.qty
            else:
                unrealized += (pos.entry_price - price) * pos.qty
        return self.cash + unrealized

    def _empty_fill(self, symbol: str, side: str) -> dict[str, Any]:
        """Return a zero-quantity fill record (used when an order cannot execute)."""
        return {
            "symbol": symbol,
            "side": side,
            "fill_price": 0.0,
            "fill_qty": 0.0,
            "commission": 0.0,
            "timestamp": datetime.now(tz=timezone.utc),
        }
