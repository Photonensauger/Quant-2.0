"""Live order execution engine using the Alpaca brokerage API.

Provides ``LiveExecutor`` which submits real orders to Alpaca Markets.
The module gracefully degrades if ``alpaca-trade-api`` is not installed:
all classes remain importable but raise ``RuntimeError`` on connection
attempts.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from quant.config.settings import TradingConfig

# ---------------------------------------------------------------------------
# Conditional Alpaca import
# ---------------------------------------------------------------------------
_ALPACA_AVAILABLE = False
try:
    import alpaca_trade_api as tradeapi  # type: ignore[import-untyped]

    _ALPACA_AVAILABLE = True
except ImportError:
    tradeapi = None  # type: ignore[assignment]
    logger.debug(
        "alpaca-trade-api not installed; LiveExecutor will not be able to connect."
    )


class LiveExecutor:
    """Real broker executor backed by the Alpaca Markets API.

    Parameters
    ----------
    config : TradingConfig | None
        Trading configuration.  Used for risk limits such as
        ``max_position_pct``.
    max_order_size : float
        Hard cap on the dollar notional of any single order.  Orders
        exceeding this limit are **rejected** before reaching the broker.
    max_retries : int
        Number of connection retry attempts before triggering an
        emergency stop (default 5).
    retry_delay : float
        Base delay in seconds between retry attempts (exponential
        back-off is applied: ``delay * 2^attempt``).

    Environment Variables
    ---------------------
    ALPACA_API_KEY : str
        Alpaca API key ID.
    ALPACA_SECRET_KEY : str
        Alpaca API secret key.
    ALPACA_BASE_URL : str
        Alpaca base URL (paper: ``https://paper-api.alpaca.markets``,
        live: ``https://api.alpaca.markets``).
    """

    def __init__(
        self,
        config: TradingConfig | None = None,
        max_order_size: float = 50_000.0,
        max_retries: int = 5,
        retry_delay: float = 1.0,
    ) -> None:
        self.config = config or TradingConfig()
        self.max_order_size = max_order_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._api: Any = None
        self._connected: bool = False

        # Read credentials from environment
        self._api_key = os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        self._base_url = os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        logger.info(
            "LiveExecutor initialised | max_order_size={:,.2f} | "
            "max_retries={} | base_url={}",
            self.max_order_size,
            self.max_retries,
            self._base_url,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish a connection to the Alpaca API with retry logic.

        Raises
        ------
        RuntimeError
            If ``alpaca-trade-api`` is not installed.
        ConnectionError
            If all retry attempts are exhausted.
        """
        if not _ALPACA_AVAILABLE:
            raise RuntimeError(
                "alpaca-trade-api is not installed. "
                "Install it with: pip install alpaca-trade-api"
            )

        if not self._api_key or not self._secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY and/or ALPACA_SECRET_KEY environment "
                "variables are not set."
            )

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Connecting to Alpaca (attempt {}/{})...",
                    attempt,
                    self.max_retries,
                )
                self._api = tradeapi.REST(
                    key_id=self._api_key,
                    secret_key=self._secret_key,
                    base_url=self._base_url,
                )
                # Verify connection by fetching account info
                account = self._api.get_account()
                self._connected = True
                logger.info(
                    "Connected to Alpaca | account_status={} | equity={}",
                    account.status,
                    account.equity,
                )
                return
            except Exception as exc:
                last_error = exc
                delay = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Connection attempt {}/{} failed: {} | retrying in {:.1f}s",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        # All retries exhausted -- emergency stop
        self._connected = False
        self._emergency_stop(
            f"Failed to connect after {self.max_retries} attempts: {last_error}"
        )
        raise ConnectionError(
            f"Alpaca connection failed after {self.max_retries} retries: {last_error}"
        )

    def disconnect(self) -> None:
        """Disconnect from the Alpaca API."""
        self._api = None
        self._connected = False
        logger.info("Disconnected from Alpaca.")

    @property
    def is_connected(self) -> bool:
        """Return True if the executor has an active broker connection."""
        return self._connected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """Submit an order to Alpaca.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. ``"AAPL"``).
        side : str
            ``"buy"`` or ``"sell"``.
        qty : float
            Number of shares (positive).
        order_type : str
            ``"market"``, ``"limit"``, ``"stop"``, ``"stop_limit"``.
        limit_price : float | None
            Required for limit / stop-limit orders.
        time_in_force : str
            ``"day"``, ``"gtc"``, ``"ioc"``, ``"fok"``.

        Returns
        -------
        dict
            ``{"order_id": str, "status": str, "fill_price": float,
              "symbol": str, "side": str, "qty": float,
              "order_type": str, "submitted_at": datetime}``

        Raises
        ------
        RuntimeError
            If not connected.
        ValueError
            If the order fails safety checks.
        """
        self._require_connection()

        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")

        # --- Safety: check max_order_size ---
        estimated_price = self._get_last_price(symbol)
        estimated_notional = qty * estimated_price
        if estimated_notional > self.max_order_size:
            raise ValueError(
                f"Order notional ${estimated_notional:,.2f} exceeds "
                f"max_order_size ${self.max_order_size:,.2f} for {symbol}. "
                f"Reduce qty or increase max_order_size."
            )

        logger.info(
            "Submitting order | {} {} {:.4f} {} | est_notional=${:,.2f}",
            side.upper(),
            symbol,
            qty,
            order_type,
            estimated_notional,
        )

        try:
            kwargs: dict[str, Any] = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
            }
            if limit_price is not None and order_type in ("limit", "stop_limit"):
                kwargs["limit_price"] = limit_price

            order = self._api.submit_order(**kwargs)

            result: dict[str, Any] = {
                "order_id": order.id,
                "status": order.status,
                "fill_price": float(order.filled_avg_price or 0.0),
                "symbol": symbol,
                "side": side,
                "qty": float(order.qty),
                "order_type": order_type,
                "submitted_at": datetime.now(tz=timezone.utc),
            }

            logger.info(
                "Order submitted | id={} | status={} | fill_price={:.4f}",
                result["order_id"],
                result["status"],
                result["fill_price"],
            )
            return result

        except Exception as exc:
            logger.error("Order submission failed for {} {}: {}", side, symbol, exc)
            return {
                "order_id": "",
                "status": "failed",
                "fill_price": 0.0,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": order_type,
                "submitted_at": datetime.now(tz=timezone.utc),
                "error": str(exc),
            }

    def close_position(self, symbol: str) -> dict[str, Any]:
        """Close the open position for *symbol* via the Alpaca API.

        Parameters
        ----------
        symbol : str
            Ticker whose position should be liquidated.

        Returns
        -------
        dict
            Order result from the close request, or an error dict.
        """
        self._require_connection()

        try:
            logger.info("Closing position for {}...", symbol)
            order = self._api.close_position(symbol)

            result: dict[str, Any] = {
                "order_id": order.id,
                "status": order.status,
                "fill_price": float(order.filled_avg_price or 0.0),
                "symbol": symbol,
                "side": order.side,
                "qty": float(order.qty),
                "submitted_at": datetime.now(tz=timezone.utc),
            }
            logger.info(
                "Position close submitted | {} | id={} | status={}",
                symbol,
                result["order_id"],
                result["status"],
            )
            return result

        except Exception as exc:
            logger.error("Failed to close position for {}: {}", symbol, exc)
            return {
                "order_id": "",
                "status": "failed",
                "symbol": symbol,
                "fill_price": 0.0,
                "submitted_at": datetime.now(tz=timezone.utc),
                "error": str(exc),
            }

    def close_all_positions(self) -> list[dict[str, Any]]:
        """Liquidate every open position via the Alpaca API.

        Returns
        -------
        list[dict]
            A list of order result dicts, one per position.
        """
        self._require_connection()

        results: list[dict[str, Any]] = []
        try:
            logger.info("Closing all positions...")
            close_responses = self._api.close_all_positions()

            for resp in close_responses:
                # close_all_positions returns a list of dicts or order-like objects
                if hasattr(resp, "id"):
                    results.append({
                        "order_id": resp.id,
                        "status": resp.status,
                        "symbol": resp.symbol,
                        "fill_price": float(resp.filled_avg_price or 0.0),
                        "submitted_at": datetime.now(tz=timezone.utc),
                    })
                elif isinstance(resp, dict):
                    order_body = resp.get("body", {})
                    results.append({
                        "order_id": order_body.get("id", ""),
                        "status": order_body.get("status", "unknown"),
                        "symbol": order_body.get("symbol", ""),
                        "fill_price": float(
                            order_body.get("filled_avg_price") or 0.0
                        ),
                        "submitted_at": datetime.now(tz=timezone.utc),
                    })

            logger.info("Close-all submitted | {} orders", len(results))

        except Exception as exc:
            logger.error("Failed to close all positions: {}", exc)
            results.append({
                "order_id": "",
                "status": "failed",
                "symbol": "_ALL_",
                "fill_price": 0.0,
                "submitted_at": datetime.now(tz=timezone.utc),
                "error": str(exc),
            })

        return results

    def get_portfolio_state(self) -> dict[str, Any]:
        """Fetch the current portfolio state from Alpaca.

        Returns
        -------
        dict
            Keys: ``positions``, ``cash``, ``equity``, ``buying_power``,
            ``account_status``.
        """
        self._require_connection()

        try:
            account = self._api.get_account()
            raw_positions = self._api.list_positions()

            positions: dict[str, dict[str, Any]] = {}
            for pos in raw_positions:
                positions[pos.symbol] = {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "qty": float(pos.qty),
                    "entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pnl": float(pos.unrealized_pl),
                    "unrealized_pnl_pct": float(pos.unrealized_plpc),
                }

            state: dict[str, Any] = {
                "positions": positions,
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "account_status": account.status,
                "equity_curve": [],  # Alpaca doesn't provide history here
                "daily_pnl": float(account.equity) - float(account.last_equity),
            }

            logger.debug(
                "Portfolio state | equity={:,.2f} | cash={:,.2f} | positions={}",
                state["equity"],
                state["cash"],
                len(positions),
            )
            return state

        except Exception as exc:
            logger.error("Failed to fetch portfolio state: {}", exc)
            return {
                "positions": {},
                "cash": 0.0,
                "equity": 0.0,
                "buying_power": 0.0,
                "account_status": "error",
                "equity_curve": [],
                "daily_pnl": 0.0,
                "error": str(exc),
            }

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Poll the status of an existing order.

        Parameters
        ----------
        order_id : str
            The Alpaca order ID.

        Returns
        -------
        dict
            Order state including ``status`` and ``fill_price``.
        """
        self._require_connection()

        try:
            order = self._api.get_order(order_id)
            return {
                "order_id": order.id,
                "status": order.status,
                "fill_price": float(order.filled_avg_price or 0.0),
                "filled_qty": float(order.filled_qty or 0.0),
                "symbol": order.symbol,
                "side": order.side,
                "order_type": order.type,
            }
        except Exception as exc:
            logger.error("Failed to get order status for {}: {}", order_id, exc)
            return {
                "order_id": order_id,
                "status": "error",
                "fill_price": 0.0,
                "error": str(exc),
            }

    def cancel_order(self, order_id: str) -> bool:
        """Attempt to cancel a pending order.

        Returns
        -------
        bool
            True if cancellation was accepted, False otherwise.
        """
        self._require_connection()

        try:
            self._api.cancel_order(order_id)
            logger.info("Order {} cancelled.", order_id)
            return True
        except Exception as exc:
            logger.error("Failed to cancel order {}: {}", order_id, exc)
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders.

        Returns
        -------
        bool
            True if the cancellation request was accepted.
        """
        self._require_connection()

        try:
            self._api.cancel_all_orders()
            logger.info("All open orders cancelled.")
            return True
        except Exception as exc:
            logger.error("Failed to cancel all orders: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connection(self) -> None:
        """Raise ``RuntimeError`` if not connected."""
        if not self._connected or self._api is None:
            raise RuntimeError(
                "LiveExecutor is not connected. Call connect() first."
            )

    def _get_last_price(self, symbol: str) -> float:
        """Fetch the latest trade price for *symbol* from Alpaca.

        Falls back to the latest quote midpoint if the trade endpoint
        fails.
        """
        try:
            trade = self._api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception:
            pass

        try:
            quote = self._api.get_latest_quote(symbol)
            bid = float(quote.bid_price) if quote.bid_price else 0.0
            ask = float(quote.ask_price) if quote.ask_price else 0.0
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
        except Exception:
            pass

        logger.warning(
            "Unable to fetch last price for {}; using 0.0 "
            "(safety check may reject order).",
            symbol,
        )
        return 0.0

    def _emergency_stop(self, reason: str) -> None:
        """Log an emergency-level message when connectivity is lost.

        In a production system this would trigger alerts, attempt to
        cancel open orders, and notify operators.
        """
        logger.critical(
            "EMERGENCY STOP | reason={} | "
            "All pending orders should be manually reviewed.",
            reason,
        )
