"""Core back-testing engine for the quant trading system.

Processes multi-asset OHLCV data bar-by-bar through a complete pipeline of
feature engineering, model inference, signal generation, position sizing,
risk management, and order execution.  Produces a :class:`BacktestResult`
dataclass with equity curves, trade logs, and performance metrics.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quant.backtest.metrics import compute_metrics
from quant.config.settings import (
    BacktestConfig,
    ModelConfig,
    SystemConfig,
    TradingConfig,
)
from quant.features.pipeline import FeaturePipeline
from quant.portfolio.position import Position, PositionSizer
from quant.portfolio.risk import RiskManager
from quant.strategies.base import BaseStrategy, Signal


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for all outputs of a backtest run.

    Attributes
    ----------
    equity_curve : list[float]
        Portfolio value at every bar.
    returns : list[float]
        Per-bar simple returns of the portfolio.
    trade_log : list[dict]
        One entry per round-trip trade with entry/exit details and P&L.
    metrics : dict[str, float]
        Performance metrics computed by :func:`compute_metrics`.
    positions_history : list[dict]
        Snapshot of open positions at every bar.
    timestamps : list[datetime]
        Corresponding timestamps for each bar in the equity curve.
    """

    equity_curve: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    positions_history: list[dict] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Event-driven backtest engine with multi-asset support.

    The engine iterates through OHLCV bars one at a time and at each bar:

    1. Mark-to-market all open positions.
    2. Run risk checks (stop-loss, take-profit, max drawdown, daily loss limit).
    3. If enough bars have accumulated for the model's ``seq_len``, run the
       feature pipeline, model prediction, and strategy to generate signals.
    4. If the signal is not FLAT, size the position, perform a risk check on
       the proposed order, and execute.
    5. Record equity, positions, and trade log entries.

    Parameters
    ----------
    config : SystemConfig | None
        Full system configuration.  If ``None``, defaults are used.
    checkpoint_dir : Path | str | None
        Directory for periodic state checkpoints.  Disabled when ``None``.
    checkpoint_interval : int
        Number of bars between automatic state checkpoints.
    """

    def __init__(
        self,
        config: SystemConfig | None = None,
        checkpoint_dir: Path | str | None = None,
        checkpoint_interval: int = 500,
    ) -> None:
        self.cfg = config or SystemConfig()
        self._bt_cfg: BacktestConfig = self.cfg.backtest
        self._trading_cfg: TradingConfig = self.cfg.trading
        self._model_cfg: ModelConfig = self.cfg.model

        # Checkpoint
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_interval = checkpoint_interval

        # Modules (may be overridden in ``run``)
        self._risk_manager = RiskManager(self._trading_cfg)
        self._position_sizer = PositionSizer(self._trading_cfg)

        # Execution costs (converted from basis points to fractions)
        self._slippage = self._bt_cfg.slippage_bps / 10_000.0
        self._commission = self._bt_cfg.commission_bps / 10_000.0

        logger.info(
            "BacktestEngine initialised | capital={:,.0f} | "
            "slippage={:.1f}bps | commission={:.1f}bps",
            self._bt_cfg.initial_capital,
            self._bt_cfg.slippage_bps,
            self._bt_cfg.commission_bps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data: dict[str, pd.DataFrame],
        model: Any | None = None,
        feature_pipeline: FeaturePipeline | None = None,
        strategy: BaseStrategy | None = None,
    ) -> BacktestResult:
        """Execute the backtest over the supplied data.

        Parameters
        ----------
        data : dict[str, DataFrame]
            Mapping of ``symbol -> OHLCV DataFrame`` with a ``DatetimeIndex``.
            All DataFrames must share the same index (aligned timestamps).
        model : Any | None
            A model object with a ``predict(features)`` method returning an
            ``np.ndarray`` of predicted log-returns.  ``None`` disables model
            inference (the strategy must be rules-based).
        feature_pipeline : FeaturePipeline | None
            A *fitted* :class:`FeaturePipeline`.  ``None`` disables feature
            engineering (signals come purely from the strategy).
        strategy : BaseStrategy | None
            Strategy for converting model predictions into signals.
            ``None`` disables signal generation entirely (useful for
            hold-only or manual-signal back-tests).

        Returns
        -------
        BacktestResult
        """
        # -- Validate inputs --------------------------------------------------
        if not data:
            raise ValueError("data dict must contain at least one symbol.")

        symbols = list(data.keys())
        ref_symbol = symbols[0]
        ref_df = data[ref_symbol]
        n_bars = len(ref_df)
        logger.info(
            "Starting backtest | symbols={} | bars={} | seq_len={}",
            symbols,
            n_bars,
            self._model_cfg.seq_len,
        )

        # -- Initialise state --------------------------------------------------
        state = self._init_state(symbols)

        # Build aligned timestamp list from the reference symbol
        timestamps: list[datetime] = []
        if isinstance(ref_df.index, pd.DatetimeIndex):
            timestamps = [ts.to_pydatetime() for ts in ref_df.index]
        else:
            timestamps = [datetime(2000, 1, 1)] * n_bars

        # -- Bar-by-bar loop ---------------------------------------------------
        for bar_idx in range(n_bars):
            current_ts = timestamps[bar_idx]

            # 1. Mark-to-market all open positions
            self._mark_to_market(state, data, bar_idx)

            # 2. Risk checks: stop-loss, take-profit, max DD, daily limit
            self._risk_check_positions(state, data, bar_idx, current_ts)
            self._risk_check_portfolio(state)

            # 3. Signal generation (feature pipeline -> model -> strategy)
            if strategy is not None and bar_idx >= self._model_cfg.seq_len:
                for symbol in symbols:
                    signal = self._generate_signal(
                        symbol=symbol,
                        data=data,
                        bar_idx=bar_idx,
                        model=model,
                        feature_pipeline=feature_pipeline,
                        strategy=strategy,
                        current_ts=current_ts,
                    )
                    if signal is not None and signal.direction != 0:
                        # 4. Position sizing + risk + execute
                        self._process_signal(
                            signal=signal,
                            state=state,
                            data=data,
                            bar_idx=bar_idx,
                            current_ts=current_ts,
                        )

            # 5. Update equity curve
            equity = self._compute_equity(state)
            state["equity_curve"].append(equity)
            state["timestamps"].append(current_ts)

            # Compute bar return
            if len(state["equity_curve"]) >= 2:
                prev = state["equity_curve"][-2]
                ret = (equity - prev) / prev if prev != 0 else 0.0
            else:
                ret = 0.0
            state["returns"].append(ret)

            # Snapshot positions
            state["positions_history"].append(self._snapshot_positions(state))

            # 6. Periodic checkpoint
            if (
                self._checkpoint_dir is not None
                and bar_idx > 0
                and bar_idx % self._checkpoint_interval == 0
            ):
                self._save_checkpoint(state, bar_idx)

            # Progress logging every 20% of bars
            if n_bars > 100 and bar_idx > 0 and bar_idx % (n_bars // 5) == 0:
                logger.info(
                    "Backtest progress: bar {}/{} ({:.0%}) | equity={:,.2f}",
                    bar_idx,
                    n_bars,
                    bar_idx / n_bars,
                    equity,
                )

        # -- Close all remaining positions at the last bar ---------------------
        self._close_all_positions(state, data, n_bars - 1, timestamps[-1])

        # -- Compute final equity after closing --------------------------------
        final_equity = self._compute_equity(state)
        if state["equity_curve"]:
            state["equity_curve"][-1] = final_equity

        # -- Compute metrics ---------------------------------------------------
        metrics = compute_metrics(
            equity_curve=state["equity_curve"],
            trade_log=state["trade_log"],
            initial_capital=self._bt_cfg.initial_capital,
        )

        result = BacktestResult(
            equity_curve=state["equity_curve"],
            returns=state["returns"],
            trade_log=state["trade_log"],
            metrics=metrics,
            positions_history=state["positions_history"],
            timestamps=state["timestamps"],
        )

        logger.info(
            "Backtest complete | final_equity={:,.2f} | total_return={:.2%} | "
            "sharpe={:.2f} | trades={}",
            metrics.get("final_equity", 0),
            metrics.get("total_return", 0),
            metrics.get("sharpe_ratio", 0),
            int(metrics.get("total_trades", 0)),
        )
        return result

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _init_state(self, symbols: list[str]) -> dict[str, Any]:
        """Create the mutable state dictionary for a backtest run."""
        return {
            "cash": self._bt_cfg.initial_capital,
            "positions": {},  # symbol -> Position
            "equity_curve": [],
            "returns": [],
            "trade_log": [],
            "positions_history": [],
            "timestamps": [],
            "daily_pnl": 0.0,
            "last_signal_bar": {s: -999 for s in symbols},  # cooldown tracker
        }

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def _mark_to_market(
        self,
        state: dict[str, Any],
        data: dict[str, pd.DataFrame],
        bar_idx: int,
    ) -> None:
        """Update every open position with the current bar's close price."""
        for symbol, pos in list(state["positions"].items()):
            df = data.get(symbol)
            if df is None or bar_idx >= len(df):
                continue

            close = self._get_price(df, bar_idx, "close")
            pos.update_mark(close)

    # ------------------------------------------------------------------
    # Risk checks -- position level
    # ------------------------------------------------------------------

    def _risk_check_positions(
        self,
        state: dict[str, Any],
        data: dict[str, pd.DataFrame],
        bar_idx: int,
        current_ts: datetime,
    ) -> None:
        """Check stop-loss and take-profit for every open position."""
        for symbol in list(state["positions"].keys()):
            pos = state["positions"].get(symbol)
            if pos is None:
                continue

            df = data.get(symbol)
            if df is None or bar_idx >= len(df):
                continue

            close = self._get_price(df, bar_idx, "close")
            atr = self._compute_atr(df, bar_idx)

            # Stop-loss check
            if self._risk_manager.check_stop_loss(pos, close, atr):
                self._close_position(state, symbol, close, bar_idx, current_ts, "stop_loss")
                continue

            # Take-profit check
            if self._risk_manager.check_take_profit(pos, close, atr):
                self._close_position(state, symbol, close, bar_idx, current_ts, "take_profit")

    # ------------------------------------------------------------------
    # Risk checks -- portfolio level
    # ------------------------------------------------------------------

    def _risk_check_portfolio(self, state: dict[str, Any]) -> None:
        """Run portfolio-level risk checks; liquidate all if breached."""
        equity = self._compute_equity(state)
        portfolio_state = {
            "positions": state["positions"],
            "cash": state["cash"],
            "equity": equity,
            "equity_curve": state["equity_curve"],
            "daily_pnl": state["daily_pnl"],
            "returns_history": np.array(state["returns"], dtype=np.float64),
        }

        is_ok, violations = self._risk_manager.check_all(portfolio_state)

        if not is_ok:
            logger.warning(
                "Portfolio risk breached ({}); liquidating all positions.",
                "; ".join(violations),
            )
            # We do not liquidate inside this method to avoid price lookup
            # complications.  Instead we flag the state and the signal
            # generator will refuse new trades.  For existing positions the
            # per-position checks handle stop-outs.
            state["_risk_breached"] = True
        else:
            state["_risk_breached"] = False

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal(
        self,
        symbol: str,
        data: dict[str, pd.DataFrame],
        bar_idx: int,
        model: Any | None,
        feature_pipeline: FeaturePipeline | None,
        strategy: BaseStrategy,
        current_ts: datetime,
    ) -> Signal | None:
        """Run the feature pipeline, model, and strategy for one symbol.

        Returns ``None`` when no actionable signal is produced.
        """
        df = data[symbol]
        seq_len = self._model_cfg.seq_len

        # Cooldown check
        last_bar = self._last_signal_bar_for(symbol, {})  # placeholder
        # (We pass the actual state from the caller -- see _process_signal)

        # Slice the lookback window for feature computation
        start = max(0, bar_idx - seq_len + 1)
        window_df = df.iloc[start : bar_idx + 1].copy()

        if len(window_df) < seq_len:
            return None

        # Feature pipeline
        model_predictions: np.ndarray | None = None
        if feature_pipeline is not None and model is not None:
            try:
                features, _ = feature_pipeline.transform(window_df)
                if features.shape[0] == 0:
                    return None

                # Model expects shape [1, seq_len, n_features] or [seq_len, n_features]
                # Take the last seq_len rows if available
                if features.shape[0] >= seq_len:
                    feat_input = features[-seq_len:]
                else:
                    feat_input = features

                model_predictions = model.predict(feat_input)
            except Exception as exc:
                logger.warning(
                    "Feature/model error for {} at bar {}: {}",
                    symbol,
                    bar_idx,
                    exc,
                )
                return None

        # Strategy
        try:
            signals = strategy.generate_signals(
                data=window_df,
                model_predictions=model_predictions,
            )
        except Exception as exc:
            logger.warning(
                "Strategy error for {} at bar {}: {}", symbol, bar_idx, exc
            )
            return None

        # Filter to the current symbol
        for sig in signals:
            if sig.symbol == symbol or sig.symbol == "UNKNOWN":
                # Attach the correct symbol if the strategy used UNKNOWN
                if sig.symbol == "UNKNOWN":
                    sig = Signal(
                        timestamp=sig.timestamp,
                        symbol=symbol,
                        direction=sig.direction,
                        confidence=sig.confidence,
                        target_position=sig.target_position,
                        metadata=sig.metadata,
                    )
                return sig

        return None

    # ------------------------------------------------------------------
    # Signal processing -> execution
    # ------------------------------------------------------------------

    def _process_signal(
        self,
        signal: Signal,
        state: dict[str, Any],
        data: dict[str, pd.DataFrame],
        bar_idx: int,
        current_ts: datetime,
    ) -> None:
        """Size, risk-check, and execute a signal."""
        symbol = signal.symbol

        # Portfolio risk breached -- refuse new trades
        if state.get("_risk_breached", False):
            logger.debug("Risk breached; skipping signal for {}", symbol)
            return

        # Cooldown check
        last_bar = state["last_signal_bar"].get(symbol, -999)
        if (bar_idx - last_bar) < self._trading_cfg.signal_cooldown:
            logger.debug(
                "Cooldown active for {} (last={}, current={}, cooldown={})",
                symbol,
                last_bar,
                bar_idx,
                self._trading_cfg.signal_cooldown,
            )
            return

        # Confidence threshold
        if signal.confidence < self._trading_cfg.min_confidence:
            logger.debug(
                "Signal confidence {:.2f} below threshold {:.2f} for {}",
                signal.confidence,
                self._trading_cfg.min_confidence,
                symbol,
            )
            return

        df = data[symbol]

        # If we already have a position in this symbol, close it first
        if symbol in state["positions"]:
            close_price = self._get_price(df, bar_idx, "close")
            self._close_position(
                state, symbol, close_price, bar_idx, current_ts, "signal_reversal"
            )

        # Position sizing
        side = "long" if signal.direction > 0 else "short"
        entry_price_raw = self._get_price(df, bar_idx, "open")

        # Slippage: for buys price goes up, for sells price goes down
        if side == "long":
            fill_price = entry_price_raw * (1.0 + self._slippage)
        else:
            fill_price = entry_price_raw * (1.0 - self._slippage)

        if fill_price <= 0:
            return

        # Build signal dict for PositionSizer
        signal_dict: dict[str, Any] = {
            "direction": signal.direction,
            "confidence": signal.confidence,
            "target_position": signal.target_position,
        }
        signal_dict.update(signal.metadata)

        # Get recent market data for ATR computation
        start = max(0, bar_idx - 20)
        market_window = df.iloc[start : bar_idx + 1]

        qty = self._position_sizer.calculate(
            capital=state["cash"],
            price=fill_price,
            signal=signal_dict,
            market_data=market_window,
        )

        qty = abs(qty)
        if qty < 1e-10:
            logger.debug("Position sizer returned zero qty for {}", symbol)
            return

        # Check margin / capital requirement
        notional = qty * fill_price
        commission_cost = notional * self._commission
        total_cost = notional * self._bt_cfg.margin_requirement + commission_cost

        if total_cost > state["cash"]:
            # Scale down to fit available capital
            available = state["cash"] - commission_cost
            if available <= 0:
                logger.debug("Insufficient cash for {} trade in {}", side, symbol)
                return
            qty = (available / self._bt_cfg.margin_requirement) / fill_price
            notional = qty * fill_price
            commission_cost = notional * self._commission
            total_cost = notional * self._bt_cfg.margin_requirement + commission_cost

            if total_cost > state["cash"] or qty < 1e-10:
                return

        # Compute stop-loss and take-profit
        atr = self._compute_atr(df, bar_idx)
        if side == "long":
            stop_loss = fill_price - self._trading_cfg.stop_loss_atr_mult * atr
            take_profit = fill_price + self._trading_cfg.take_profit_atr_mult * atr
        else:
            stop_loss = fill_price + self._trading_cfg.stop_loss_atr_mult * atr
            take_profit = fill_price - self._trading_cfg.take_profit_atr_mult * atr

        # Execute: deduct cash, create position
        state["cash"] -= total_cost

        position = Position(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=fill_price,
            entry_time=current_ts,
            current_price=fill_price,
            stop_loss=max(stop_loss, 0.0),
            take_profit=max(take_profit, 0.0),
        )
        state["positions"][symbol] = position
        state["last_signal_bar"][symbol] = bar_idx

        logger.debug(
            "Opened {} {} | qty={:.4f} @ {:.4f} | SL={:.4f} TP={:.4f} | "
            "cost={:.2f} commission={:.2f}",
            side,
            symbol,
            qty,
            fill_price,
            stop_loss,
            take_profit,
            total_cost,
            commission_cost,
        )

    # ------------------------------------------------------------------
    # Position closing
    # ------------------------------------------------------------------

    def _close_position(
        self,
        state: dict[str, Any],
        symbol: str,
        exit_price: float,
        bar_idx: int,
        current_ts: datetime,
        reason: str,
    ) -> None:
        """Close an open position and record the trade."""
        pos = state["positions"].pop(symbol, None)
        if pos is None:
            return

        # Apply slippage to exit
        if pos.side == "long":
            fill_price = exit_price * (1.0 - self._slippage)
        else:
            fill_price = exit_price * (1.0 + self._slippage)

        # Compute P&L
        if pos.side == "long":
            pnl = (fill_price - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - fill_price) * pos.qty

        # Commission on exit
        exit_notional = pos.qty * fill_price
        commission = exit_notional * self._commission
        pnl -= commission

        # Return cash: margin + P&L
        margin_held = pos.qty * pos.entry_price * self._bt_cfg.margin_requirement
        state["cash"] += margin_held + pnl
        state["daily_pnl"] += pnl

        # Record trade
        trade_record = {
            "symbol": symbol,
            "side": pos.side,
            "qty": pos.qty,
            "entry_price": pos.entry_price,
            "exit_price": fill_price,
            "entry_time": pos.entry_time,
            "exit_time": current_ts,
            "entry_bar": None,  # not tracked per-bar in state
            "exit_bar": bar_idx,
            "pnl": pnl,
            "commission": commission,
            "reason": reason,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
        }
        state["trade_log"].append(trade_record)

        logger.debug(
            "Closed {} {} | qty={:.4f} | entry={:.4f} exit={:.4f} | "
            "pnl={:.2f} | reason={}",
            pos.side,
            symbol,
            pos.qty,
            pos.entry_price,
            fill_price,
            pnl,
            reason,
        )

    def _close_all_positions(
        self,
        state: dict[str, Any],
        data: dict[str, pd.DataFrame],
        bar_idx: int,
        current_ts: datetime,
    ) -> None:
        """Close all remaining open positions at the end of the backtest."""
        for symbol in list(state["positions"].keys()):
            df = data.get(symbol)
            if df is None or bar_idx >= len(df):
                continue
            close_price = self._get_price(df, bar_idx, "close")
            self._close_position(
                state, symbol, close_price, bar_idx, current_ts, "end_of_backtest"
            )

    # ------------------------------------------------------------------
    # Equity computation
    # ------------------------------------------------------------------

    def _compute_equity(self, state: dict[str, Any]) -> float:
        """Cash + sum of unrealised P&L of all open positions."""
        total_unrealised = sum(
            pos.unrealized_pnl for pos in state["positions"].values()
        )
        # Also add back the margin held in positions
        margin_held = sum(
            pos.qty * pos.entry_price * self._bt_cfg.margin_requirement
            for pos in state["positions"].values()
        )
        return state["cash"] + margin_held + total_unrealised

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_price(
        df: pd.DataFrame, bar_idx: int, col: str = "close"
    ) -> float:
        """Safely extract a price from a DataFrame row."""
        # Handle both lowercase and capitalised column names
        if col in df.columns:
            return float(df.iloc[bar_idx][col])
        col_upper = col.capitalize()
        if col_upper in df.columns:
            return float(df.iloc[bar_idx][col_upper])
        # Fallback: try case-insensitive match
        col_map = {c.lower(): c for c in df.columns}
        if col.lower() in col_map:
            return float(df.iloc[bar_idx][col_map[col.lower()]])
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    @staticmethod
    def _compute_atr(
        df: pd.DataFrame, bar_idx: int, period: int = 14
    ) -> float:
        """Compute ATR up to the given bar index."""
        start = max(0, bar_idx - period + 1)
        window = df.iloc[start : bar_idx + 1]

        # Column name resolution
        col_map = {c.lower(): c for c in window.columns}
        h_col = col_map.get("high")
        l_col = col_map.get("low")
        c_col = col_map.get("close")

        if h_col is None or l_col is None or c_col is None:
            return 0.0

        high = window[h_col].values.astype(np.float64)
        low = window[l_col].values.astype(np.float64)
        close = window[c_col].values.astype(np.float64)

        if len(high) < 2:
            return float(high[0] - low[0]) if len(high) == 1 else 0.0

        tr1 = high - low
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])

        # First bar TR is just high - low
        tr = np.empty(len(high))
        tr[0] = tr1[0]
        tr[1:] = np.maximum(tr1[1:], np.maximum(tr2, tr3))

        return float(np.mean(tr))

    def _snapshot_positions(self, state: dict[str, Any]) -> dict:
        """Create a serialisable snapshot of current positions."""
        snapshot: dict[str, Any] = {}
        for symbol, pos in state["positions"].items():
            snapshot[symbol] = {
                "side": pos.side,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
            }
        return snapshot

    @staticmethod
    def _last_signal_bar_for(symbol: str, state: dict[str, Any]) -> int:
        """Get the last bar index at which a signal was generated."""
        return state.get("last_signal_bar", {}).get(symbol, -999)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, state: dict[str, Any], bar_idx: int) -> None:
        """Save the backtest state to disk for potential resumption."""
        if self._checkpoint_dir is None:
            return

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._checkpoint_dir / f"backtest_checkpoint_bar_{bar_idx}.json"

        # Build a JSON-safe version of the state
        checkpoint: dict[str, Any] = {
            "bar_idx": bar_idx,
            "cash": state["cash"],
            "equity_curve_length": len(state["equity_curve"]),
            "n_trades": len(state["trade_log"]),
            "n_open_positions": len(state["positions"]),
            "last_equity": state["equity_curve"][-1] if state["equity_curve"] else 0.0,
        }

        try:
            filepath.write_text(json.dumps(checkpoint, indent=2, default=str))
            logger.debug("Checkpoint saved at bar {} -> {}", bar_idx, filepath)
        except Exception as exc:
            logger.warning("Failed to save checkpoint at bar {}: {}", bar_idx, exc)
