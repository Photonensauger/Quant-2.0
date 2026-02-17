"""Performance metrics for backtest evaluation.

Computes all standard quantitative finance metrics from an equity curve
and trade log: returns, risk-adjusted ratios, drawdown statistics, and
trade-level analytics.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRADING_DAYS_PER_YEAR: int = 252
_EPSILON: float = 1e-10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    equity_curve: list[float],
    trade_log: list[dict],
    initial_capital: float,
) -> dict[str, float]:
    """Compute a comprehensive dictionary of backtest performance metrics.

    Parameters
    ----------
    equity_curve : list[float]
        Time-series of portfolio equity values (one per bar).
    trade_log : list[dict]
        Each entry should contain at minimum:
            ``pnl`` (float), ``side`` (str), ``entry_time`` (datetime),
            ``exit_time`` (datetime), ``entry_price`` (float),
            ``exit_price`` (float), ``qty`` (float).
    initial_capital : float
        Starting portfolio value.

    Returns
    -------
    dict[str, float]
        Named metrics.  Keys are lowercase with underscores.
    """
    if len(equity_curve) < 2:
        logger.warning("Equity curve has fewer than 2 points; returning empty metrics.")
        return _empty_metrics()

    equity = np.asarray(equity_curve, dtype=np.float64)
    returns = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], _EPSILON)

    metrics: dict[str, float] = {}

    # -- Return metrics -------------------------------------------------------
    metrics["initial_capital"] = initial_capital
    metrics["final_equity"] = float(equity[-1])
    metrics["total_return"] = _total_return(equity, initial_capital)
    metrics["annualized_return"] = _annualized_return(equity, initial_capital)

    # -- Risk metrics ---------------------------------------------------------
    metrics["volatility"] = _annualized_volatility(returns)
    metrics["sharpe_ratio"] = _sharpe_ratio(returns)
    metrics["sortino_ratio"] = _sortino_ratio(returns)
    metrics["calmar_ratio"] = _calmar_ratio(
        metrics["annualized_return"], equity
    )

    # -- Drawdown metrics -----------------------------------------------------
    dd, dd_dur = _drawdown_stats(equity)
    metrics["max_drawdown"] = dd
    metrics["max_drawdown_duration"] = dd_dur  # in bars

    # -- Trade-level metrics --------------------------------------------------
    trade_metrics = _trade_metrics(trade_log)
    metrics.update(trade_metrics)

    logger.info(
        "Metrics computed | total_return={:.2%} | sharpe={:.2f} | "
        "max_dd={:.2%} | trades={}",
        metrics["total_return"],
        metrics["sharpe_ratio"],
        metrics["max_drawdown"],
        metrics["total_trades"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Return metrics
# ---------------------------------------------------------------------------

def _total_return(equity: np.ndarray, initial_capital: float) -> float:
    """Simple total return: (final - initial) / initial."""
    if initial_capital <= 0:
        return 0.0
    return float((equity[-1] - initial_capital) / initial_capital)


def _annualized_return(equity: np.ndarray, initial_capital: float) -> float:
    """Annualized return assuming 252 trading days per year."""
    n_bars = len(equity) - 1
    if n_bars <= 0 or initial_capital <= 0:
        return 0.0

    total = equity[-1] / initial_capital
    if total <= 0:
        return -1.0

    years = n_bars / _TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0

    return float(total ** (1.0 / years) - 1.0)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def _annualized_volatility(returns: np.ndarray) -> float:
    """Annualized standard deviation of returns."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio (assumes zero risk-free rate by default)."""
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free_rate / _TRADING_DAYS_PER_YEAR
    mean_excess = np.mean(excess)
    std = np.std(excess, ddof=1)

    if std < _EPSILON:
        return 0.0

    return float(mean_excess / std * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation in denominator)."""
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free_rate / _TRADING_DAYS_PER_YEAR
    mean_excess = np.mean(excess)

    downside = excess[excess < 0]
    if len(downside) < 1:
        # No negative returns -- infinite Sortino; cap at a large number
        return 100.0 if mean_excess > 0 else 0.0

    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < _EPSILON:
        return 0.0

    return float(mean_excess / downside_std * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _calmar_ratio(annualized_return: float, equity: np.ndarray) -> float:
    """Calmar ratio = annualized return / max drawdown."""
    max_dd, _ = _drawdown_stats(equity)
    if max_dd < _EPSILON:
        return 0.0
    return float(annualized_return / max_dd)


# ---------------------------------------------------------------------------
# Drawdown metrics
# ---------------------------------------------------------------------------

def _drawdown_stats(equity: np.ndarray) -> tuple[float, float]:
    """Compute max drawdown (fractional) and max drawdown duration (bars).

    Returns
    -------
    (max_drawdown, max_drawdown_duration)
        Both as floats.  Drawdown is a positive fraction (e.g. 0.15 = 15%).
        Duration is measured in bars (number of bars from peak to recovery
        of that peak, or end of series if never recovered).
    """
    if len(equity) < 2:
        return 0.0, 0.0

    running_max = np.maximum.accumulate(equity)
    safe_max = np.where(running_max > 0, running_max, _EPSILON)
    drawdowns = (running_max - equity) / safe_max

    max_dd = float(np.max(drawdowns))

    # Max drawdown duration: longest streak below previous peak
    peak_idx = 0
    max_duration = 0
    current_duration = 0

    for i in range(1, len(equity)):
        if equity[i] >= running_max[i]:
            # At or above the running max -- reset duration counter
            current_duration = 0
        else:
            current_duration += 1
            max_duration = max(max_duration, current_duration)

    return max_dd, float(max_duration)


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------

def _trade_metrics(trade_log: list[dict]) -> dict[str, float]:
    """Extract trade-level metrics from the trade log."""
    metrics: dict[str, float] = {}

    total_trades = len(trade_log)
    metrics["total_trades"] = float(total_trades)

    if total_trades == 0:
        metrics["win_rate"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["avg_win"] = 0.0
        metrics["avg_loss"] = 0.0
        metrics["expectancy"] = 0.0
        metrics["avg_trade_duration"] = 0.0
        return metrics

    pnls = np.array([t.get("pnl", 0.0) for t in trade_log], dtype=np.float64)

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    n_wins = len(wins)
    n_losses = len(losses)

    # Win rate
    metrics["win_rate"] = float(n_wins / total_trades)

    # Average win / average loss
    avg_win = float(np.mean(wins)) if n_wins > 0 else 0.0
    avg_loss = float(np.mean(losses)) if n_losses > 0 else 0.0
    metrics["avg_win"] = avg_win
    metrics["avg_loss"] = avg_loss

    # Profit factor = gross profits / |gross losses|
    gross_profit = float(np.sum(wins)) if n_wins > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if n_losses > 0 else 0.0
    metrics["profit_factor"] = (
        float(gross_profit / gross_loss) if gross_loss > _EPSILON else 0.0
    )

    # Expectancy (average P&L per trade)
    metrics["expectancy"] = float(np.mean(pnls))

    # Average trade duration (in bars or timedelta)
    durations: list[float] = []
    for t in trade_log:
        entry = t.get("entry_time")
        exit_ = t.get("exit_time")
        if isinstance(entry, datetime) and isinstance(exit_, datetime):
            delta = (exit_ - entry).total_seconds()
            durations.append(delta)
        elif isinstance(entry, (int, float)) and isinstance(exit_, (int, float)):
            durations.append(float(exit_ - entry))

    if durations:
        metrics["avg_trade_duration"] = float(np.mean(durations))
    else:
        # Fall back to bar-count if entry/exit bars are stored
        bar_durations = []
        for t in trade_log:
            eb = t.get("entry_bar")
            xb = t.get("exit_bar")
            if eb is not None and xb is not None:
                bar_durations.append(float(xb - eb))
        metrics["avg_trade_duration"] = (
            float(np.mean(bar_durations)) if bar_durations else 0.0
        )

    return metrics


# ---------------------------------------------------------------------------
# Empty metrics fallback
# ---------------------------------------------------------------------------

def _empty_metrics() -> dict[str, float]:
    """Return a zeroed-out metrics dictionary when computation is impossible."""
    return {
        "initial_capital": 0.0,
        "final_equity": 0.0,
        "total_return": 0.0,
        "annualized_return": 0.0,
        "volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_duration": 0.0,
        "total_trades": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "expectancy": 0.0,
        "avg_trade_duration": 0.0,
    }
