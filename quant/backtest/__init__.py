"""Backtest layer -- event-driven backtesting engine and performance metrics.

Public API::

    from quant.backtest import BacktestEngine, BacktestResult, compute_metrics
"""

from quant.backtest.engine import BacktestEngine, BacktestResult
from quant.backtest.metrics import compute_metrics

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "compute_metrics",
]
