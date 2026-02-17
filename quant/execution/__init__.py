"""Execution layer for order submission and fill management.

Exports
-------
PaperExecutor
    Simulated executor for backtesting and paper trading.
LiveExecutor
    Real broker executor backed by the Alpaca Markets API.
"""

from quant.execution.live import LiveExecutor
from quant.execution.paper import PaperExecutor

__all__ = ["PaperExecutor", "LiveExecutor"]
