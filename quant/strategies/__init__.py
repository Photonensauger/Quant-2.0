"""Strategies layer -- signal generation and ensemble aggregation.

Public API
----------
Signal             Canonical trading signal dataclass.
BaseStrategy       Abstract base class for all strategies.
MLSignalStrategy   Converts ML log-return predictions to directional signals.
EnsembleStrategy   Aggregates signals from multiple sub-strategies.
"""

from quant.strategies.base import BaseStrategy, Signal
from quant.strategies.ensemble import EnsembleStrategy
from quant.strategies.ml_signal import MLSignalStrategy

__all__ = [
    "Signal",
    "BaseStrategy",
    "MLSignalStrategy",
    "EnsembleStrategy",
]
