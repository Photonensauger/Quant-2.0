"""Strategies layer -- signal generation and ensemble aggregation.

Public API
----------
Signal                      Canonical trading signal dataclass.
BaseStrategy                Abstract base class for all strategies.
MLSignalStrategy            Converts ML log-return predictions to directional signals.
EnsembleStrategy            Aggregates signals from multiple sub-strategies.
MeanReversionStrategy       Rule-based mean-reversion (RSI + Bollinger Bands).
TrendFollowingStrategy      Rule-based trend-following (MACD + ADX).
VolatilityTargetingStrategy Scales position size to a target volatility.
RegimeAdaptiveStrategy      Meta-strategy adapting to changepoint regimes.
"""

from quant.strategies.base import BaseStrategy, Signal
from quant.strategies.ensemble import EnsembleStrategy
from quant.strategies.mean_reversion import MeanReversionStrategy
from quant.strategies.ml_signal import MLSignalStrategy
from quant.strategies.regime_adaptive import RegimeAdaptiveStrategy
from quant.strategies.trend_following import TrendFollowingStrategy
from quant.strategies.volatility_targeting import VolatilityTargetingStrategy

__all__ = [
    "Signal",
    "BaseStrategy",
    "MLSignalStrategy",
    "EnsembleStrategy",
    "MeanReversionStrategy",
    "TrendFollowingStrategy",
    "VolatilityTargetingStrategy",
    "RegimeAdaptiveStrategy",
]
