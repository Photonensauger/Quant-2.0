"""Trend-following strategy based on MACD crossover and ADX filter.

Only trades when ADX indicates a trending market (above threshold).
Direction follows the sign of MACD minus its signal line.  Purely
rule-based -- ignores ``model_predictions``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from quant.config.settings import TradingConfig
from quant.strategies.base import BaseStrategy, Signal


class TrendFollowingStrategy(BaseStrategy):
    """Rule-based trend-following strategy.

    Parameters
    ----------
    config : TradingConfig
        Central trading configuration.
    name : str | None
        Human-readable name; defaults to ``"TrendFollowingStrategy"``.
    """

    def __init__(
        self, config: TradingConfig, name: str | None = None,
    ) -> None:
        super().__init__(config, name=name)
        logger.info(
            "TrendFollowingStrategy '{}' | ADX threshold={}",
            self.name,
            self.config.tf_adx_threshold,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        """Generate trend-following signals from MACD and ADX.

        Required columns in *data*: ``macd``, ``macd_signal``, ``adx``,
        ``close``.
        """
        required = {"macd", "macd_signal", "adx", "close"}
        if not required.issubset(set(data.columns)):
            logger.debug(
                "{}: missing columns {} -- skipping",
                self.name,
                required - set(data.columns),
            )
            return []

        row = data.iloc[-1]
        macd_val = float(row["macd"])
        macd_sig = float(row["macd_signal"])
        adx_val = float(row["adx"])

        if np.isnan(macd_val) or np.isnan(macd_sig) or np.isnan(adx_val):
            return []

        threshold = self.config.tf_adx_threshold

        # ADX must exceed threshold to confirm trend
        if adx_val < threshold:
            return []

        # Direction from MACD - signal
        macd_diff = macd_val - macd_sig
        direction = int(np.sign(macd_diff))
        if direction == 0:
            return []

        # Confidence: ADX scaled from [threshold, 75] to [0, 1]
        adx_max = 75.0
        confidence = float(np.clip(
            (adx_val - threshold) / (adx_max - threshold), 0.0, 1.0,
        ))

        if confidence < self.config.min_confidence:
            return []

        target_position = float(np.clip(direction * confidence, -1.0, 1.0))
        timestamp = self._latest_timestamp(data)
        symbol = self._infer_symbol(data)

        signal = Signal(
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            target_position=target_position,
            metadata={
                "strategy": self.name,
                "adx": adx_val,
                "macd_diff": macd_diff,
            },
        )

        logger.info(
            "{} | {} | SIGNAL dir={} conf={:.3f} (ADX={:.1f})",
            self.name, symbol, direction, confidence, adx_val,
        )
        return [signal]
