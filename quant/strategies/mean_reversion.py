"""Mean reversion strategy based on RSI and Bollinger Band extremes.

Generates long signals when RSI is oversold AND price is near the lower
Bollinger Band, and short signals for the mirror condition.  Purely
rule-based -- ignores ``model_predictions``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from quant.config.settings import TradingConfig
from quant.strategies.base import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """Rule-based mean-reversion strategy.

    Parameters
    ----------
    config : TradingConfig
        Central trading configuration.
    name : str | None
        Human-readable name; defaults to ``"MeanReversionStrategy"``.
    """

    def __init__(
        self, config: TradingConfig, name: str | None = None,
    ) -> None:
        super().__init__(config, name=name)
        logger.info(
            "MeanReversionStrategy '{}' | RSI oversold={} overbought={} | "
            "BB proximity={}",
            self.name,
            self.config.mr_rsi_oversold,
            self.config.mr_rsi_overbought,
            self.config.mr_bb_proximity,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        """Generate mean-reversion signals from RSI and Bollinger Band %B.

        Required columns in *data*: ``rsi_14``, ``bb_pct``, ``close``.
        """
        required = {"rsi_14", "bb_pct", "close"}
        if not required.issubset(set(data.columns)):
            logger.debug(
                "{}: missing columns {} -- skipping",
                self.name,
                required - set(data.columns),
            )
            return []

        row = data.iloc[-1]
        rsi = float(row["rsi_14"])
        bb_pct = float(row["bb_pct"])

        if np.isnan(rsi) or np.isnan(bb_pct):
            return []

        oversold = self.config.mr_rsi_oversold
        overbought = self.config.mr_rsi_overbought
        proximity = self.config.mr_bb_proximity

        direction = 0
        if rsi < oversold and bb_pct < proximity:
            direction = 1  # long -- expect reversion up
        elif rsi > overbought and bb_pct > (1.0 - proximity):
            direction = -1  # short -- expect reversion down

        if direction == 0:
            return []

        # Confidence: average of RSI strength and BB strength
        if direction == 1:
            rsi_strength = (oversold - rsi) / oversold
            bb_strength = (proximity - bb_pct) / proximity
        else:
            rsi_strength = (rsi - overbought) / (100.0 - overbought)
            bb_strength = (bb_pct - (1.0 - proximity)) / proximity

        confidence = float(np.clip((rsi_strength + bb_strength) / 2.0, 0.0, 1.0))

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
            metadata={"strategy": self.name, "rsi": rsi, "bb_pct": bb_pct},
        )

        logger.info(
            "{} | {} | SIGNAL dir={} conf={:.3f} (RSI={:.1f}, BB%={:.3f})",
            self.name, symbol, direction, confidence, rsi, bb_pct,
        )
        return [signal]
