"""Volatility-targeting strategy that scales position size to achieve a
target annualised volatility.

Computes realised volatility from recent log-returns and adjusts the
target position so the portfolio's ex-ante volatility matches a target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from quant.config.settings import TradingConfig
from quant.strategies.base import BaseStrategy, Signal


class VolatilityTargetingStrategy(BaseStrategy):
    """Volatility-targeting strategy.

    Parameters
    ----------
    config : TradingConfig
        Central trading configuration.
    name : str | None
        Human-readable name; defaults to ``"VolatilityTargetingStrategy"``.
    """

    def __init__(
        self, config: TradingConfig, name: str | None = None,
    ) -> None:
        super().__init__(config, name=name)
        logger.info(
            "VolatilityTargetingStrategy '{}' | target_vol={} lookback={} "
            "max_leverage={}",
            self.name,
            self.config.vt_target_vol,
            self.config.vt_vol_lookback,
            self.config.vt_max_leverage,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        """Generate a volatility-targeted signal.

        Required column: ``close`` (at least ``vt_vol_lookback + 1`` rows).
        """
        if "close" not in data.columns:
            logger.debug("{}: missing 'close' column -- skipping", self.name)
            return []

        lookback = self.config.vt_vol_lookback
        if len(data) < lookback + 1:
            logger.debug(
                "{}: insufficient data ({} < {}) -- skipping",
                self.name, len(data), lookback + 1,
            )
            return []

        close = data["close"].values.astype(np.float64)
        log_returns = np.diff(np.log(close[-lookback - 1:]))

        ann_factor = self.config.vt_annualization_factor
        realized_vol = float(np.std(log_returns, ddof=1) * np.sqrt(ann_factor))
        realized_vol = max(realized_vol, 1e-10)

        target_vol = self.config.vt_target_vol
        max_lev = self.config.vt_max_leverage

        scale = target_vol / realized_vol
        scale = min(scale, max_lev)

        # Direction: from model_predictions if available, else last return
        if model_predictions is not None:
            preds = np.asarray(model_predictions, dtype=np.float64).ravel()
            direction = int(np.sign(np.mean(preds))) if preds.size > 0 else 0
        else:
            direction = int(np.sign(log_returns[-1]))

        if direction == 0:
            return []

        target_position = float(np.clip(direction * scale, -1.0, 1.0))
        confidence = float(np.clip(min(scale, 1.0), 0.0, 1.0))

        if confidence < self.config.min_confidence:
            return []

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
                "realized_vol": realized_vol,
                "scale": scale,
            },
        )

        logger.info(
            "{} | {} | SIGNAL dir={} conf={:.3f} target={:.3f} "
            "(vol={:.4f}, scale={:.2f})",
            self.name, symbol, direction, confidence,
            target_position, realized_vol, scale,
        )
        return [signal]
