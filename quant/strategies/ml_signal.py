"""ML-based signal strategy.

Converts numeric model predictions (log-return forecasts) into discrete
directional :class:`Signal` objects, applying:

* **Confidence scaling** -- prediction magnitude relative to a rolling
  standard deviation of historical returns.
* **Cooldown** -- suppress new signals for ``signal_cooldown`` bars after
  the last signal for a given symbol.
* **Confirmation** -- a directional view must persist for
  ``confirmation_bars`` consecutive bars before a signal is emitted.
* **Minimum confidence** -- signals below ``min_confidence`` are discarded.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from quant.config.settings import TradingConfig
from quant.strategies.base import BaseStrategy, Signal


class MLSignalStrategy(BaseStrategy):
    """Strategy that wraps a single ML model's log-return predictions.

    Parameters
    ----------
    config : TradingConfig
        Central trading configuration (supplies cooldown, confirmation,
        min_confidence thresholds).
    name : str | None
        Human-readable name; defaults to ``"MLSignalStrategy"``.
    lookback_std : int
        Number of recent close-to-close returns used to compute the
        rolling standard deviation for confidence scaling.  Defaults to
        ``252`` (~ one trading year at daily bars).
    """

    def __init__(
        self,
        config: TradingConfig,
        name: str | None = None,
        lookback_std: int = 252,
    ) -> None:
        super().__init__(config, name=name)
        self.lookback_std = lookback_std

        # Internal state: per-symbol tracking
        # bars_since_signal[symbol] -- bars elapsed since last emitted signal
        self._bars_since_signal: dict[str, int] = defaultdict(lambda: self.config.signal_cooldown + 1)
        # confirmation_counter[symbol] -- how many consecutive bars the same
        # direction has been observed
        self._confirmation_counter: dict[str, int] = defaultdict(int)
        # last_direction[symbol] -- the direction seen on the previous bar
        self._last_direction: dict[str, int] = defaultdict(int)

        logger.info(
            "MLSignalStrategy '{}' | cooldown={} | confirmation={} | "
            "min_confidence={:.2f} | lookback_std={}",
            self.name,
            self.config.signal_cooldown,
            self.config.confirmation_bars,
            self.config.min_confidence,
            self.lookback_std,
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def generate_signals(
        self,
        data: pd.DataFrame,
        model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        """Convert model log-return predictions into a list of signals.

        Parameters
        ----------
        data : pd.DataFrame
            Recent bars; must contain a ``"close"`` column.
        model_predictions : np.ndarray | None
            1-D array of shape ``[forecast_horizon]`` with predicted
            log-returns for the next *N* bars.  If ``None``, an empty
            list is returned (nothing to convert).

        Returns
        -------
        list[Signal]
        """
        if model_predictions is None:
            logger.debug("{}: no model predictions supplied -- skipping", self.name)
            return []

        predictions = np.asarray(model_predictions, dtype=np.float64).ravel()
        if predictions.size == 0:
            return []

        symbol = self._infer_symbol(data)
        timestamp = self._latest_timestamp(data)

        # --- 1. Aggregate prediction to a single scalar ----------------
        # Use the mean predicted log-return across the forecast horizon.
        mean_pred: float = float(np.mean(predictions))

        # --- 2. Determine raw direction --------------------------------
        raw_direction = int(np.sign(mean_pred))  # -1, 0, +1

        # --- 3. Compute confidence via rolling std ----------------------
        confidence = self._compute_confidence(data, mean_pred)

        # --- 4. Increment per-symbol tick counters ----------------------
        self._bars_since_signal[symbol] += 1

        # --- 5. Confirmation gate --------------------------------------
        if raw_direction == self._last_direction[symbol] and raw_direction != 0:
            self._confirmation_counter[symbol] += 1
        else:
            self._confirmation_counter[symbol] = 1 if raw_direction != 0 else 0
        self._last_direction[symbol] = raw_direction

        confirmed = self._confirmation_counter[symbol] >= self.config.confirmation_bars

        # --- 6. Cooldown gate ------------------------------------------
        cooldown_ok = self._bars_since_signal[symbol] > self.config.signal_cooldown

        # --- 7. Confidence gate ----------------------------------------
        confidence_ok = confidence >= self.config.min_confidence

        # --- 8. Emit or suppress ---------------------------------------
        if raw_direction == 0 or not confirmed or not cooldown_ok or not confidence_ok:
            logger.trace(
                "{} | {} | dir={} conf={:.3f} confirmed={} cooldown_ok={} "
                "confidence_ok={} => SUPPRESSED",
                self.name,
                symbol,
                raw_direction,
                confidence,
                confirmed,
                cooldown_ok,
                confidence_ok,
            )
            return []

        # Target position: direction * confidence (scaled to [-1, 1])
        target_position = float(np.clip(raw_direction * confidence, -1.0, 1.0))

        signal = Signal(
            timestamp=timestamp,
            symbol=symbol,
            direction=raw_direction,
            confidence=confidence,
            target_position=target_position,
            metadata={
                "strategy": self.name,
                "mean_prediction": mean_pred,
                "forecast_horizon": len(predictions),
            },
        )

        # Reset cooldown counter
        self._bars_since_signal[symbol] = 0

        logger.info(
            "{} | {} | SIGNAL dir={} conf={:.3f} target={:.3f}",
            self.name,
            symbol,
            signal.direction,
            signal.confidence,
            signal.target_position,
        )
        return [signal]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_confidence(
        self,
        data: pd.DataFrame,
        mean_prediction: float,
    ) -> float:
        """Map prediction magnitude to ``[0, 1]`` confidence.

        Confidence is defined as::

            conf = min(|mean_prediction| / (k * rolling_std), 1.0)

        where *k* = 1.0.  This means a prediction whose magnitude equals
        the historical return standard deviation receives confidence 1.0.
        Smaller predictions yield proportionally lower confidence.
        """
        if "close" not in data.columns or len(data) < 2:
            # Fallback: cannot compute std, use raw magnitude clamped.
            return float(np.clip(abs(mean_prediction) * 100.0, 0.0, 1.0))

        # Compute log-returns from close prices
        close = data["close"].values.astype(np.float64)
        log_returns = np.diff(np.log(close + 1e-12))

        # Use at most `lookback_std` recent returns
        if len(log_returns) > self.lookback_std:
            log_returns = log_returns[-self.lookback_std:]

        hist_std = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 1e-8
        hist_std = max(hist_std, 1e-8)  # guard against zero

        confidence = abs(mean_prediction) / hist_std
        return float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all internal state (useful between backtest runs)."""
        self._bars_since_signal.clear()
        self._confirmation_counter.clear()
        self._last_direction.clear()
        logger.debug("{}: internal state reset", self.name)
