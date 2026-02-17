"""Regime-adaptive meta-strategy that wraps sub-strategies and adjusts
confidence based on changepoint detection scores.

When ``cp_score`` is high (regime change detected), confidence is reduced
to protect against false signals during transitions.  In dual mode with
both a "trend" and "reversion" sub-strategy, the active strategy is
selected based on ADX.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from quant.config.settings import TradingConfig
from quant.strategies.base import BaseStrategy, Signal


class RegimeAdaptiveStrategy(BaseStrategy):
    """Meta-strategy that adapts sub-strategy signals to the current regime.

    Parameters
    ----------
    config : TradingConfig
        Central trading configuration.
    sub_strategies : dict[str, BaseStrategy]
        Named sub-strategies.  Keys are used for dual-mode selection
        (e.g. ``"trend"`` and ``"reversion"``).
    name : str | None
        Human-readable name; defaults to ``"RegimeAdaptiveStrategy"``.
    """

    def __init__(
        self,
        config: TradingConfig,
        sub_strategies: dict[str, BaseStrategy],
        name: str | None = None,
    ) -> None:
        super().__init__(config, name=name)
        if not sub_strategies:
            raise ValueError(
                "RegimeAdaptiveStrategy requires at least one sub-strategy"
            )
        self.sub_strategies = dict(sub_strategies)

        # Detect dual mode
        keys = set(self.sub_strategies.keys())
        self._dual_mode = ("trend" in keys) and ("reversion" in keys)

        logger.info(
            "RegimeAdaptiveStrategy '{}' | {} sub-strategies | dual_mode={} | "
            "cp_threshold={} | reduction={}",
            self.name,
            len(self.sub_strategies),
            self._dual_mode,
            self.config.ra_cp_threshold,
            self.config.ra_cp_confidence_reduction,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        """Generate regime-adapted signals.

        In dual mode (sub-strategies named ``"trend"`` and ``"reversion"``):
        - ADX >= ``tf_adx_threshold`` -> use the trend strategy
        - ADX < ``tf_adx_threshold`` -> use the reversion strategy

        In standard mode: collect signals from all sub-strategies.

        All signals' confidence is reduced when ``cp_score`` exceeds
        ``ra_cp_threshold``.
        """
        # Determine which strategies to query
        if self._dual_mode:
            active_strategies = self._select_dual_strategy(data)
        else:
            active_strategies = self.sub_strategies

        # Collect signals from active strategies
        raw_signals: list[Signal] = []
        for strat_name, strategy in active_strategies.items():
            try:
                sigs = strategy.generate_signals(data, model_predictions)
                raw_signals.extend(sigs)
            except Exception:
                logger.exception(
                    "{}: sub-strategy '{}' raised an exception -- skipping",
                    self.name, strat_name,
                )

        if not raw_signals:
            return []

        # Apply changepoint-based confidence reduction
        cp_score = self._get_cp_score(data)
        threshold = self.config.ra_cp_threshold
        reduction = self.config.ra_cp_confidence_reduction
        min_conf = self.config.min_confidence

        adjusted_signals: list[Signal] = []
        for sig in raw_signals:
            adj_confidence = sig.confidence
            if cp_score > threshold:
                adj_confidence *= (1.0 - reduction * cp_score)
            adj_confidence = float(np.clip(adj_confidence, 0.0, 1.0))

            # Suppress if confidence drops below minimum
            if adj_confidence < min_conf:
                logger.debug(
                    "{}: suppressing signal for {} (adj_conf={:.3f} < {:.3f})",
                    self.name, sig.symbol, adj_confidence, min_conf,
                )
                continue

            target_position = float(
                np.clip(sig.direction * adj_confidence, -1.0, 1.0)
            )
            adjusted_signals.append(Signal(
                timestamp=sig.timestamp,
                symbol=sig.symbol,
                direction=sig.direction,
                confidence=adj_confidence,
                target_position=target_position,
                metadata={
                    **sig.metadata,
                    "meta_strategy": self.name,
                    "cp_score": cp_score,
                    "confidence_before_adjustment": sig.confidence,
                },
            ))

        return adjusted_signals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_dual_strategy(
        self, data: pd.DataFrame,
    ) -> dict[str, BaseStrategy]:
        """In dual mode, select trend or reversion based on ADX."""
        if "adx" in data.columns:
            adx_val = float(data["adx"].iloc[-1])
            if not np.isnan(adx_val):
                if adx_val >= self.config.tf_adx_threshold:
                    return {"trend": self.sub_strategies["trend"]}
                else:
                    return {"reversion": self.sub_strategies["reversion"]}

        # Fallback: use all strategies
        return self.sub_strategies

    @staticmethod
    def _get_cp_score(data: pd.DataFrame) -> float:
        """Extract the latest changepoint score from the data."""
        if "cp_score" in data.columns:
            val = float(data["cp_score"].iloc[-1])
            return val if not np.isnan(val) else 0.0
        return 0.0
