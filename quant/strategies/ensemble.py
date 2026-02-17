"""Ensemble strategy that aggregates signals from multiple sub-strategies.

The :class:`EnsembleStrategy` maintains per-model (or per-strategy) weights,
performs weighted voting to determine the consensus direction, and computes
a weighted-average confidence.  Weights are updated online via
:meth:`update_weights` using per-model rolling Sharpe ratios.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quant.config.settings import TradingConfig
from quant.strategies.base import BaseStrategy, Signal


class EnsembleStrategy(BaseStrategy):
    """Aggregate signals from multiple strategies / models.

    Parameters
    ----------
    config : TradingConfig
        Central trading configuration.
    strategies : dict[str, BaseStrategy]
        Mapping of ``model_name -> strategy_instance``.  Each strategy is
        called independently inside :meth:`generate_signals`.
    name : str | None
        Human-readable name; defaults to ``"EnsembleStrategy"``.
    initial_weights : dict[str, float] | None
        Starting per-model weights.  If ``None``, all models start with
        equal weight ``1 / n``.
    min_weight : float
        Floor for any single model weight after normalisation.  Prevents a
        model from being entirely silenced.  Defaults to ``0.05``.
    """

    def __init__(
        self,
        config: TradingConfig,
        strategies: dict[str, BaseStrategy],
        name: str | None = None,
        initial_weights: dict[str, float] | None = None,
        min_weight: float = 0.05,
    ) -> None:
        super().__init__(config, name=name)
        if not strategies:
            raise ValueError("EnsembleStrategy requires at least one sub-strategy")

        self.strategies = dict(strategies)
        self.min_weight = min_weight

        # --- Initialise weights ----------------------------------------
        if initial_weights is not None:
            self.weights: dict[str, float] = {
                k: float(v) for k, v in initial_weights.items()
            }
            # Fill in any strategies missing from the supplied weights
            for key in self.strategies:
                if key not in self.weights:
                    self.weights[key] = 1.0 / len(self.strategies)
        else:
            n = len(self.strategies)
            self.weights = {k: 1.0 / n for k in self.strategies}

        self._normalise_weights()

        logger.info(
            "EnsembleStrategy '{}' | {} sub-strategies | weights={}",
            self.name,
            len(self.strategies),
            {k: round(v, 4) for k, v in self.weights.items()},
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def generate_signals(
        self,
        data: pd.DataFrame,
        model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        """Generate an ensemble signal by aggregating sub-strategy signals.

        Each sub-strategy is called with the same ``data``.
        ``model_predictions`` is forwarded to every sub-strategy as-is; if
        per-model predictions are needed, prefer constructing dedicated
        :class:`MLSignalStrategy` instances externally and passing them
        in ``strategies``.

        Aggregation rules
        -----------------
        1. **Direction** -- weighted vote across sub-strategy signals.
           The sign of the sum of ``weight_i * direction_i`` becomes the
           consensus direction.  A sum whose absolute value is below a
           small threshold (``0.1``) yields ``direction = 0`` (flat).
        2. **Confidence** -- weighted average of individual confidences,
           restricted to models that agree with the consensus direction.
        3. **Target position** -- ``consensus_direction * confidence``.

        Parameters
        ----------
        data : pd.DataFrame
            Recent OHLCV + feature bars.
        model_predictions : np.ndarray | None
            Optional model forecast passed to every sub-strategy.

        Returns
        -------
        list[Signal]
        """
        # Collect signals from every sub-strategy -----------------------
        all_signals: dict[str, list[Signal]] = {}
        for model_name, strategy in self.strategies.items():
            try:
                sigs = strategy.generate_signals(data, model_predictions)
                all_signals[model_name] = sigs
            except Exception:
                logger.exception(
                    "EnsembleStrategy '{}': sub-strategy '{}' raised an "
                    "exception -- skipping",
                    self.name,
                    model_name,
                )
                all_signals[model_name] = []

        # Flatten to (model_name, signal) pairs -------------------------
        flat: list[tuple[str, Signal]] = [
            (name, sig)
            for name, sigs in all_signals.items()
            for sig in sigs
        ]

        if not flat:
            logger.debug(
                "EnsembleStrategy '{}': no sub-strategy signals -- nothing to aggregate",
                self.name,
            )
            return []

        # Group by symbol -----------------------------------------------
        by_symbol: dict[str, list[tuple[str, Signal]]] = {}
        for model_name, sig in flat:
            by_symbol.setdefault(sig.symbol, []).append((model_name, sig))

        ensemble_signals: list[Signal] = []
        for symbol, entries in by_symbol.items():
            signal = self._aggregate_for_symbol(symbol, entries, data)
            if signal is not None:
                ensemble_signals.append(signal)

        return ensemble_signals

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------
    def update_weights(self, performance: dict[str, float]) -> None:
        """Update model weights using per-model rolling Sharpe ratios.

        Parameters
        ----------
        performance : dict[str, float]
            Mapping of ``model_name -> rolling_sharpe``.  Models not
            present in the dict keep their current weight.

        The new raw weight for each model is::

            w_i = max(sharpe_i, 0) + epsilon

        Negative Sharpe values are floored at zero (the model still gets
        ``min_weight`` after normalisation, but contributes little).
        """
        epsilon = 1e-6
        for model_name, sharpe in performance.items():
            if model_name in self.weights:
                self.weights[model_name] = max(float(sharpe), 0.0) + epsilon
            else:
                logger.warning(
                    "EnsembleStrategy '{}': unknown model '{}' in "
                    "performance dict -- ignoring",
                    self.name,
                    model_name,
                )

        self._normalise_weights()
        logger.info(
            "EnsembleStrategy '{}' weights updated: {}",
            self.name,
            {k: round(v, 4) for k, v in self.weights.items()},
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        """Serialise the ensemble state to a plain dict.

        Returns
        -------
        dict[str, Any]
            Contains ``weights``, ``min_weight``, ``name``, and each
            sub-strategy's name.
        """
        return {
            "name": self.name,
            "min_weight": self.min_weight,
            "weights": copy.deepcopy(self.weights),
            "strategy_names": list(self.strategies.keys()),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore ensemble state from a dict produced by :meth:`get_state`.

        Parameters
        ----------
        state : dict[str, Any]
            State dictionary (e.g. loaded from disk).
        """
        self.name = state.get("name", self.name)
        self.min_weight = state.get("min_weight", self.min_weight)
        saved_weights: dict[str, float] = state.get("weights", {})

        for key in self.strategies:
            if key in saved_weights:
                self.weights[key] = float(saved_weights[key])

        self._normalise_weights()
        logger.info(
            "EnsembleStrategy '{}' state loaded | weights={}",
            self.name,
            {k: round(v, 4) for k, v in self.weights.items()},
        )

    def save_state(self, path: str | Path) -> None:
        """Persist ensemble state as JSON.

        Parameters
        ----------
        path : str | Path
            File path to write.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_state(), f, indent=2)
        logger.info("EnsembleStrategy '{}' state saved to {}", self.name, path)

    @classmethod
    def load_state_from_file(cls, path: str | Path) -> dict[str, Any]:
        """Read a state dict from a JSON file.

        This is a convenience class method; the caller is responsible for
        constructing a new :class:`EnsembleStrategy` and passing the dict
        to :meth:`load_state`.

        Parameters
        ----------
        path : str | Path
            Path to the JSON state file.

        Returns
        -------
        dict[str, Any]
        """
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _aggregate_for_symbol(
        self,
        symbol: str,
        entries: list[tuple[str, Signal]],
        data: pd.DataFrame,
    ) -> Signal | None:
        """Perform weighted aggregation for a single symbol.

        Returns ``None`` if the resulting confidence is below
        ``min_confidence`` or the direction is flat.
        """
        # Weighted direction vote
        weighted_direction_sum: float = 0.0
        total_weight: float = 0.0
        for model_name, sig in entries:
            w = self.weights.get(model_name, 0.0)
            weighted_direction_sum += w * sig.direction
            total_weight += w

        if total_weight < 1e-12:
            return None

        normalised_vote = weighted_direction_sum / total_weight  # in [-1, 1]

        # Consensus direction (with dead-zone)
        if abs(normalised_vote) < 0.1:
            consensus_direction = 0
        else:
            consensus_direction = int(np.sign(normalised_vote))

        if consensus_direction == 0:
            return None

        # Weighted average confidence (only models agreeing with consensus)
        conf_num: float = 0.0
        conf_den: float = 0.0
        for model_name, sig in entries:
            if sig.direction == consensus_direction:
                w = self.weights.get(model_name, 0.0)
                conf_num += w * sig.confidence
                conf_den += w

        confidence = conf_num / conf_den if conf_den > 1e-12 else 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Minimum confidence gate
        if confidence < self.config.min_confidence:
            logger.debug(
                "EnsembleStrategy '{}' | {} | confidence {:.3f} < "
                "min_confidence {:.3f} => suppressed",
                self.name,
                symbol,
                confidence,
                self.config.min_confidence,
            )
            return None

        target_position = float(
            np.clip(consensus_direction * confidence, -1.0, 1.0)
        )
        timestamp = self._latest_timestamp(data)

        signal = Signal(
            timestamp=timestamp,
            symbol=symbol,
            direction=consensus_direction,
            confidence=confidence,
            target_position=target_position,
            metadata={
                "strategy": self.name,
                "weighted_vote": round(normalised_vote, 4),
                "n_sub_signals": len(entries),
                "contributing_models": [m for m, _ in entries],
            },
        )

        logger.info(
            "EnsembleStrategy '{}' | {} | SIGNAL dir={} conf={:.3f} "
            "target={:.3f} (vote={:.3f}, n={})",
            self.name,
            symbol,
            signal.direction,
            signal.confidence,
            signal.target_position,
            normalised_vote,
            len(entries),
        )
        return signal

    def _normalise_weights(self) -> None:
        """Normalise weights to sum to 1, enforcing ``min_weight`` floor."""
        # Apply floor
        for key in self.weights:
            self.weights[key] = max(self.weights[key], self.min_weight)

        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total
