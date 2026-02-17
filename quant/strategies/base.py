"""Base strategy abstractions and the canonical Signal dataclass.

Every concrete strategy inherits from :class:`BaseStrategy` and produces
:class:`Signal` objects that downstream layers (portfolio, execution) consume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quant.config.settings import TradingConfig


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
@dataclass
class Signal:
    """Atomic trading signal emitted by a strategy.

    Parameters
    ----------
    timestamp : datetime
        Time the signal was generated.
    symbol : str
        Instrument identifier (e.g. ``"BTC-USD"``).
    direction : int
        Directional view: ``-1`` (short), ``0`` (flat), ``+1`` (long).
    confidence : float
        Signal strength in ``[0.0, 1.0]``.  A value of ``0.0`` means no
        conviction; ``1.0`` means maximum conviction.
    target_position : float
        Desired fractional position in ``[-1.0, 1.0]``.  Negative values
        represent short exposure.
    metadata : dict[str, Any]
        Arbitrary extra information (model name, feature importances, ...).
    """

    timestamp: datetime
    symbol: str
    direction: int  # -1, 0, +1
    confidence: float  # [0.0, 1.0]
    target_position: float  # [-1.0, 1.0]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise ValueError(
                f"direction must be -1, 0, or +1, got {self.direction}"
            )
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.target_position = float(np.clip(self.target_position, -1.0, 1.0))


# ---------------------------------------------------------------------------
# BaseStrategy
# ---------------------------------------------------------------------------
class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses **must** implement :meth:`generate_signals`.  The constructor
    accepts a :class:`TradingConfig` so every strategy has access to shared
    trading parameters (cooldown, confidence thresholds, etc.).

    Parameters
    ----------
    config : TradingConfig
        Central trading configuration.
    name : str | None
        Human-readable identifier; defaults to the class name.
    """

    def __init__(self, config: TradingConfig, name: str | None = None) -> None:
        self.config = config
        self.name: str = name or self.__class__.__name__
        logger.info("Initialised strategy '{}'", self.name)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        model_predictions: np.ndarray | None = None,
    ) -> list[Signal]:
        """Produce trading signals for one or more assets.

        Parameters
        ----------
        data : pd.DataFrame
            Recent OHLCV + feature bars.  The index should be a
            ``DatetimeIndex`` and must contain at least a ``"close"`` column.
        model_predictions : np.ndarray | None
            Model forecast array of shape ``[forecast_horizon]`` representing
            predicted log-returns.  ``None`` when the strategy does not use a
            model (e.g. pure rules-based).

        Returns
        -------
        list[Signal]
            One :class:`Signal` per asset/symbol.  An empty list means *no
            actionable signal this bar*.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience helpers available to all strategies
    # ------------------------------------------------------------------
    def _latest_timestamp(self, data: pd.DataFrame) -> datetime:
        """Return the last timestamp in the DataFrame."""
        if isinstance(data.index, pd.DatetimeIndex):
            return data.index[-1].to_pydatetime()
        return datetime.utcnow()

    def _infer_symbol(self, data: pd.DataFrame) -> str:
        """Best-effort extraction of the symbol from *data*.

        Checks for a ``"symbol"`` column; falls back to a ``symbol``
        attribute on the DataFrame (some custom pipelines attach one),
        or returns ``"UNKNOWN"``.
        """
        if "symbol" in data.columns:
            return str(data["symbol"].iloc[-1])
        return getattr(data, "symbol", "UNKNOWN")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__}(name={self.name!r})>"
