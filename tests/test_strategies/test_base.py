"""Tests for quant.strategies.base – Signal dataclass and BaseStrategy ABC."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quant.config.settings import TradingConfig
from quant.strategies.base import BaseStrategy, Signal


# ---------------------------------------------------------------------------
# Tests – Signal
# ---------------------------------------------------------------------------

class TestSignalCreation:
    """A Signal should be constructable with valid parameters."""

    def test_signal_creation(self) -> None:
        ts = datetime(2025, 1, 15, 12, 0, 0)
        sig = Signal(
            timestamp=ts,
            symbol="BTC-USD",
            direction=1,
            confidence=0.8,
            target_position=0.6,
            metadata={"model": "test"},
        )
        assert sig.timestamp == ts
        assert sig.symbol == "BTC-USD"
        assert sig.direction == 1
        assert sig.confidence == pytest.approx(0.8)
        assert sig.target_position == pytest.approx(0.6)
        assert sig.metadata == {"model": "test"}

    def test_signal_defaults(self) -> None:
        sig = Signal(
            timestamp=datetime.now(),
            symbol="ETH-USD",
            direction=0,
            confidence=0.5,
            target_position=0.0,
        )
        assert sig.metadata == {}


class TestSignalDirectionValidation:
    """Direction must be -1, 0, or +1."""

    def test_signal_direction_validation(self) -> None:
        with pytest.raises(ValueError, match="direction must be -1, 0, or \\+1"):
            Signal(
                timestamp=datetime.now(),
                symbol="X",
                direction=2,
                confidence=0.5,
                target_position=0.5,
            )

    @pytest.mark.parametrize("direction", [-1, 0, 1])
    def test_valid_directions(self, direction: int) -> None:
        sig = Signal(
            timestamp=datetime.now(),
            symbol="X",
            direction=direction,
            confidence=0.5,
            target_position=0.0,
        )
        assert sig.direction == direction


class TestSignalConfidenceClamped:
    """Confidence is clamped to [0.0, 1.0] in __post_init__."""

    def test_confidence_clamped_above(self) -> None:
        sig = Signal(
            timestamp=datetime.now(),
            symbol="X",
            direction=1,
            confidence=1.5,
            target_position=0.5,
        )
        assert sig.confidence == pytest.approx(1.0)

    def test_confidence_clamped_below(self) -> None:
        sig = Signal(
            timestamp=datetime.now(),
            symbol="X",
            direction=-1,
            confidence=-0.3,
            target_position=-0.5,
        )
        assert sig.confidence == pytest.approx(0.0)

    def test_target_position_clamped(self) -> None:
        sig = Signal(
            timestamp=datetime.now(),
            symbol="X",
            direction=1,
            confidence=0.5,
            target_position=2.0,
        )
        assert sig.target_position == pytest.approx(1.0)

    def test_target_position_clamped_negative(self) -> None:
        sig = Signal(
            timestamp=datetime.now(),
            symbol="X",
            direction=-1,
            confidence=0.5,
            target_position=-3.0,
        )
        assert sig.target_position == pytest.approx(-1.0)
