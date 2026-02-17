"""Core orchestration layer: state management and self-training loop."""

from quant.core.state_manager import SystemStateManager
from quant.core.self_trainer import SelfTrainer

__all__ = ["SystemStateManager", "SelfTrainer"]
