"""Models layer -- neural network architectures, losses, and training."""

from quant.models.adversarial_regime import AdversarialRegimeModel
from quant.models.base import BaseModel
from quant.models.causal_transformer import CausalDiscoveryTransformer
from quant.models.entropic_diffusion import EntropicPortfolioDiffusion
from quant.models.hamiltonian_ode import HamiltonianNeuralODE
from quant.models.itransformer import ITransformer
from quant.models.losses import CombinedLoss, DifferentiableSharpeRatio, DirectionalLoss
from quant.models.lstm import AttentionLSTM
from quant.models.momentum_transformer import MomentumTransformer
from quant.models.rl_portfolio import ActionResult, EvalResult, PPOAgent
from quant.models.schrodinger_transformer import SchrodingerTransformer
from quant.models.topological_attention import TopologicalAttentionNetwork
from quant.models.trainer import Trainer
from quant.models.transformer import DecoderTransformer

__all__ = [
    # Base
    "BaseModel",
    # Architectures
    "DecoderTransformer",
    "ITransformer",
    "AttentionLSTM",
    "MomentumTransformer",
    # Zug 37 Models
    "CausalDiscoveryTransformer",
    "SchrodingerTransformer",
    "TopologicalAttentionNetwork",
    "HamiltonianNeuralODE",
    "EntropicPortfolioDiffusion",
    "AdversarialRegimeModel",
    # RL
    "PPOAgent",
    "ActionResult",
    "EvalResult",
    # Losses
    "DifferentiableSharpeRatio",
    "DirectionalLoss",
    "CombinedLoss",
    # Training
    "Trainer",
]
