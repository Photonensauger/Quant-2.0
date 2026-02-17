"""Central configuration dataclasses for all system components."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_device() -> torch.device:
    """Detect best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Configuration for data providers and storage."""

    cache_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_CACHE_DIR", "data/cache")
    )
    default_interval: str = "5m"
    default_lookback_days: int = 30
    supported_intervals: tuple[str, ...] = ("1m", "5m", "15m", "1h", "1d")

    # API retry settings
    max_retries: int = 3
    retry_backoff_base: float = 1.0  # seconds; exponential: base * 2^attempt

    # Validation
    min_rows: int = 100  # minimum rows for a usable dataset


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    """Configuration for the feature engineering pipeline."""

    # Technical indicator warmup (max lookback of any indicator)
    warmup_period: int = 50

    # Normalization
    rolling_window: int = 252  # rolling z-score window
    correlation_threshold: float = 0.95  # drop features above this corr
    epsilon: float = 1e-8  # avoid division by zero

    # BOCPD
    bocpd_lambda: float = 200.0  # expected run length (hazard = 1/lambda)
    bocpd_alpha0: float = 1.0
    bocpd_beta0: float = 1.0
    bocpd_kappa0: float = 1.0
    bocpd_mu0: float = 0.0

    # Targets
    forecast_horizon: int = 5  # predict N bars ahead log-return


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Configuration for all neural network models."""

    # Shared
    seq_len: int = 60  # input sequence length
    forecast_horizon: int = 5
    n_features: int = 50  # set dynamically after feature pipeline

    # Transformer (Decoder-Only)
    tf_d_model: int = 64
    tf_n_heads: int = 4
    tf_n_layers: int = 3
    tf_d_ff: int = 256
    tf_dropout: float = 0.1

    # iTransformer
    itf_d_model: int = 64
    itf_n_heads: int = 4
    itf_n_layers: int = 2
    itf_d_ff: int = 256
    itf_dropout: float = 0.1

    # AttentionLSTM
    lstm_hidden_size: int = 128
    lstm_n_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_attn_heads: int = 4

    # MomentumTransformer
    mom_d_model: int = 64
    mom_n_heads: int = 4
    mom_n_layers: int = 2
    mom_d_ff: int = 256
    mom_dropout: float = 0.1

    # PPO Agent
    ppo_hidden_size: int = 128
    ppo_n_layers: int = 2
    ppo_clip_epsilon: float = 0.2
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_entropy_coeff: float = 0.01
    ppo_value_coeff: float = 0.5

    # SchrodingerTransformer
    sqt_d_model: int = 64
    sqt_n_heads: int = 4
    sqt_n_layers: int = 2
    sqt_d_ff: int = 256
    sqt_dropout: float = 0.1
    sqt_n_regimes: int = 4

    # TopologicalAttentionNetwork
    top_d_model: int = 64
    top_n_heads: int = 4
    top_n_layers: int = 2
    top_d_ff: int = 256
    top_dropout: float = 0.1
    top_window_size: int = 10
    top_n_scales: int = 3

    # AdversarialRegimeModel
    adv_d_model: int = 64
    adv_n_heads: int = 4
    adv_n_layers: int = 2
    adv_d_ff: int = 256
    adv_dropout: float = 0.1
    adv_n_regimes: int = 3

    # EntropicPortfolioDiffusion
    epd_d_model: int = 64
    epd_n_heads: int = 4
    epd_n_layers: int = 2
    epd_d_ff: int = 256
    epd_dropout: float = 0.1
    epd_n_diffusion_steps: int = 10
    epd_entropy_weight: float = 0.01

    # CausalDiscoveryTransformer
    cdt_d_model: int = 64
    cdt_n_heads: int = 4
    cdt_n_layers: int = 2
    cdt_d_ff: int = 256
    cdt_dropout: float = 0.1
    cdt_sparsity_weight: float = 0.01

    # HamiltonianNeuralODE
    hno_d_model: int = 64
    hno_n_leapfrog_steps: int = 6
    hno_dropout: float = 0.1
    hno_hidden_size: int = 128


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Training loop
    max_epochs: int = 100
    early_stopping_patience: int = 10
    batch_size: int = 64

    # Walk-forward
    n_splits: int = 5
    gap_size: int = 10  # bars between train and test
    test_ratio: float = 0.2

    # Online retraining
    retrain_interval: int = 500  # bars between retrains
    retrain_min_samples: int = 200
    retrain_lr_factor: float = 0.1
    retrain_epochs: int = 10
    sharpe_threshold: float = 0.5  # retrain if rolling sharpe below this
    cp_score_threshold: float = 0.8  # retrain if changepoint score above this

    # Checkpoint
    checkpoint_interval: int = 100  # bars between state saves


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------
@dataclass
class TradingConfig:
    """Configuration for trading strategies and risk management."""

    # Position sizing
    max_position_pct: float = 0.1  # max 10% of capital per position
    kelly_fraction: float = 0.25  # quarter-Kelly for safety

    # Risk limits
    max_drawdown: float = 0.15  # 15% max drawdown -> liquidate
    daily_loss_limit: float = 0.03  # 3% daily loss -> stop trading
    max_var_95: float = 0.05  # 5% VaR limit
    max_correlation: float = 0.7  # max portfolio correlation

    # Strategy
    signal_cooldown: int = 5  # bars between signals for same asset
    min_confidence: float = 0.6  # minimum signal confidence to trade
    confirmation_bars: int = 2  # bars signal must persist

    # Execution
    slippage_bps: float = 5.0  # 5 basis points slippage
    commission_bps: float = 10.0  # 10 basis points commission

    # Stop loss / Take profit
    stop_loss_atr_mult: float = 2.0  # stop loss = 2x ATR
    take_profit_atr_mult: float = 3.0  # take profit = 3x ATR


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 100_000.0
    slippage_bps: float = 5.0
    commission_bps: float = 10.0
    margin_requirement: float = 1.0  # 1.0 = no leverage


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
@dataclass
class SystemConfig:
    """Top-level system configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    device: torch.device = field(default_factory=get_device)

    checkpoint_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("CHECKPOINT_DIR", "checkpoints")
    )
    state_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("STATE_DIR", "state")
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
