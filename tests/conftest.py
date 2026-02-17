"""Shared pytest fixtures for the Quant 2.0 test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quant.config.settings import DataConfig, ModelConfig, SystemConfig


# ---------------------------------------------------------------------------
# OHLCV DataFrame fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Return a DataFrame with 300 rows of realistic random-walk OHLCV data."""
    np.random.seed(42)
    n = 300

    # Random-walk close price starting at 100
    returns = np.random.normal(loc=0.0002, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))

    # Derive open (previous close with small jitter)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    open_ += np.random.normal(0, 0.05, size=n)

    # High is always >= max(open, close), low is always <= min(open, close)
    intraday_range = np.abs(np.random.normal(0, 0.5, size=n))
    high = np.maximum(open_, close) + intraday_range
    low = np.minimum(open_, close) - intraday_range

    volume = np.random.lognormal(mean=15, sigma=1.0, size=n).astype(np.float64)

    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "datetime"
    return df


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_config() -> SystemConfig:
    """Return a default SystemConfig."""
    return SystemConfig()


@pytest.fixture
def sample_model_config() -> ModelConfig:
    """Return a default ModelConfig."""
    return ModelConfig()


@pytest.fixture
def small_model_config() -> ModelConfig:
    """Return a ModelConfig with small dimensions for fast CPU tests."""
    return ModelConfig(
        seq_len=16,
        forecast_horizon=1,
        n_features=8,
        # Transformer (Decoder-Only)
        tf_d_model=32,
        tf_n_heads=2,
        tf_n_layers=1,
        tf_d_ff=64,
        tf_dropout=0.0,
        # iTransformer
        itf_d_model=32,
        itf_n_heads=2,
        itf_n_layers=1,
        itf_d_ff=64,
        itf_dropout=0.0,
        # AttentionLSTM
        lstm_hidden_size=32,
        lstm_n_layers=1,
        lstm_dropout=0.0,
        lstm_attn_heads=2,
        # MomentumTransformer
        mom_d_model=32,
        mom_n_heads=2,
        mom_n_layers=1,
        mom_d_ff=64,
        mom_dropout=0.0,
        # SchrodingerTransformer
        sqt_d_model=32,
        sqt_n_heads=2,
        sqt_n_layers=1,
        sqt_d_ff=64,
        sqt_dropout=0.0,
        sqt_n_regimes=3,
        # TopologicalAttentionNetwork
        top_d_model=32,
        top_n_heads=2,
        top_n_layers=1,
        top_d_ff=64,
        top_dropout=0.0,
        top_window_size=5,
        top_n_scales=2,
        # AdversarialRegimeModel
        adv_d_model=32,
        adv_n_heads=2,
        adv_n_layers=1,
        adv_d_ff=64,
        adv_dropout=0.0,
        adv_n_regimes=3,
        # EntropicPortfolioDiffusion
        epd_d_model=32,
        epd_n_heads=2,
        epd_n_layers=1,
        epd_d_ff=64,
        epd_dropout=0.0,
        epd_n_diffusion_steps=3,
        epd_entropy_weight=0.01,
        # CausalDiscoveryTransformer
        cdt_d_model=32,
        cdt_n_heads=2,
        cdt_n_layers=1,
        cdt_d_ff=64,
        cdt_dropout=0.0,
        cdt_sparsity_weight=0.01,
        # HamiltonianNeuralODE
        hno_d_model=32,
        hno_n_leapfrog_steps=3,
        hno_dropout=0.0,
        hno_hidden_size=64,
    )


# ---------------------------------------------------------------------------
# Features fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_features() -> np.ndarray:
    """Return a (300, 50) feature array of random floats."""
    np.random.seed(123)
    return np.random.randn(300, 50).astype(np.float32)


# ---------------------------------------------------------------------------
# Torch fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def cpu_device() -> torch.device:
    """Return the CPU device explicitly."""
    return torch.device("cpu")


@pytest.fixture
def sample_batch(small_model_config: ModelConfig, cpu_device: torch.device) -> torch.Tensor:
    """Return a random tensor [batch=4, seq_len, n_features] on CPU."""
    torch.manual_seed(0)
    return torch.randn(
        4,
        small_model_config.seq_len,
        small_model_config.n_features,
        device=cpu_device,
    )


# ---------------------------------------------------------------------------
# Temporary directory fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory (pytest built-in tmp_path wrapper)."""
    return tmp_path
