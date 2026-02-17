"""Tests for quant.config.settings configuration dataclasses."""

from __future__ import annotations

import torch

from quant.config.settings import (
    BacktestConfig,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    SystemConfig,
    TradingConfig,
    TrainingConfig,
    get_device,
)


class TestDataclassesInstantiate:
    """All configuration dataclasses should instantiate with defaults without errors."""

    def test_dataclasses_instantiate(self) -> None:
        data = DataConfig()
        assert data.default_interval == "5m"
        assert data.max_retries == 3

        features = FeatureConfig()
        assert features.warmup_period == 50

        model = ModelConfig()
        assert model.seq_len == 60
        assert model.forecast_horizon == 5

        training = TrainingConfig()
        assert training.max_epochs == 100

        trading = TradingConfig()
        assert trading.max_position_pct == 0.1

        backtest = BacktestConfig()
        assert backtest.initial_capital == 100_000.0


class TestDeviceDetection:
    """get_device should return a valid torch.device."""

    def test_device_detection(self) -> None:
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda", "mps")


class TestSystemConfigAggregatesAll:
    """SystemConfig should aggregate all sub-configs."""

    def test_system_config_aggregates_all(self, sample_config: SystemConfig) -> None:
        assert isinstance(sample_config.data, DataConfig)
        assert isinstance(sample_config.features, FeatureConfig)
        assert isinstance(sample_config.model, ModelConfig)
        assert isinstance(sample_config.training, TrainingConfig)
        assert isinstance(sample_config.trading, TradingConfig)
        assert isinstance(sample_config.backtest, BacktestConfig)
        assert isinstance(sample_config.device, torch.device)
        assert sample_config.checkpoint_dir is not None
        assert sample_config.state_dir is not None
        assert sample_config.log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
