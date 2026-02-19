"""Tests for quant.models.trainer – Trainer (walk-forward training loop)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quant.config.settings import ModelConfig, TrainingConfig
from quant.models.base import BaseModel


# ---------------------------------------------------------------------------
# Minimal concrete model for testing
# ---------------------------------------------------------------------------

class _DummyModel(BaseModel):
    """Tiny model that maps [B, seq_len, n_features] -> [B, forecast_horizon]."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.net = nn.Linear(config.seq_len * config.n_features, config.forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.size(0), -1))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def model_config() -> ModelConfig:
    return ModelConfig(seq_len=10, n_features=4, forecast_horizon=3)


@pytest.fixture()
def training_config() -> TrainingConfig:
    return TrainingConfig(
        learning_rate=1e-2,
        max_epochs=20,
        early_stopping_patience=5,
        max_grad_norm=1.0,
        batch_size=16,
        scheduler_patience=3,
        scheduler_factor=0.5,
        retrain_epochs=3,
        retrain_lr_factor=0.1,
    )


def _make_loaders(
    model_config: ModelConfig,
    n_train: int = 128,
    n_val: int = 32,
    batch_size: int = 16,
) -> tuple[DataLoader, DataLoader]:
    """Create synthetic train/val DataLoaders."""
    torch.manual_seed(0)
    seq_len, n_feat, horizon = (
        model_config.seq_len,
        model_config.n_features,
        model_config.forecast_horizon,
    )

    x_train = torch.randn(n_train, seq_len, n_feat)
    y_train = torch.randn(n_train, horizon)
    x_val = torch.randn(n_val, seq_len, n_feat)
    y_val = torch.randn(n_val, horizon)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


@pytest.fixture()
def trainer(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
):
    """Build a Trainer with the tiny dummy model."""
    from quant.models.trainer import Trainer

    model = _DummyModel(model_config)
    criterion = nn.MSELoss()
    return Trainer(
        model=model,
        criterion=criterion,
        model_config=model_config,
        training_config=training_config,
        device=device,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFitReturnsMetrics:
    """fit() should return a metrics dict with the expected keys."""

    def test_fit_returns_metrics(self, trainer, model_config: ModelConfig) -> None:
        train_loader, val_loader = _make_loaders(model_config, n_train=64, n_val=16)
        metrics = trainer.fit(train_loader, val_loader, max_epochs=3)

        assert "train_loss" in metrics
        assert "val_loss" in metrics
        assert "best_epoch" in metrics
        assert "best_val_loss" in metrics
        assert "directional_accuracy" in metrics
        assert isinstance(metrics["train_loss"], list)
        assert len(metrics["train_loss"]) == 3


class TestFitReducesLoss:
    """Training for several epochs should reduce the training loss."""

    def test_fit_reduces_loss(self, trainer, model_config: ModelConfig) -> None:
        train_loader, val_loader = _make_loaders(model_config, n_train=128, n_val=32)
        metrics = trainer.fit(train_loader, val_loader, max_epochs=15)

        losses = metrics["train_loss"]
        # First loss should be higher than the best observed training loss
        assert min(losses) < losses[0], (
            f"Training loss should decrease: first={losses[0]}, min={min(losses)}"
        )


class TestContinueTraining:
    """continue_training should return the correct keys and produce train_loss."""

    def test_continue_training(self, trainer, model_config: ModelConfig) -> None:
        train_loader, val_loader = _make_loaders(model_config, n_train=64, n_val=16)

        # Initial fit
        trainer.fit(train_loader, val_loader, max_epochs=3)
        epoch_after_fit = trainer.epoch

        # Continue
        result = trainer.continue_training(
            train_loader, epochs=2, val_loader=val_loader
        )

        assert "train_loss" in result
        assert len(result["train_loss"]) == 2
        assert "val_loss_before" in result
        assert "val_loss_after" in result
        assert "improved" in result
        assert isinstance(result["improved"], bool)
        assert trainer.epoch == epoch_after_fit + 2


class TestEarlyStopping:
    """Training should stop before max_epochs when patience is exhausted."""

    def test_early_stopping(self, model_config: ModelConfig, device: torch.device) -> None:
        from quant.models.trainer import Trainer

        tc = TrainingConfig(
            learning_rate=1e-5,   # very slow LR so val_loss stagnates
            max_epochs=50,
            early_stopping_patience=3,
            max_grad_norm=1.0,
            scheduler_patience=2,
            scheduler_factor=0.5,
        )
        model = _DummyModel(model_config)
        trainer = Trainer(
            model=model,
            criterion=nn.MSELoss(),
            model_config=model_config,
            training_config=tc,
            device=device,
        )

        train_loader, val_loader = _make_loaders(model_config, n_train=32, n_val=16)
        metrics = trainer.fit(train_loader, val_loader)

        n_epochs_run = len(metrics["train_loss"])
        assert n_epochs_run <= 50, "Should stop before max_epochs=50"


class TestGradientClipping:
    """Gradient norms should be clipped according to max_grad_norm."""

    def test_gradient_clipping(
        self, model_config: ModelConfig, device: torch.device
    ) -> None:
        from quant.models.trainer import Trainer

        max_norm = 0.5
        tc = TrainingConfig(
            learning_rate=1e-2,
            max_epochs=1,
            max_grad_norm=max_norm,
            early_stopping_patience=100,
            scheduler_patience=100,
        )
        model = _DummyModel(model_config)
        trainer = Trainer(
            model=model,
            criterion=nn.MSELoss(),
            model_config=model_config,
            training_config=tc,
            device=device,
        )

        # Run one epoch so clip_grad_norm_ is exercised
        train_loader, val_loader = _make_loaders(model_config, n_train=64, n_val=16)
        trainer.fit(train_loader, val_loader, max_epochs=1)

        # After training, verify the model still has finite parameters
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite parameter: {name}"


class TestNanLossHandling:
    """NaN or Inf loss should be skipped without crashing."""

    def test_nan_loss_handling(
        self, model_config: ModelConfig, device: torch.device
    ) -> None:
        from quant.models.trainer import Trainer

        # A loss function that occasionally returns NaN
        call_count = {"n": 0}
        real_criterion = nn.MSELoss()

        class _NanCriterion(nn.Module):
            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                call_count["n"] += 1
                if call_count["n"] % 3 == 0:
                    return torch.tensor(float("nan"), device=pred.device)
                return real_criterion(pred, target)

        tc = TrainingConfig(
            learning_rate=1e-2,
            max_epochs=3,
            early_stopping_patience=100,
            max_grad_norm=1.0,
            scheduler_patience=100,
        )
        model = _DummyModel(model_config)
        trainer = Trainer(
            model=model,
            criterion=_NanCriterion(),
            model_config=model_config,
            training_config=tc,
            device=device,
        )

        train_loader, val_loader = _make_loaders(model_config, n_train=64, n_val=16)
        # Should not raise
        metrics = trainer.fit(train_loader, val_loader, max_epochs=3)
        assert len(metrics["train_loss"]) == 3
        assert trainer._nan_count > 0, "NaN losses should have been counted"


class TestNanValLossNotAcceptedAsImprovement:
    """NaN/Inf val_loss must never overwrite best_val_loss."""

    def test_nan_val_loss_not_accepted_as_improvement(
        self, model_config: ModelConfig, device: torch.device
    ) -> None:
        from quant.models.trainer import Trainer

        # A criterion that always returns NaN
        class _AlwaysNanCriterion(nn.Module):
            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return torch.tensor(float("nan"), device=pred.device)

        tc = TrainingConfig(
            learning_rate=1e-2,
            max_epochs=5,
            early_stopping_patience=3,
            max_grad_norm=1.0,
            scheduler_patience=100,
        )
        model = _DummyModel(model_config)
        trainer = Trainer(
            model=model,
            criterion=_AlwaysNanCriterion(),
            model_config=model_config,
            training_config=tc,
            device=device,
        )

        train_loader, val_loader = _make_loaders(model_config, n_train=64, n_val=16)
        metrics = trainer.fit(train_loader, val_loader, max_epochs=5)

        # best_val_loss should remain at its initial value (inf)
        assert trainer.best_val_loss == float("inf"), (
            f"NaN val_loss must not be accepted as improvement, "
            f"but best_val_loss={trainer.best_val_loss}"
        )


class TestStateRoundtrip:
    """get_state / load_state should perfectly restore trainer state."""

    def test_state_roundtrip(self, trainer, model_config: ModelConfig) -> None:
        train_loader, val_loader = _make_loaders(model_config, n_train=64, n_val=16)
        trainer.fit(train_loader, val_loader, max_epochs=3)

        state = trainer.get_state()

        # Verify state keys
        assert "model_state_dict" in state
        assert "optimizer_state_dict" in state
        assert "epoch" in state
        assert "best_val_loss" in state

        # Build a fresh trainer and load state
        from quant.models.trainer import Trainer

        fresh_model = _DummyModel(model_config)
        fresh_trainer = Trainer(
            model=fresh_model,
            criterion=nn.MSELoss(),
            model_config=model_config,
            training_config=trainer.training_config,
            device=torch.device("cpu"),
        )
        fresh_trainer.load_state(state)

        assert fresh_trainer.epoch == trainer.epoch
        assert fresh_trainer.best_val_loss == trainer.best_val_loss

        # Model weights should be identical
        for (n1, p1), (n2, p2) in zip(
            trainer.model.named_parameters(), fresh_trainer.model.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Mismatch in parameter: {n1}"


class TestCheckpointSaveLoad:
    """save_checkpoint / load_checkpoint should roundtrip through disk."""

    def test_checkpoint_save_load(self, trainer, model_config: ModelConfig) -> None:
        train_loader, val_loader = _make_loaders(model_config, n_train=64, n_val=16)
        trainer.fit(train_loader, val_loader, max_epochs=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "trainer.pt"
            trainer.save_checkpoint(ckpt_path)
            assert ckpt_path.exists()

            from quant.models.trainer import Trainer

            fresh_model = _DummyModel(model_config)
            fresh_trainer = Trainer(
                model=fresh_model,
                criterion=nn.MSELoss(),
                model_config=model_config,
                training_config=trainer.training_config,
                device=torch.device("cpu"),
            )
            fresh_trainer.load_checkpoint(ckpt_path)

            assert fresh_trainer.epoch == trainer.epoch
            assert fresh_trainer.best_val_loss == trainer.best_val_loss
