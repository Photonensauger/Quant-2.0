"""Tests for quant.core.self_trainer -- SelfTrainer orchestration loop.

All external dependencies (models, trainers, feature pipeline, ensemble,
risk manager, position sizer, executor) are mocked so the tests run
without network access or GPU.
"""

from __future__ import annotations

import signal as signal_mod
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from quant.config.settings import SystemConfig, TrainingConfig
from quant.core.self_trainer import SelfTrainer
from quant.core.state_manager import STATE_VERSION, SystemStateManager


# ---------------------------------------------------------------------------
# Helpers -- mock components
# ---------------------------------------------------------------------------

def _mock_model(n_features: int = 8) -> MagicMock:
    """Return a mock model with state_dict, load_state_dict, eval, __call__."""
    model = MagicMock()
    model.state_dict.return_value = {"w": torch.tensor([1.0, 2.0])}
    model.load_state_dict = MagicMock()
    model.eval = MagicMock()
    # Forward pass returns a single prediction
    model.__call__ = MagicMock(return_value=torch.tensor([[0.001]]))
    model.return_value = torch.tensor([[0.001]])
    return model


def _mock_trainer() -> MagicMock:
    """Return a mock trainer with get_state, load_state, continue_training."""
    trainer = MagicMock()
    trainer.get_state.return_value = {
        "model_state_dict": {"w": torch.tensor([1.0])},
        "optimizer_state_dict": None,
        "epoch": 5,
        "best_val_loss": 0.01,
        "model_config": {},
    }
    trainer.best_val_loss = 0.01
    trainer.epoch = 5
    trainer.load_state = MagicMock()
    trainer.continue_training.return_value = {
        "improved": True,
        "val_loss_before": 0.02,
        "val_loss_after": 0.01,
    }
    return trainer


def _mock_feature_pipeline() -> MagicMock:
    """Return a mock feature pipeline."""
    fp = MagicMock()
    fp.transform.return_value = (
        np.random.randn(60, 8).astype(np.float32),  # features
        np.random.randn(60).astype(np.float32),  # targets
    )
    fp.get_state.return_value = {
        "feature_names": ["f1", "f2"],
        "dropped_features": [],
        "rolling_mean": {},
        "rolling_std": {},
        "cpd_state": {},
    }
    fp.load_state = MagicMock()
    return fp


def _mock_ensemble() -> MagicMock:
    """Return a mock ensemble strategy."""
    ens = MagicMock()
    ens.generate_signals.return_value = []
    ens.get_state.return_value = {"weights": {"m1": 1.0}}
    ens.load_state = MagicMock()
    ens.update_weights = MagicMock()
    return ens


def _mock_risk_manager() -> MagicMock:
    """Return a mock risk manager that always passes."""
    rm = MagicMock()
    rm.check_all.return_value = (True, [])
    return rm


def _mock_position_sizer() -> MagicMock:
    ps = MagicMock()
    ps.calculate.return_value = 0.0
    return ps


def _mock_executor() -> MagicMock:
    ex = MagicMock()
    ex.get_portfolio_state.return_value = {
        "positions": {},
        "cash": 100_000.0,
        "equity": 100_000.0,
        "equity_curve": [100_000.0],
        "daily_pnl": 0.0,
    }
    ex.close_all_positions.return_value = []
    ex.execute.return_value = None
    return ex


def _make_bar_data(n_rows: int = 60) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame suitable for passing to process_bar."""
    rng = np.random.RandomState(42)
    close = 100.0 + rng.randn(n_rows).cumsum() * 0.1
    return pd.DataFrame(
        {
            "open": close + rng.randn(n_rows) * 0.01,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.lognormal(10, 0.5, n_rows),
        },
        index=pd.date_range("2024-01-01", periods=n_rows, freq="5min", tz="UTC"),
    )


def _build_trainer(
    tmp_path: Path,
    config_overrides: dict[str, Any] | None = None,
) -> SelfTrainer:
    """Build a SelfTrainer with all-mock components."""
    cfg = SystemConfig()
    cfg.state_dir = tmp_path / "state"
    cfg.model.seq_len = 16
    cfg.model.n_features = 8
    cfg.device = torch.device("cpu")

    if config_overrides:
        for key, val in config_overrides.items():
            setattr(cfg.training, key, val)

    return SelfTrainer(
        config=cfg,
        models={"m1": _mock_model()},
        trainers={"m1": _mock_trainer()},
        feature_pipeline=_mock_feature_pipeline(),
        ensemble=_mock_ensemble(),
        risk_manager=_mock_risk_manager(),
        position_sizer=_mock_position_sizer(),
        executor=_mock_executor(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessBarIncrementsCounter:

    def test_process_bar_increments_counter(self, tmp_path: Path) -> None:
        """Each call to process_bar increments bar_counter by 1."""
        trainer = _build_trainer(tmp_path)
        trainer._running = True

        assert trainer.bar_counter == 0

        result1 = trainer.process_bar(_make_bar_data())
        assert trainer.bar_counter == 1
        assert result1["bar_counter"] == 1

        result2 = trainer.process_bar(_make_bar_data())
        assert trainer.bar_counter == 2
        assert result2["bar_counter"] == 2


class TestRetrainTriggerInterval:

    def test_retrain_trigger_interval(self, tmp_path: Path) -> None:
        """Retraining triggers when bar_counter - last_retrain_bar >= retrain_interval."""
        trainer = _build_trainer(
            tmp_path,
            config_overrides={
                "retrain_interval": 5,
                "retrain_min_samples": 1,
                "checkpoint_interval": 99999,
            },
        )
        trainer._running = True

        # Process bars until the retrain interval
        results = []
        for _ in range(6):
            results.append(trainer.process_bar(_make_bar_data()))

        # At least one bar should have triggered retraining
        retrained = [r for r in results if r.get("retrained")]
        assert len(retrained) >= 1


class TestRetrainTriggerSharpe:

    def test_retrain_trigger_sharpe(self, tmp_path: Path) -> None:
        """Retraining triggers when rolling Sharpe falls below the threshold."""
        trainer = _build_trainer(
            tmp_path,
            config_overrides={
                "retrain_interval": 99999,  # disable interval trigger
                "retrain_min_samples": 1,
                "sharpe_threshold": 100.0,  # always below this
                "checkpoint_interval": 99999,
            },
        )
        trainer._running = True

        # Inject negative returns history to push Sharpe below threshold
        trainer._returns_history = [-0.05] * 250
        trainer._rolling_sharpe = -2.0
        trainer._samples_since_retrain = 200

        result = trainer.process_bar(_make_bar_data())

        # The Sharpe trigger should have fired
        assert result.get("retrained") is True


class TestCheckpointInterval:

    def test_checkpoint_interval(self, tmp_path: Path) -> None:
        """A checkpoint is saved when bar_counter is a multiple of checkpoint_interval."""
        trainer = _build_trainer(
            tmp_path,
            config_overrides={
                "checkpoint_interval": 3,
                "retrain_interval": 99999,
            },
        )
        trainer._running = True

        results = []
        for _ in range(6):
            results.append(trainer.process_bar(_make_bar_data()))

        # Bars 3 and 6 should have triggered checkpoints
        checkpointed = [r for r in results if r.get("checkpointed")]
        assert len(checkpointed) >= 1


class TestGetFullState:

    def test_get_full_state(self, tmp_path: Path) -> None:
        """get_full_state returns a dict with all expected top-level keys."""
        trainer = _build_trainer(tmp_path)
        trainer._running = True
        trainer.bar_counter = 42

        state = trainer.get_full_state()

        expected_keys = {
            "version",
            "timestamp",
            "bar_counter",
            "models",
            "feature_pipeline",
            "bocpd",
            "ensemble",
            "performance",
            "training",
        }
        assert expected_keys.issubset(set(state.keys()))
        assert state["bar_counter"] == 42
        assert state["version"] == STATE_VERSION


class TestLoadFullState:

    def test_load_full_state(self, tmp_path: Path) -> None:
        """load_full_state restores bar_counter and performance state."""
        trainer = _build_trainer(tmp_path)

        saved_state = {
            "version": STATE_VERSION,
            "timestamp": "2024-01-01T00:00:00",
            "bar_counter": 500,
            "models": {
                "m1": {
                    "state_dict": {"w": torch.tensor([3.0, 4.0])},
                    "config": {},
                    "optimizer_state": None,
                    "epoch": 10,
                    "best_val_loss": 0.005,
                }
            },
            "feature_pipeline": {
                "feature_names": ["f1"],
                "dropped_features": [],
                "rolling_means": {},
                "rolling_stds": {},
            },
            "bocpd": {},
            "ensemble": {"weights": {"m1": 1.0}, "rolling_sharpes": {"m1": 1.5}},
            "performance": {
                "equity_history": [100_000.0, 101_000.0],
                "returns_history": [0.0, 0.01],
                "trade_history": [],
                "rolling_sharpe": 1.5,
            },
            "training": {
                "last_retrain_bar": 400,
                "retrain_count": 3,
                "samples_since_retrain": 100,
            },
        }

        trainer.load_full_state(saved_state)

        assert trainer.bar_counter == 500
        assert trainer._rolling_sharpe == 1.5
        assert trainer._last_retrain_bar == 400
        assert trainer._retrain_count == 3
        assert trainer._equity_history == [100_000.0, 101_000.0]


class TestShutdownSavesState:

    def test_shutdown_saves_state(self, tmp_path: Path) -> None:
        """_shutdown saves full state to disk via the state manager."""
        trainer = _build_trainer(tmp_path)
        trainer._running = True
        trainer.bar_counter = 77

        trainer._shutdown()

        assert trainer._running is False

        # State should have been saved
        mgr = SystemStateManager(trainer.config.state_dir)
        loaded = mgr.load()
        assert loaded is not None
        assert loaded["bar_counter"] == 77

    def test_shutdown_idempotent(self, tmp_path: Path) -> None:
        """Calling _shutdown multiple times does not raise."""
        trainer = _build_trainer(tmp_path)
        trainer._running = True

        trainer._shutdown()
        trainer._shutdown()  # second call should be a no-op

        assert trainer._running is False


class TestEnsembleWeightUpdate:

    def test_ensemble_weight_update(self, tmp_path: Path) -> None:
        """After process_bar, ensemble.update_weights is called."""
        trainer = _build_trainer(tmp_path)
        trainer._running = True

        trainer.process_bar(_make_bar_data())

        trainer.ensemble.update_weights.assert_called_once()


class TestStartLoadsState:

    def test_start_loads_state(self, tmp_path: Path) -> None:
        """start() loads an existing checkpoint and restores bar_counter."""
        # First, save a checkpoint
        cfg = SystemConfig()
        cfg.state_dir = tmp_path / "state"
        cfg.model.seq_len = 16
        cfg.model.n_features = 8
        cfg.device = torch.device("cpu")

        mgr = SystemStateManager(cfg.state_dir)
        saved_state = {
            "version": STATE_VERSION,
            "timestamp": "2024-06-01T00:00:00",
            "bar_counter": 250,
            "models": {},
            "feature_pipeline": {},
            "bocpd": {},
            "ensemble": {},
            "performance": {
                "equity_history": [],
                "returns_history": [],
                "trade_history": [],
                "rolling_sharpe": 0.0,
            },
            "training": {
                "last_retrain_bar": 200,
                "retrain_count": 1,
                "samples_since_retrain": 50,
            },
        }
        mgr.save(saved_state)

        # Build a trainer pointing at the same state dir
        trainer = SelfTrainer(
            config=cfg,
            models={"m1": _mock_model()},
            trainers={"m1": _mock_trainer()},
            feature_pipeline=_mock_feature_pipeline(),
            ensemble=_mock_ensemble(),
            risk_manager=_mock_risk_manager(),
            position_sizer=_mock_position_sizer(),
            executor=_mock_executor(),
        )

        # Patch signal handlers to avoid interfering with test runner
        with patch("signal.signal"):
            trainer.start()

        assert trainer.bar_counter == 250
        assert trainer._running is True
        assert trainer._last_retrain_bar == 200

        # Clean up
        trainer._running = False

    def test_start_fresh_when_no_state(self, tmp_path: Path) -> None:
        """start() on an empty state dir starts fresh with bar_counter=0."""
        trainer = _build_trainer(tmp_path)

        with patch("signal.signal"):
            trainer.start()

        assert trainer.bar_counter == 0
        assert trainer._running is True

        trainer._running = False
