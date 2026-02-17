"""Walk-forward model trainer with online-learning support."""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from quant.config.settings import ModelConfig, TrainingConfig, get_device
from quant.models.base import BaseModel


class Trainer:
    """Walk-forward trainer with early stopping, NaN handling, and MPS OOM fallback.

    Parameters
    ----------
    model : BaseModel
        The model to train (must be a subclass of :class:`BaseModel`).
    criterion : nn.Module
        Loss function.
    model_config : ModelConfig
        Model configuration (stored in checkpoints).
    training_config : TrainingConfig
        Training hyper-parameters.
    device : torch.device | str | None
        Device to train on.  Defaults to auto-detect.
    """

    def __init__(
        self,
        model: BaseModel,
        criterion: nn.Module,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = torch.device(device) if isinstance(device, str) else (device or get_device())
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.model_config = model_config
        self.training_config = training_config

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=training_config.scheduler_patience,
            factor=training_config.scheduler_factor,
        )

        # Tracking
        self.epoch: int = 0
        self.best_val_loss: float = float("inf")
        self._nan_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_device(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(t.to(self.device) for t in tensors)

    def _fallback_to_cpu(self) -> None:
        """Move model and optimiser state to CPU after an MPS/CUDA OOM."""
        if self.device.type == "cpu":
            return
        logger.warning(
            "OOM on {} -- falling back to CPU.  Training will be slower.",
            self.device,
        )
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        # Re-create optimizer so its internal state tensors are on CPU
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer.param_groups[0]["lr"],
            weight_decay=self.training_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.training_config.scheduler_patience,
            factor=self.training_config.scheduler_factor,
        )

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            x, y = self._to_device(batch[0], batch[1])

            try:
                pred = self.model(x)
                loss = self.criterion(pred, y)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() or "MPS" in str(exc):
                    self._fallback_to_cpu()
                    x, y = self._to_device(x, y)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                else:
                    raise

            # NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                self._nan_count += 1
                logger.error(
                    "NaN/Inf loss detected (count={}) -- skipping gradient step.",
                    self._nan_count,
                )
                self.optimizer.zero_grad()
                continue

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.training_config.max_grad_norm,
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct_dir = 0
        total_dir = 0
        n_batches = 0

        for batch in loader:
            x, y = self._to_device(batch[0], batch[1])

            try:
                pred = self.model(x)
                loss = self.criterion(pred, y)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() or "MPS" in str(exc):
                    self._fallback_to_cpu()
                    x, y = self._to_device(x, y)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                else:
                    raise

            total_loss += loss.item()
            n_batches += 1

            # Directional accuracy (compare sign of first prediction step)
            if pred.dim() == 2 and pred.size(-1) >= 1 and y.dim() == 2 and y.size(-1) >= 1:
                pred_sign = torch.sign(pred[:, 0])
                true_sign = torch.sign(y[:, 0])
                correct_dir += (pred_sign == true_sign).sum().item()
                total_dir += pred_sign.numel()

        avg_loss = total_loss / max(n_batches, 1)
        dir_acc = correct_dir / max(total_dir, 1)
        return {"val_loss": avg_loss, "directional_accuracy": dir_acc}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int | None = None,
        patience: int | None = None,
    ) -> dict[str, Any]:
        """Train the model with early stopping.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        max_epochs : int | None
            Override ``TrainingConfig.max_epochs``.
        patience : int | None
            Override ``TrainingConfig.early_stopping_patience``.

        Returns
        -------
        dict
            ``{"train_loss": [...], "val_loss": [...], "best_epoch": int,
              "best_val_loss": float, "directional_accuracy": float}``
        """
        max_epochs = max_epochs or self.training_config.max_epochs
        patience = patience or self.training_config.early_stopping_patience

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_epoch = self.epoch
        best_dir_acc = 0.0
        patience_counter = 0

        logger.info(
            "Starting training: max_epochs={}, patience={}, device={}, lr={:.2e}",
            max_epochs, patience, self.device, self.optimizer.param_groups[0]["lr"],
        )

        t0 = time.time()

        for ep in range(1, max_epochs + 1):
            self.epoch += 1

            train_loss = self._train_one_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            val_loss = val_metrics["val_loss"]
            dir_acc = val_metrics["directional_accuracy"]

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch {:>4d} | train_loss={:.6f} | val_loss={:.6f} | "
                "dir_acc={:.2%} | lr={:.2e}",
                self.epoch, train_loss, val_loss, dir_acc, current_lr,
            )

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = self.epoch
                best_dir_acc = dir_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch {} (best epoch={})",
                        self.epoch, best_epoch,
                    )
                    break

        elapsed = time.time() - t0
        logger.info(
            "Training complete in {:.1f}s | best_epoch={} | best_val_loss={:.6f}",
            elapsed, best_epoch, self.best_val_loss,
        )

        return {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "best_epoch": best_epoch,
            "best_val_loss": self.best_val_loss,
            "directional_accuracy": best_dir_acc,
        }

    def continue_training(
        self,
        loader: DataLoader,
        epochs: int | None = None,
        lr_factor: float | None = None,
        val_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        """Continue / online-retrain the model on new data.

        Parameters
        ----------
        loader : DataLoader
            New training data.
        epochs : int | None
            Number of fine-tuning epochs (default: ``TrainingConfig.retrain_epochs``).
        lr_factor : float | None
            Multiply current LR by this factor for fine-tuning
            (default: ``TrainingConfig.retrain_lr_factor``).
        val_loader : DataLoader | None
            Optional validation loader to assess improvement.

        Returns
        -------
        dict
            ``{"train_loss": [...], "val_loss_before": float,
              "val_loss_after": float, "improved": bool}``
        """
        epochs = epochs or self.training_config.retrain_epochs
        lr_factor = lr_factor or self.training_config.retrain_lr_factor

        # Evaluate before retraining
        val_loss_before = float("inf")
        if val_loader is not None:
            val_metrics = self._validate(val_loader)
            val_loss_before = val_metrics["val_loss"]

        # Scale learning rate
        original_lr = self.optimizer.param_groups[0]["lr"]
        new_lr = original_lr * lr_factor
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr

        logger.info(
            "Online retraining: epochs={}, lr={:.2e} -> {:.2e}",
            epochs, original_lr, new_lr,
        )

        train_losses: list[float] = []
        for ep in range(1, epochs + 1):
            self.epoch += 1
            loss = self._train_one_epoch(loader)
            train_losses.append(loss)
            logger.debug("Retrain epoch {:>3d} | loss={:.6f}", ep, loss)

        # Evaluate after retraining
        val_loss_after = float("inf")
        if val_loader is not None:
            val_metrics = self._validate(val_loader)
            val_loss_after = val_metrics["val_loss"]

        # Restore original LR base
        for pg in self.optimizer.param_groups:
            pg["lr"] = original_lr

        improved = val_loss_after < val_loss_before

        logger.info(
            "Online retrain done | val_loss: {:.6f} -> {:.6f} | improved={}",
            val_loss_before, val_loss_after, improved,
        )

        return {
            "train_loss": train_losses,
            "val_loss_before": val_loss_before,
            "val_loss_after": val_loss_after,
            "improved": improved,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        """Return a serialisable dict of the full trainer state."""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "nan_count": self._nan_count,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore trainer from a state dict produced by :meth:`get_state`."""
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.epoch = state["epoch"]
        self.best_val_loss = state["best_val_loss"]
        self._nan_count = state.get("nan_count", 0)
        logger.info(
            "Trainer state restored (epoch={}, best_val_loss={:.6f})",
            self.epoch, self.best_val_loss,
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save the full trainer state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_state(), path)
        logger.info("Trainer checkpoint saved to {}", path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a trainer checkpoint from disk."""
        path = Path(path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state(state)
        logger.info("Trainer checkpoint loaded from {}", path)
