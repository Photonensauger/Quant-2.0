"""Abstract base model with checkpoint save/load."""

from __future__ import annotations

import abc
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig, get_device


class BaseModel(nn.Module, abc.ABC):
    """Abstract base for all prediction models.

    Subclasses must implement ``forward(x)`` where
    *x* has shape ``[batch, seq_len, n_features]``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Forward contract
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``[batch, seq_len, n_features]``.

        Returns
        -------
        Tensor
            Prediction tensor whose shape depends on the concrete model.
        """

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def save(
        self,
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
        best_val_loss: float = float("inf"),
        extras: dict[str, Any] | None = None,
    ) -> Path:
        """Persist model checkpoint to *path*.

        The checkpoint dict always contains:
        - ``model_state_dict``
        - ``config`` (serialised :class:`ModelConfig`)
        - ``optimizer_state`` (if an optimizer is supplied)
        - ``epoch``
        - ``best_val_loss``
        - ``model_class`` (fully-qualified class name for safe reloading)
        - any additional key/value pairs passed via *extras*
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_class": f"{type(self).__module__}.{type(self).__qualname__}",
        }
        if extras:
            checkpoint.update(extras)

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to {} (epoch={}, val_loss={:.6f})", path, epoch, best_val_loss)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
        map_location: str | torch.device | None = None,
    ) -> tuple["BaseModel", dict[str, Any]]:
        """Load a checkpoint and return ``(model, checkpoint_dict)``.

        Parameters
        ----------
        path : str | Path
            Path to the ``.pt`` / ``.pth`` checkpoint file.
        device : torch.device | str | None
            Device to place the model on.  Defaults to auto-detect.
        map_location : str | torch.device | None
            Passed to :func:`torch.load`.  Defaults to *device*.

        Returns
        -------
        tuple[BaseModel, dict]
            The reconstituted model and the raw checkpoint dict (useful for
            restoring the optimizer, epoch counter, etc.).
        """
        path = Path(path)
        if device is None:
            device = get_device()
        if map_location is None:
            map_location = device

        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        config = ModelConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        logger.info(
            "Model loaded from {} (epoch={}, val_loss={:.6f}) -> {}",
            path,
            checkpoint.get("epoch", "?"),
            checkpoint.get("best_val_loss", float("inf")),
            device,
        )
        return model, checkpoint

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{type(self).__name__}("
            f"params={self.count_parameters():,}, "
            f"device={next(self.parameters()).device})"
        )
