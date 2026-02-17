"""Custom loss functions for quantitative trading models."""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger


class DifferentiableSharpeRatio(nn.Module):
    """Differentiable negative Sharpe ratio loss.

    Given position signals and realised returns, compute the portfolio's Sharpe
    ratio in a fully differentiable way and return its negation (since we
    minimise losses).

    Forward
    -------
    positions : Tensor ``[B, T]``
        Position sizing in ``[-1, +1]`` for each time step.
    returns : Tensor ``[B, T]``
        Realised asset returns for each time step.

    Returns
    -------
    Tensor
        Scalar: negative Sharpe ratio (lower = better Sharpe).
    """

    def __init__(self, annualisation_factor: float = 252.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.annualisation_factor = annualisation_factor
        self.eps = eps

    def forward(self, positions: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        # Portfolio returns: element-wise product of positions and asset returns
        portfolio_returns = positions * returns  # [B, T]

        # Mean and std across time dimension
        mean_ret = portfolio_returns.mean(dim=-1)  # [B]
        std_ret = portfolio_returns.std(dim=-1)  # [B]

        # Sharpe ratio per sample
        sharpe = mean_ret / (std_ret + self.eps)  # [B]

        # Annualise
        sharpe = sharpe * (self.annualisation_factor ** 0.5)

        # Return negative mean Sharpe across batch (lower = better)
        return -sharpe.mean()


class DirectionalLoss(nn.Module):
    """Penalises wrong direction of prediction.

    Adds a penalty when the predicted return has the wrong sign relative to
    the actual return.  The penalty is proportional to the magnitude of the
    actual return (larger missed moves are penalised more).

    Forward
    -------
    predictions : Tensor ``[B, H]``
        Predicted returns.
    targets : Tensor ``[B, H]``
        Actual returns.

    Returns
    -------
    Tensor
        Scalar directional loss.
    """

    def __init__(self, penalty_weight: float = 1.0) -> None:
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Sign agreement: +1 when same sign, -1 when different
        sign_agreement = torch.sign(predictions) * torch.sign(targets)  # [B, H]

        # Penalty for wrong direction: where sign_agreement < 0
        wrong_dir_mask = (sign_agreement < 0).float()

        # Penalty proportional to target magnitude
        penalty = wrong_dir_mask * targets.abs()

        return self.penalty_weight * penalty.mean()


class CombinedLoss(nn.Module):
    """Weighted sum of MSE loss and directional loss.

    ``loss = mse_weight * MSE(pred, target) + dir_weight * DirectionalLoss(pred, target)``

    Parameters
    ----------
    mse_weight : float
        Weight for the MSE component.
    directional_weight : float
        Weight for the directional component.
    directional_penalty : float
        Penalty weight passed to :class:`DirectionalLoss`.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        directional_weight: float = 0.5,
        directional_penalty: float = 1.0,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.directional_weight = directional_weight
        self.mse_loss = nn.MSELoss()
        self.directional_loss = DirectionalLoss(penalty_weight=directional_penalty)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(predictions, targets)
        directional = self.directional_loss(predictions, targets)
        total = self.mse_weight * mse + self.directional_weight * directional
        return total
