"""Tests for quant.models.losses – DifferentiableSharpeRatio, DirectionalLoss, CombinedLoss."""

from __future__ import annotations

import pytest
import torch

from quant.models.losses import (
    CombinedLoss,
    DifferentiableSharpeRatio,
    DirectionalLoss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# DifferentiableSharpeRatio
# ---------------------------------------------------------------------------

class TestSharpeScalar:
    """The Sharpe loss should return a single scalar."""

    def test_sharpe_loss_scalar(self, device: torch.device) -> None:
        loss_fn = DifferentiableSharpeRatio().to(device)
        positions = torch.randn(4, 50, device=device)
        returns = torch.randn(4, 50, device=device) * 0.01
        loss = loss_fn(positions, returns)

        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


class TestSharpeNegativeForGoodStrategy:
    """A strategy that is consistently profitable should yield negative loss
    (negative Sharpe = lower = better for minimisation)."""

    def test_sharpe_loss_negative_for_good_strat(self, device: torch.device) -> None:
        loss_fn = DifferentiableSharpeRatio().to(device)

        # Construct a clearly profitable strategy: positions always match return sign.
        torch.manual_seed(42)
        returns = torch.randn(2, 100, device=device) * 0.01
        # Position = sign of returns (perfect foresight)
        positions = torch.sign(returns)

        loss = loss_fn(positions, returns)
        # Perfect-foresight strategy -> high positive Sharpe -> negative loss
        assert loss.item() < 0.0, (
            f"Perfect-foresight strategy should give negative Sharpe loss, got {loss.item()}"
        )


# ---------------------------------------------------------------------------
# DirectionalLoss
# ---------------------------------------------------------------------------

class TestDirectionalLossZeroForCorrect:
    """When predictions have the same sign as targets, directional loss is zero."""

    def test_directional_loss_zero_for_correct(self, device: torch.device) -> None:
        loss_fn = DirectionalLoss(penalty_weight=1.0).to(device)

        predictions = torch.tensor([[0.5, -0.3, 0.1]], device=device)
        targets = torch.tensor([[0.2, -0.8, 0.05]], device=device)  # same signs

        loss = loss_fn(predictions, targets)
        assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-7), (
            f"Expected 0.0, got {loss.item()}"
        )

    def test_directional_loss_positive_for_wrong(self, device: torch.device) -> None:
        loss_fn = DirectionalLoss(penalty_weight=1.0).to(device)

        predictions = torch.tensor([[0.5, -0.3]], device=device)
        targets = torch.tensor([[-0.2, 0.8]], device=device)  # opposite signs

        loss = loss_fn(predictions, targets)
        assert loss.item() > 0.0, "Wrong direction should incur a positive penalty"


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------

class TestCombinedLossWeightedSum:
    """CombinedLoss should equal mse_weight * MSE + directional_weight * DirectionalLoss."""

    def test_combined_loss_weighted_sum(self, device: torch.device) -> None:
        mse_w = 1.0
        dir_w = 0.5
        combined_fn = CombinedLoss(
            mse_weight=mse_w,
            directional_weight=dir_w,
            directional_penalty=1.0,
        ).to(device)

        preds = torch.tensor([[0.5, -0.3, 0.1]], device=device)
        targets = torch.tensor([[-0.2, -0.8, 0.05]], device=device)

        combined_loss = combined_fn(preds, targets)

        # Manually compute components
        mse_val = torch.nn.MSELoss()(preds, targets)
        dir_val = DirectionalLoss(penalty_weight=1.0)(preds, targets)
        expected = mse_w * mse_val + dir_w * dir_val

        assert torch.allclose(combined_loss, expected, atol=1e-6), (
            f"CombinedLoss={combined_loss.item()}, expected={expected.item()}"
        )


# ---------------------------------------------------------------------------
# Gradient through Sharpe
# ---------------------------------------------------------------------------

class TestGradientThroughSharpe:
    """Gradients should flow through the differentiable Sharpe loss."""

    def test_gradient_through_sharpe(self, device: torch.device) -> None:
        loss_fn = DifferentiableSharpeRatio().to(device)

        positions = torch.randn(2, 50, device=device, requires_grad=True)
        returns = torch.randn(2, 50, device=device)

        loss = loss_fn(positions, returns)
        loss.backward()

        assert positions.grad is not None, "Gradients should flow to positions"
        assert not torch.isnan(positions.grad).any(), "Gradients contain NaN"
        assert positions.grad.shape == positions.shape
