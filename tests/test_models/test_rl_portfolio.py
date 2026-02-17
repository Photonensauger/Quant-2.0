"""Tests for quant.models.rl_portfolio – PPOAgent."""

from __future__ import annotations

import pytest
import torch

from quant.config.settings import ModelConfig
from quant.models.rl_portfolio import PPOAgent, ActionResult, EvalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def config() -> ModelConfig:
    return ModelConfig(
        ppo_hidden_size=32,
        ppo_n_layers=1,
    )


@pytest.fixture()
def agent(config: ModelConfig, device: torch.device) -> PPOAgent:
    state_dim = 16
    n_assets = 4
    model = PPOAgent(config, state_dim=state_dim, n_assets=n_assets)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelectActionShape:
    """select_action should return tensors with the correct shapes."""

    def test_select_action_shape(self, agent: PPOAgent, device: torch.device) -> None:
        state = torch.randn(agent.state_dim, device=device)
        result = agent.select_action(state)

        assert isinstance(result, ActionResult)
        assert result.action.shape == (agent.n_assets,)
        assert result.log_prob.dim() == 0, "log_prob should be a scalar"
        assert result.value.dim() == 0, "value should be a scalar"

    def test_select_action_shape_batched_input(self, agent: PPOAgent, device: torch.device) -> None:
        state = torch.randn(1, agent.state_dim, device=device)
        result = agent.select_action(state)

        assert result.action.shape == (agent.n_assets,)
        assert result.log_prob.dim() == 0
        assert result.value.dim() == 0


class TestActionSumsToOne:
    """Portfolio weights from select_action should sum to 1 (softmax output)."""

    def test_action_sums_to_one(self, agent: PPOAgent, device: torch.device) -> None:
        state = torch.randn(agent.state_dim, device=device)
        result = agent.select_action(state)

        assert torch.allclose(
            result.action.sum(), torch.tensor(1.0, device=device), atol=1e-5
        ), f"Weights should sum to 1.0, got {result.action.sum().item()}"

    def test_action_non_negative(self, agent: PPOAgent, device: torch.device) -> None:
        state = torch.randn(agent.state_dim, device=device)
        result = agent.select_action(state)

        assert (result.action >= 0.0).all(), "Softmax weights must be non-negative"


class TestEvaluateActionsShape:
    """evaluate_actions should return correctly shaped outputs."""

    def test_evaluate_actions_shape(self, agent: PPOAgent, device: torch.device) -> None:
        batch_size = 8
        states = torch.randn(batch_size, agent.state_dim, device=device)
        actions = torch.randn(batch_size, agent.n_assets, device=device)

        result = agent.evaluate_actions(states, actions)

        assert isinstance(result, EvalResult)
        assert result.log_probs.shape == (batch_size,)
        assert result.values.shape == (batch_size,)
        assert result.entropy.dim() == 0, "entropy should be a scalar"


class TestEntropyPositive:
    """Entropy of the Normal policy should be positive."""

    def test_entropy_positive(self, agent: PPOAgent, device: torch.device) -> None:
        batch_size = 8
        states = torch.randn(batch_size, agent.state_dim, device=device)
        actions = torch.randn(batch_size, agent.n_assets, device=device)

        result = agent.evaluate_actions(states, actions)
        assert result.entropy.item() > 0.0, (
            f"Entropy should be strictly positive, got {result.entropy.item()}"
        )


class TestGradientFlows:
    """Gradients should propagate through the actor and critic heads."""

    def test_gradient_flows(self, config: ModelConfig, device: torch.device) -> None:
        agent = PPOAgent(config, state_dim=16, n_assets=4).to(device)
        agent.train()

        batch_size = 4
        states = torch.randn(batch_size, agent.state_dim, device=device)
        actions = torch.randn(batch_size, agent.n_assets, device=device)

        result = agent.evaluate_actions(states, actions)

        # Backprop through a combined objective
        loss = -result.log_probs.mean() + result.values.mean() - 0.01 * result.entropy
        loss.backward()

        # Check that all parameter gradients are non-None
        for name, param in agent.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in: {name}"
