"""PPO Actor-Critic agent for portfolio optimisation."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig


class ActionResult(NamedTuple):
    """Container returned by :meth:`PPOAgent.select_action`."""

    action: torch.Tensor  # [n_assets] portfolio weights
    log_prob: torch.Tensor  # scalar
    value: torch.Tensor  # scalar


class EvalResult(NamedTuple):
    """Container returned by :meth:`PPOAgent.evaluate_actions`."""

    log_probs: torch.Tensor  # [B]
    values: torch.Tensor  # [B]
    entropy: torch.Tensor  # scalar


def _build_mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """Construct a simple MLP with GELU activations."""
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(n_layers):
        layers.extend([nn.Linear(prev, hidden), nn.GELU()])
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PPOAgent(nn.Module):
    """Proximal Policy Optimisation agent for portfolio allocation.

    The agent receives a flattened state vector and outputs portfolio weights
    over *n_assets* using a Dirichlet-parameterised policy (via softmax of
    concentration parameters).

    Parameters
    ----------
    config : ModelConfig
        Pulls ``ppo_hidden_size``, ``ppo_n_layers``, ``ppo_clip_epsilon``,
        ``ppo_entropy_coeff``, ``ppo_value_coeff``.
    state_dim : int
        Dimensionality of the flattened observation.
    n_assets : int
        Number of portfolio assets (action dimension).
    """

    def __init__(self, config: ModelConfig, state_dim: int, n_assets: int) -> None:
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.n_assets = n_assets

        hidden = config.ppo_hidden_size
        n_layers = config.ppo_n_layers

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # Actor head -> concentration parameters for a Dirichlet / softmax policy
        self.actor_head = _build_mlp(hidden, hidden, n_assets, n_layers)

        # Critic head -> state value
        self.critic_head = _build_mlp(hidden, hidden, 1, n_layers)

        # Log standard deviation for exploration (learnable per-asset)
        self.log_std = nn.Parameter(torch.zeros(n_assets))

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "PPOAgent: state_dim={}, n_assets={}, hidden={}, layers={}, params={:,}",
            state_dim, n_assets, hidden, n_layers, n_params,
        )

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p, gain=0.01)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _features(self, state: torch.Tensor) -> torch.Tensor:
        return self.feature_net(state)

    def _actor(self, features: torch.Tensor) -> torch.distributions.Normal:
        """Return a Normal distribution over raw (pre-softmax) action logits."""
        mean = self.actor_head(features)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def _critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic_head(features).squeeze(-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_action(self, state: torch.Tensor) -> ActionResult:
        """Sample an action for a single state.

        Parameters
        ----------
        state : Tensor
            ``[state_dim]`` or ``[1, state_dim]``

        Returns
        -------
        ActionResult
            ``(action [n_assets], log_prob, value)``
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self._features(state)
        dist = self._actor(features)
        raw_action = dist.rsample()  # [1, n_assets]
        log_prob = dist.log_prob(raw_action).sum(dim=-1)  # scalar

        # Convert to portfolio weights via softmax
        action = torch.softmax(raw_action, dim=-1).squeeze(0)  # [n_assets]
        value = self._critic(features).squeeze(0)

        return ActionResult(action=action, log_prob=log_prob.squeeze(0), value=value)

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> EvalResult:
        """Evaluate a batch of state-action pairs.

        Parameters
        ----------
        states : Tensor
            ``[B, state_dim]``
        actions : Tensor
            ``[B, n_assets]`` -- the *raw* pre-softmax actions stored in the
            replay buffer.

        Returns
        -------
        EvalResult
            ``(log_probs [B], values [B], entropy scalar)``
        """
        features = self._features(states)
        dist = self._actor(features)
        values = self._critic(features)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        return EvalResult(log_probs=log_probs, values=values, entropy=entropy)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Return the critic's value estimate.

        Parameters
        ----------
        state : Tensor
            ``[B, state_dim]`` or ``[state_dim]``

        Returns
        -------
        Tensor
            ``[B]`` or scalar
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features = self._features(state)
        return self._critic(features)
