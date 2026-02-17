"""Hamiltonian Neural ODE -- symplectic leapfrog integrator for time-series."""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig
from quant.models.base import BaseModel


class HamiltonianNeuralODE(BaseModel):
    """Hamiltonian-inspired neural ODE using discrete symplectic (leapfrog) integration.

    The input is projected into a phase space of position ``q`` and momentum
    ``p`` coordinates.  Learned gradient networks approximate ``dH/dq`` and
    ``dH/dp``, and a leapfrog integrator evolves the phase space forward.
    No external ODE solver (``torchdiffeq``) is required.

    Input:  ``[batch, seq_len, n_features]``
    Output: ``[batch, forecast_horizon]``
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        d_model = config.hno_d_model
        n_leapfrog_steps = config.hno_n_leapfrog_steps
        dropout = config.hno_dropout
        hidden_size = config.hno_hidden_size

        assert d_model % 2 == 0, f"hno_d_model must be even, got {d_model}"

        self.d_model = d_model
        self.n_leapfrog_steps = n_leapfrog_steps
        self.half_d = d_model // 2

        # Input projection: [B, T, n_features] -> [B, d_model] (via mean-pool over time)
        self.input_proj = nn.Sequential(
            nn.Linear(config.n_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learned gradient networks (instead of autograd.grad on a Hamiltonian)
        # grad_q_net: dH/dq -> used to update p
        self.grad_q_net = nn.Sequential(
            nn.Linear(self.half_d, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.half_d),
        )

        # grad_p_net: dH/dp -> used to update q
        self.grad_p_net = nn.Sequential(
            nn.Linear(self.half_d, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.half_d),
        )

        # Step size (learnable)
        self.dt = nn.Parameter(torch.tensor(0.1))

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, config.forecast_horizon)

        self._init_weights()

        logger.info(
            "HamiltonianNeuralODE: d_model={}, leapfrog_steps={}, hidden={}, params={:,}",
            d_model, n_leapfrog_steps, hidden_size, self.count_parameters(),
        )

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if name == "dt":
                continue  # keep scalar init
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _leapfrog_step(self, q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One symplectic leapfrog integration step.

        Parameters
        ----------
        q, p : Tensor
            Position and momentum, each ``[B, half_d]``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated (q, p).
        """
        dt = self.dt

        # Half-step momentum update: p <- p - (dt/2) * dH/dq
        p = p - (dt / 2) * self.grad_q_net(q)

        # Full-step position update: q <- q + dt * dH/dp
        q = q + dt * self.grad_p_net(p)

        # Half-step momentum update: p <- p - (dt/2) * dH/dq
        p = p - (dt / 2) * self.grad_q_net(q)

        return q, p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            ``[batch, seq_len, n_features]``

        Returns
        -------
        Tensor
            ``[batch, forecast_horizon]``
        """
        # Project and pool over time
        h = self.input_proj(x)  # [B, T, d_model]
        h = h.mean(dim=1)  # [B, d_model]

        # Split into position and momentum
        q = h[:, : self.half_d]  # [B, half_d]
        p = h[:, self.half_d :]  # [B, half_d]

        # Symplectic leapfrog integration
        for _ in range(self.n_leapfrog_steps):
            q, p = self._leapfrog_step(q, p)

        # Recombine phase space
        h_out = torch.cat([q, p], dim=-1)  # [B, d_model]
        h_out = self.output_norm(h_out)

        # Final projection
        out = self.output_proj(h_out)  # [B, forecast_horizon]
        return out
