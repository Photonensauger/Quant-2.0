"""Schroedinger Transformer -- quantum-inspired regime superposition model."""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig
from quant.models.base import BaseModel
from quant.models.transformer import CausalTransformerDecoderLayer, SinusoidalPositionalEncoding


class SchrodingerTransformer(BaseModel):
    """Quantum-inspired Transformer with parallel regime branches.

    Multiple Transformer branches ("regime wavefunctions") process the input
    in parallel.  A wavefunction network computes regime probabilities from
    the input summary.  The final output is the softmax-weighted combination
    of branch outputs -- a "wavefunction collapse" at inference time.

    Input:  ``[batch, seq_len, n_features]``
    Output: ``[batch, forecast_horizon]``
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        d_model = config.sqt_d_model
        n_heads = config.sqt_n_heads
        n_layers = config.sqt_n_layers
        d_ff = config.sqt_d_ff
        dropout = config.sqt_dropout
        n_regimes = config.sqt_n_regimes

        self.n_regimes = n_regimes

        # Shared input projection
        self.input_proj = nn.Linear(config.n_features, d_model)

        # Shared positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=config.seq_len + 64, dropout=dropout,
        )

        # Per-regime Transformer branches
        self.regime_branches = nn.ModuleList()
        for _ in range(n_regimes):
            branch = nn.ModuleList(
                [CausalTransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
            )
            self.regime_branches.append(branch)

        self.final_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_regimes)])

        # Per-regime output heads
        self.regime_heads = nn.ModuleList(
            [nn.Linear(d_model, config.forecast_horizon) for _ in range(n_regimes)]
        )

        # Wavefunction network: input summary -> regime probabilities
        self.wavefunction = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_regimes),
        )

        # Store last regime probabilities for interpretability
        self.last_regime_probs: torch.Tensor | None = None

        self._init_weights()

        logger.info(
            "SchrodingerTransformer: d_model={}, heads={}, layers={}, regimes={}, params={:,}",
            d_model, n_heads, n_layers, n_regimes, self.count_parameters(),
        )

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

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
        # Shared input embedding
        h = self.input_proj(x)  # [B, T, d_model]
        h = self.pos_enc(h)

        causal_mask = self._make_causal_mask(h.size(1), h.device)

        # Compute regime probabilities from input summary (mean-pool)
        input_summary = h.mean(dim=1)  # [B, d_model]
        regime_logits = self.wavefunction(input_summary)  # [B, n_regimes]
        regime_probs = torch.softmax(regime_logits, dim=-1)  # [B, n_regimes]
        self.last_regime_probs = regime_probs.detach()

        # Run each regime branch
        branch_outputs = []
        for i, (branch, norm, head) in enumerate(
            zip(self.regime_branches, self.final_norms, self.regime_heads)
        ):
            h_branch = h
            for layer in branch:
                h_branch = layer(h_branch, causal_mask)
            h_branch = norm(h_branch)
            out_i = head(h_branch[:, -1, :])  # [B, forecast_horizon]
            branch_outputs.append(out_i)

        # Stack: [B, n_regimes, forecast_horizon]
        stacked = torch.stack(branch_outputs, dim=1)

        # Wavefunction collapse: weighted sum over regimes
        weights = regime_probs.unsqueeze(-1)  # [B, n_regimes, 1]
        out = (stacked * weights).sum(dim=1)  # [B, forecast_horizon]

        return out
