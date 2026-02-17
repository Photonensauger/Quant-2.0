"""Causal Discovery Transformer -- learns a causal adjacency matrix over features."""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig
from quant.models.base import BaseModel
from quant.models.transformer import CausalTransformerDecoderLayer, SinusoidalPositionalEncoding


class CausalDiscoveryTransformer(BaseModel):
    """Transformer with a learned causal adjacency matrix for feature interactions.

    A soft ``[N, N]`` adjacency matrix (sigmoid-gated) modulates the input
    features via Granger-causal gating before feeding into a standard causal
    Transformer decoder.  The adjacency matrix is a learnable parameter that
    can be inspected after training for interpretability.

    Input:  ``[batch, seq_len, n_features]``
    Output: ``[batch, forecast_horizon]``
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        d_model = config.cdt_d_model
        n_heads = config.cdt_n_heads
        n_layers = config.cdt_n_layers
        d_ff = config.cdt_d_ff
        dropout = config.cdt_dropout
        n_features = config.n_features

        # Learnable causal adjacency matrix [N, N]
        self.causal_adj = nn.Parameter(torch.zeros(n_features, n_features))

        # Input projection (after causal gating)
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=config.seq_len + 64, dropout=dropout,
        )

        # Causal decoder layers
        self.layers = nn.ModuleList(
            [CausalTransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, config.forecast_horizon)

        self._init_weights()

        logger.info(
            "CausalDiscoveryTransformer: d_model={}, heads={}, layers={}, n_features={}, params={:,}",
            d_model, n_heads, n_layers, n_features, self.count_parameters(),
        )

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if name == "causal_adj":
                nn.init.zeros_(p)
            elif p.dim() > 1:
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
        # Causal gating: x_gated = x @ sigmoid(causal_adj)
        # sigmoid(causal_adj) is [N, N], acts as soft feature interaction gate
        gate = torch.sigmoid(self.causal_adj)  # [N, N]
        x_gated = torch.matmul(x, gate)  # [B, T, N]

        # Project to d_model
        h = self.input_proj(x_gated)
        h = self.pos_enc(h)

        causal_mask = self._make_causal_mask(h.size(1), h.device)

        for layer in self.layers:
            h = layer(h, causal_mask)

        h = self.final_norm(h)

        # Last time-step -> forecast
        out = self.output_proj(h[:, -1, :])  # [B, forecast_horizon]
        return out
