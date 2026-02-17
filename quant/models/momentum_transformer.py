"""MomentumTransformer -- Transformer that outputs trading positions in [-1, +1].

Unlike the return-prediction transformers, this model directly outputs a
position signal suitable for momentum / trend-following strategies.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig
from quant.models.base import BaseModel
from quant.models.transformer import SinusoidalPositionalEncoding, CausalTransformerDecoderLayer


class MomentumTransformer(BaseModel):
    """Decoder-only Transformer that outputs a position in ``[-1, +1]`` via tanh.

    Input:  ``[batch, seq_len, n_features]``
    Output: ``[batch, 1]``  (position signal, -1 = full short, +1 = full long)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        d_model = config.mom_d_model
        n_heads = config.mom_n_heads
        n_layers = config.mom_n_layers
        d_ff = config.mom_d_ff
        dropout = config.mom_dropout

        # Input projection
        self.input_proj = nn.Linear(config.n_features, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=config.seq_len + 64, dropout=dropout,
        )

        # Causal decoder layers
        self.layers = nn.ModuleList(
            [CausalTransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Output projection -> scalar position via tanh
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )

        self._init_weights()

        logger.info(
            "MomentumTransformer: d_model={}, heads={}, layers={}, params={:,}",
            d_model, n_heads, n_layers, self.count_parameters(),
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
            ``[batch, 1]`` with values in ``[-1, +1]``
        """
        # [B, T, n_features] -> [B, T, d_model]
        h = self.input_proj(x)
        h = self.pos_enc(h)

        causal_mask = self._make_causal_mask(h.size(1), h.device)

        for layer in self.layers:
            h = layer(h, causal_mask)

        h = self.final_norm(h)

        # Last time-step -> position
        position = self.output_proj(h[:, -1, :])  # [B, 1]
        return position
