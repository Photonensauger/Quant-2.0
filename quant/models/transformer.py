"""Decoder-only Transformer for time-series forecasting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig
from quant.models.base import BaseModel


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x* of shape ``[B, T, D]``."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CausalTransformerDecoderLayer(nn.Module):
    """Single decoder layer with causal (masked) self-attention + feed-forward."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm causal self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            is_causal=False,  # we pass the explicit mask
        )
        x = x + self.dropout(attn_out)

        # Pre-norm feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class DecoderTransformer(BaseModel):
    """Decoder-only Transformer for time-series return prediction.

    Input:  ``[batch, seq_len, n_features]``
    Output: ``[batch, forecast_horizon]``
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        d_model = config.tf_d_model
        n_heads = config.tf_n_heads
        n_layers = config.tf_n_layers
        d_ff = config.tf_d_ff
        dropout = config.tf_dropout

        # Input projection: n_features -> d_model
        self.input_proj = nn.Linear(config.n_features, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=config.seq_len + 64, dropout=dropout)

        # Decoder layers
        self.layers = nn.ModuleList(
            [CausalTransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Output projection: take last token -> forecast_horizon
        self.output_proj = nn.Linear(d_model, config.forecast_horizon)

        self._init_weights()

        logger.info(
            "DecoderTransformer: d_model={}, heads={}, layers={}, params={:,}",
            d_model, n_heads, n_layers, self.count_parameters(),
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # Causal mask
    # ------------------------------------------------------------------
    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask filled with ``-inf`` (additive mask)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        # [B, T, n_features] -> [B, T, d_model]
        h = self.input_proj(x)
        h = self.pos_enc(h)

        causal_mask = self._make_causal_mask(h.size(1), h.device)

        for layer in self.layers:
            h = layer(h, causal_mask)

        h = self.final_norm(h)

        # Use last time-step representation
        out = self.output_proj(h[:, -1, :])  # [B, forecast_horizon]
        return out
