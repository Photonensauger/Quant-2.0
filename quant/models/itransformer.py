"""iTransformer -- inverted Transformer for time-series forecasting.

Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective for
Time Series Forecasting" (ICLR 2024).

Key idea: transpose the input so that each *feature* becomes a token and the
temporal dimension becomes the embedding.  A standard encoder then captures
cross-variate dependencies while the temporal projection captures temporal
patterns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig
from quant.models.base import BaseModel


class _EncoderLayer(nn.Module):
    """Standard Transformer encoder layer (pre-norm)."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention (no causal mask -- full attention over features)
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        # Pre-norm feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class ITransformer(BaseModel):
    """Inverted Transformer for multivariate time-series forecasting.

    Pipeline
    --------
    1. Transpose input: ``[B, T, N]`` -> ``[B, N, T]``
       Each of the *N* features is now a token with embedding length *T*.
    2. Project temporal dim: ``T`` -> ``d_model``  (per-feature linear).
    3. Apply *L* standard Transformer encoder layers over the *N* tokens.
    4. Output projection: ``d_model`` -> ``forecast_horizon`` (per-feature).
    5. Aggregate across features -> ``[B, forecast_horizon]``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        d_model = config.itf_d_model
        n_heads = config.itf_n_heads
        n_layers = config.itf_n_layers
        d_ff = config.itf_d_ff
        dropout = config.itf_dropout
        seq_len = config.seq_len
        n_features = config.n_features
        forecast_horizon = config.forecast_horizon

        # Per-feature temporal embedding: seq_len -> d_model
        self.temporal_proj = nn.Linear(seq_len, d_model)

        # Learnable feature-level positional embedding
        self.feature_embed = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        # Encoder layers (full attention over feature dimension)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Per-feature output projection: d_model -> forecast_horizon
        self.output_proj = nn.Linear(d_model, forecast_horizon)

        # Aggregation across features -> single forecast vector
        self.aggregator = nn.Linear(n_features, 1)

        self._init_weights()

        logger.info(
            "ITransformer: d_model={}, heads={}, layers={}, n_features={}, params={:,}",
            d_model, n_heads, n_layers, n_features, self.count_parameters(),
        )

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        # 1. Transpose: [B, T, N] -> [B, N, T]
        x = x.transpose(1, 2)

        # 2. Temporal projection: [B, N, T] -> [B, N, d_model]
        h = self.temporal_proj(x)

        # Add feature positional embedding
        h = h + self.feature_embed

        # 3. Encoder layers over feature tokens
        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)

        # 4. Output projection: [B, N, d_model] -> [B, N, forecast_horizon]
        out = self.output_proj(h)

        # 5. Aggregate features: [B, N, forecast_horizon] -> [B, forecast_horizon]
        # Transpose to [B, forecast_horizon, N] then reduce N -> 1
        out = out.transpose(1, 2)  # [B, forecast_horizon, N]
        out = self.aggregator(out).squeeze(-1)  # [B, forecast_horizon]

        return out
