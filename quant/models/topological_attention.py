"""Topological Attention Network -- TDA-inspired multi-scale attention."""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig
from quant.models.base import BaseModel
from quant.models.transformer import CausalTransformerDecoderLayer, SinusoidalPositionalEncoding


class TopologicalAttentionNetwork(BaseModel):
    """Topological Data Analysis-inspired Transformer.

    Uses a pure-PyTorch TDA approximation: sliding-window distance matrices
    with multi-scale filtration and learnable thresholds to compute soft
    Betti-0 proxies (connected-component counts).  These topological features
    are concatenated with raw features before being fed into a Transformer
    encoder.

    Input:  ``[batch, seq_len, n_features]``
    Output: ``[batch, forecast_horizon]``
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        d_model = config.top_d_model
        n_heads = config.top_n_heads
        n_layers = config.top_n_layers
        d_ff = config.top_d_ff
        dropout = config.top_dropout
        window_size = config.top_window_size
        n_scales = config.top_n_scales

        self.window_size = window_size
        self.n_scales = n_scales

        # Learnable filtration thresholds per scale
        self.scale_params = nn.Parameter(torch.linspace(0.1, 2.0, n_scales))

        # Topological features: n_scales values per time step
        topo_features = n_scales
        total_features = config.n_features + topo_features

        # Input projection (raw features + topo features -> d_model)
        self.input_proj = nn.Linear(total_features, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=config.seq_len + 64, dropout=dropout,
        )

        # Transformer encoder layers (using CausalTransformerDecoderLayer for consistency)
        self.layers = nn.ModuleList(
            [CausalTransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, config.forecast_horizon)

        self._init_weights()

        logger.info(
            "TopologicalAttentionNetwork: d_model={}, heads={}, layers={}, "
            "window={}, scales={}, params={:,}",
            d_model, n_heads, n_layers, window_size, n_scales, self.count_parameters(),
        )

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if name == "scale_params":
                continue  # keep linspace initialization
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def _compute_topo_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute soft Betti-0 proxy features from sliding-window distance matrices.

        Parameters
        ----------
        x : Tensor
            ``[batch, seq_len, n_features]``

        Returns
        -------
        Tensor
            ``[batch, seq_len, n_scales]``
        """
        B, T, F = x.shape
        w = min(self.window_size, T)

        # Pad start with zeros so we get T outputs
        padded = nn.functional.pad(x, (0, 0, w - 1, 0))  # [B, T + w - 1, F]

        topo_out = []
        for t in range(T):
            window = padded[:, t : t + w, :]  # [B, w, F]

            # Pairwise L2 distance matrix [B, w, w]
            diff = window.unsqueeze(2) - window.unsqueeze(1)  # [B, w, 1, F] - [B, 1, w, F]
            dist = torch.norm(diff, dim=-1)  # [B, w, w]

            # Multi-scale soft Betti-0 proxy
            scale_features = []
            for s in range(self.n_scales):
                threshold = torch.abs(self.scale_params[s])
                # Soft connectivity: sigmoid of (threshold - dist)
                # High value -> points are connected at this scale
                connectivity = torch.sigmoid((threshold - dist) * 5.0)  # [B, w, w]
                # Betti-0 proxy: w - sum of connections / w (normalized)
                connected_count = connectivity.sum(dim=-1).mean(dim=-1)  # [B]
                scale_features.append(connected_count)

            topo_out.append(torch.stack(scale_features, dim=-1))  # [B, n_scales]

        return torch.stack(topo_out, dim=1)  # [B, T, n_scales]

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
        # Compute topological features
        topo = self._compute_topo_features(x)  # [B, T, n_scales]

        # Concatenate raw features with topo features
        h = torch.cat([x, topo], dim=-1)  # [B, T, n_features + n_scales]

        # Project to d_model
        h = self.input_proj(h)
        h = self.pos_enc(h)

        causal_mask = self._make_causal_mask(h.size(1), h.device)

        for layer in self.layers:
            h = layer(h, causal_mask)

        h = self.final_norm(h)

        # Last time-step -> forecast
        out = self.output_proj(h[:, -1, :])  # [B, forecast_horizon]
        return out
