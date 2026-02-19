"""Multi-layer LSTM with multi-head self-attention on hidden states."""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig
from quant.models.base import BaseModel


class AttentionLSTM(BaseModel):
    """LSTM encoder with temporal self-attention for return prediction.

    Architecture
    ------------
    1. Multi-layer LSTM encodes the input sequence.
    2. Multi-head self-attention is applied over the full sequence of LSTM
       hidden states to capture long-range dependencies the recurrence
       may have compressed.
    3. A linear output head maps the attended context to the forecast.

    Input:  ``[batch, seq_len, n_features]``
    Output: ``[batch, forecast_horizon]``
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        hidden = config.lstm_hidden_size
        n_layers = config.lstm_n_layers
        dropout = config.lstm_dropout
        n_heads = config.lstm_attn_heads
        n_features = config.n_features
        forecast_horizon = config.forecast_horizon

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        # Multi-head self-attention over the LSTM hidden state sequence
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, forecast_horizon),
        )

        self._init_weights()

        logger.info(
            "AttentionLSTM: hidden={}, layers={}, attn_heads={}, params={:,}",
            hidden, n_layers, n_heads, self.count_parameters(),
        )

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "lstm" in name and "weight" in name:
                nn.init.orthogonal_(p)
            elif p.dim() > 1:
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
        # LSTM encoding: [B, T, n_features] -> [B, T, hidden]
        lstm_out, _ = self.lstm(x)

        # Self-attention over time-steps
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(lstm_out + attn_out)  # residual + norm

        # Use the last time-step of the attended output
        context = attn_out[:, -1, :]  # [B, hidden]

        out = self.output_head(context)  # [B, forecast_horizon]

        # NaN/Inf guard: replace with zeros to prevent downstream crashes
        if not torch.isfinite(out).all():
            out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))

        return out
