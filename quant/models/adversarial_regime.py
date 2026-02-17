"""Adversarial Regime Model -- GAN-inspired generator/discriminator for regime-aware trading."""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig
from quant.models.base import BaseModel
from quant.models.transformer import CausalTransformerDecoderLayer, SinusoidalPositionalEncoding


class AdversarialRegimeModel(BaseModel):
    """GAN-inspired model with generator, discriminator, and regime classifier.

    - **Generator**: Transformer encoder that predicts returns (standard ``forward()``).
    - **Discriminator**: MLP with sigmoid that classifies predictions as real/fake
      (auxiliary method ``discriminate()``).
    - **Regime Classifier**: MLP that classifies the current market regime
      (auxiliary method ``classify_regime()``).

    The standard ``forward()`` returns only the generator output, making the
    model fully compatible with the existing trainer pipeline.

    Input:  ``[batch, seq_len, n_features]``
    Output: ``[batch, forecast_horizon]``
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        d_model = config.adv_d_model
        n_heads = config.adv_n_heads
        n_layers = config.adv_n_layers
        d_ff = config.adv_d_ff
        dropout = config.adv_dropout
        n_regimes = config.adv_n_regimes

        self.n_regimes = n_regimes

        # --- Generator (Transformer Encoder) ---
        self.input_proj = nn.Linear(config.n_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=config.seq_len + 64, dropout=dropout,
        )
        self.encoder_layers = nn.ModuleList(
            [CausalTransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.generator_head = nn.Linear(d_model, config.forecast_horizon)

        # --- Discriminator (MLP with sigmoid) ---
        self.discriminator = nn.Sequential(
            nn.Linear(config.forecast_horizon, d_ff),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
            nn.Sigmoid(),
        )

        # --- Regime Classifier (MLP) ---
        self.regime_classifier = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_regimes),
        )

        self._init_weights()

        logger.info(
            "AdversarialRegimeModel: d_model={}, heads={}, layers={}, regimes={}, params={:,}",
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

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input through the Transformer and return last-step representation.

        Parameters
        ----------
        x : Tensor
            ``[batch, seq_len, n_features]``

        Returns
        -------
        Tensor
            ``[batch, d_model]``
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)

        causal_mask = self._make_causal_mask(h.size(1), h.device)
        for layer in self.encoder_layers:
            h = layer(h, causal_mask)
        h = self.encoder_norm(h)

        return h[:, -1, :]  # [B, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generator forward pass -- returns return predictions.

        Parameters
        ----------
        x : Tensor
            ``[batch, seq_len, n_features]``

        Returns
        -------
        Tensor
            ``[batch, forecast_horizon]``
        """
        context = self._encode(x)
        return self.generator_head(context)

    def discriminate(self, predictions: torch.Tensor) -> torch.Tensor:
        """Discriminator: classify predictions as real or fake.

        Parameters
        ----------
        predictions : Tensor
            ``[batch, forecast_horizon]``

        Returns
        -------
        Tensor
            ``[batch, 1]`` with values in ``[0, 1]``
        """
        return self.discriminator(predictions)

    def classify_regime(self, x: torch.Tensor) -> torch.Tensor:
        """Regime classifier: predict market regime from input.

        Parameters
        ----------
        x : Tensor
            ``[batch, seq_len, n_features]``

        Returns
        -------
        Tensor
            ``[batch, n_regimes]`` (logits)
        """
        context = self._encode(x)
        return self.regime_classifier(context)
