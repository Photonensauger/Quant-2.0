"""Entropic Portfolio Diffusion -- iterative denoising with entropy regularisation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from loguru import logger

from quant.config.settings import ModelConfig
from quant.models.base import BaseModel
from quant.models.transformer import CausalTransformerDecoderLayer, SinusoidalPositionalEncoding


class EntropicPortfolioDiffusion(BaseModel):
    """Diffusion-inspired model with iterative denoising refinement.

    A context encoder (Transformer) produces an initial prediction which is
    then iteratively refined through ``n_diffusion_steps`` denoising steps.
    Each step concatenates the current prediction with the context and a
    sinusoidal step embedding, then applies a denoiser MLP with a residual
    update.  The process is fully deterministic (no sampling).

    An entropy regularisation term is computed as ``aux_loss`` to encourage
    diverse predictions.

    Input:  ``[batch, seq_len, n_features]``
    Output: ``[batch, forecast_horizon]``
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        d_model = config.epd_d_model
        n_heads = config.epd_n_heads
        n_layers = config.epd_n_layers
        d_ff = config.epd_d_ff
        dropout = config.epd_dropout
        n_diffusion_steps = config.epd_n_diffusion_steps
        self.entropy_weight = config.epd_entropy_weight

        self.n_diffusion_steps = n_diffusion_steps
        self.d_model = d_model

        # Context encoder: Transformer
        self.input_proj = nn.Linear(config.n_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=config.seq_len + 64, dropout=dropout,
        )
        self.encoder_layers = nn.ModuleList(
            [CausalTransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # Initial prediction head
        self.init_pred = nn.Linear(d_model, config.forecast_horizon)

        # Step embedding dimension
        step_embed_dim = 32

        # Denoiser MLP: concat(pred, context_summary, step_embed) -> residual
        denoiser_input_dim = config.forecast_horizon + d_model + step_embed_dim
        self.denoiser = nn.Sequential(
            nn.Linear(denoiser_input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, config.forecast_horizon),
        )

        # Sinusoidal step embedding
        self.step_embed_dim = step_embed_dim
        self._register_step_embeddings(n_diffusion_steps, step_embed_dim)

        # Auxiliary loss placeholder
        self.aux_loss: torch.Tensor | None = None

        self._init_weights()

        logger.info(
            "EntropicPortfolioDiffusion: d_model={}, heads={}, layers={}, "
            "diffusion_steps={}, params={:,}",
            d_model, n_heads, n_layers, n_diffusion_steps, self.count_parameters(),
        )

    def _register_step_embeddings(self, n_steps: int, dim: int) -> None:
        """Pre-compute sinusoidal embeddings for each diffusion step."""
        pe = torch.zeros(n_steps, dim)
        position = torch.arange(0, n_steps, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("step_embeddings", pe)  # [n_steps, dim]

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def _compute_entropy_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute entropy regularisation on the prediction distribution.

        We treat the softmax of predictions as a probability distribution and
        compute its entropy.  Higher entropy is encouraged (negative loss).
        """
        probs = torch.softmax(pred, dim=-1)
        log_probs = torch.log_softmax(pred, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return -entropy * self.entropy_weight  # negative because we maximise entropy

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
        B = x.size(0)

        # Encode context
        h = self.input_proj(x)
        h = self.pos_enc(h)

        causal_mask = self._make_causal_mask(h.size(1), h.device)
        for layer in self.encoder_layers:
            h = layer(h, causal_mask)
        h = self.encoder_norm(h)

        # Context summary: last time-step
        context = h[:, -1, :]  # [B, d_model]

        # Initial prediction
        pred = self.init_pred(context)  # [B, forecast_horizon]

        # Iterative denoising refinement
        for step in range(self.n_diffusion_steps):
            step_emb = self.step_embeddings[step].unsqueeze(0).expand(B, -1)  # [B, step_embed_dim]
            denoiser_input = torch.cat([pred, context, step_emb], dim=-1)
            residual = self.denoiser(denoiser_input)  # [B, forecast_horizon]
            pred = pred + residual  # residual update

        # Compute entropy auxiliary loss
        self.aux_loss = self._compute_entropy_loss(pred)

        return pred
