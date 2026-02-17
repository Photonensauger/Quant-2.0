"""Tests for quant.models.transformer -- Decoder-only Transformer."""

from __future__ import annotations

import torch
import pytest

from quant.config.settings import ModelConfig
from quant.models.transformer import DecoderTransformer


class TestDecoderTransformer:
    """Tests for DecoderTransformer."""

    def test_forward_shape(
        self,
        small_model_config: ModelConfig,
        sample_batch: torch.Tensor,
        cpu_device: torch.device,
    ) -> None:
        """Output shape must be [batch, forecast_horizon]."""
        model = DecoderTransformer(small_model_config).to(cpu_device)
        model.eval()

        with torch.no_grad():
            out = model(sample_batch)

        assert out.shape == (
            sample_batch.shape[0],
            small_model_config.forecast_horizon,
        ), f"Expected shape ({sample_batch.shape[0]}, {small_model_config.forecast_horizon}), got {out.shape}"

    def test_causal_mask(self, cpu_device: torch.device) -> None:
        """Causal mask should be upper-triangular with -inf above diagonal."""
        seq_len = 8
        mask = DecoderTransformer._make_causal_mask(seq_len, cpu_device)

        assert mask.shape == (seq_len, seq_len)

        # Diagonal and below should be 0
        lower = torch.tril(mask, diagonal=0)
        assert torch.all(lower == 0.0), "Lower triangle should be zero"

        # Above diagonal should be -inf
        upper_vals = mask[torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)]
        assert torch.all(upper_vals == float("-inf")), (
            "Upper triangle should be -inf"
        )

    def test_different_batch_sizes(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """Model should handle batch sizes of 1, 2, 8, and 16."""
        model = DecoderTransformer(small_model_config).to(cpu_device)
        model.eval()
        torch.manual_seed(0)

        for batch_size in (1, 2, 8, 16):
            x = torch.randn(
                batch_size,
                small_model_config.seq_len,
                small_model_config.n_features,
                device=cpu_device,
            )
            with torch.no_grad():
                out = model(x)
            assert out.shape == (
                batch_size,
                small_model_config.forecast_horizon,
            ), f"Wrong shape for batch_size={batch_size}: {out.shape}"

    def test_gradient_flows(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """Gradients must flow from loss back to all model parameters."""
        model = DecoderTransformer(small_model_config).to(cpu_device)
        model.train()
        torch.manual_seed(1)

        x = torch.randn(
            2,
            small_model_config.seq_len,
            small_model_config.n_features,
            device=cpu_device,
        )
        target = torch.randn(
            2,
            small_model_config.forecast_horizon,
            device=cpu_device,
        )

        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"No gradient for parameter: {name}"
                )
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for parameter: {name}"
                )
