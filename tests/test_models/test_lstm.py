"""Tests for quant.models.lstm -- AttentionLSTM."""

from __future__ import annotations

import torch
import pytest

from quant.config.settings import ModelConfig
from quant.models.lstm import AttentionLSTM


class TestAttentionLSTM:
    """Tests for AttentionLSTM."""

    def test_forward_shape(
        self,
        small_model_config: ModelConfig,
        sample_batch: torch.Tensor,
        cpu_device: torch.device,
    ) -> None:
        """Output shape must be [batch, forecast_horizon]."""
        model = AttentionLSTM(small_model_config).to(cpu_device)
        model.eval()

        with torch.no_grad():
            out = model(sample_batch)

        assert out.shape == (
            sample_batch.shape[0],
            small_model_config.forecast_horizon,
        ), f"Expected shape ({sample_batch.shape[0]}, {small_model_config.forecast_horizon}), got {out.shape}"

    def test_attention_mechanism(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """The model should contain an attention layer that processes LSTM output."""
        model = AttentionLSTM(small_model_config).to(cpu_device)

        # Verify the attention module exists and has the right embedding dim
        assert hasattr(model, "attn"), "Model must have 'attn' attribute"
        assert model.attn.embed_dim == small_model_config.lstm_hidden_size

        # Verify attention norm exists
        assert hasattr(model, "attn_norm"), "Model must have 'attn_norm'"

        # Forward pass should work end-to-end
        torch.manual_seed(42)
        x = torch.randn(
            2,
            small_model_config.seq_len,
            small_model_config.n_features,
            device=cpu_device,
        )
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, small_model_config.forecast_horizon)
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_different_batch_sizes(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """Model should handle batch sizes of 1, 2, 8, and 16."""
        model = AttentionLSTM(small_model_config).to(cpu_device)
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
        model = AttentionLSTM(small_model_config).to(cpu_device)
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
