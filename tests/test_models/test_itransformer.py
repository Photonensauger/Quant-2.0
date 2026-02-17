"""Tests for quant.models.itransformer -- Inverted Transformer."""

from __future__ import annotations

import torch
import pytest

from quant.config.settings import ModelConfig
from quant.models.itransformer import ITransformer


class TestITransformer:
    """Tests for ITransformer."""

    def test_forward_shape(
        self,
        small_model_config: ModelConfig,
        sample_batch: torch.Tensor,
        cpu_device: torch.device,
    ) -> None:
        """Output shape must be [batch, forecast_horizon]."""
        model = ITransformer(small_model_config).to(cpu_device)
        model.eval()

        with torch.no_grad():
            out = model(sample_batch)

        assert out.shape == (
            sample_batch.shape[0],
            small_model_config.forecast_horizon,
        ), f"Expected shape ({sample_batch.shape[0]}, {small_model_config.forecast_horizon}), got {out.shape}"

    def test_feature_as_token(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """In the iTransformer, each feature is a token. Changing n_features
        should change the number of tokens (and require a new model)."""
        torch.manual_seed(0)

        # Two configs with different n_features
        cfg_a = ModelConfig(
            seq_len=small_model_config.seq_len,
            forecast_horizon=small_model_config.forecast_horizon,
            n_features=6,
            itf_d_model=32,
            itf_n_heads=2,
            itf_n_layers=1,
            itf_d_ff=64,
            itf_dropout=0.0,
        )
        cfg_b = ModelConfig(
            seq_len=small_model_config.seq_len,
            forecast_horizon=small_model_config.forecast_horizon,
            n_features=12,
            itf_d_model=32,
            itf_n_heads=2,
            itf_n_layers=1,
            itf_d_ff=64,
            itf_dropout=0.0,
        )

        model_a = ITransformer(cfg_a).to(cpu_device)
        model_b = ITransformer(cfg_b).to(cpu_device)

        # The feature_embed parameter size encodes n_features
        assert model_a.feature_embed.shape[1] == 6
        assert model_b.feature_embed.shape[1] == 12

        # Both should produce correct output shapes
        x_a = torch.randn(2, cfg_a.seq_len, cfg_a.n_features, device=cpu_device)
        x_b = torch.randn(2, cfg_b.seq_len, cfg_b.n_features, device=cpu_device)

        with torch.no_grad():
            out_a = model_a(x_a)
            out_b = model_b(x_b)

        assert out_a.shape == (2, cfg_a.forecast_horizon)
        assert out_b.shape == (2, cfg_b.forecast_horizon)

    def test_different_batch_sizes(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """Model should handle batch sizes of 1, 2, 8, and 16."""
        model = ITransformer(small_model_config).to(cpu_device)
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
        model = ITransformer(small_model_config).to(cpu_device)
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
