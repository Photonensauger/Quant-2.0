"""Tests for quant.models.topological_attention -- Topological Attention Network."""

from __future__ import annotations

import torch

from quant.config.settings import ModelConfig
from quant.models.topological_attention import TopologicalAttentionNetwork


class TestTopologicalAttentionNetwork:
    """Tests for TopologicalAttentionNetwork."""

    def test_forward_shape(
        self,
        small_model_config: ModelConfig,
        sample_batch: torch.Tensor,
        cpu_device: torch.device,
    ) -> None:
        """Output shape must be [batch, forecast_horizon]."""
        model = TopologicalAttentionNetwork(small_model_config).to(cpu_device)
        model.eval()

        with torch.no_grad():
            out = model(sample_batch)

        assert out.shape == (
            sample_batch.shape[0],
            small_model_config.forecast_horizon,
        )

    def test_different_batch_sizes(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """Model should handle batch sizes of 1, 2, 8, and 16."""
        model = TopologicalAttentionNetwork(small_model_config).to(cpu_device)
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
            assert out.shape == (batch_size, small_model_config.forecast_horizon)

    def test_gradient_flows(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """Gradients must flow from loss back to all model parameters."""
        model = TopologicalAttentionNetwork(small_model_config).to(cpu_device)
        model.train()
        torch.manual_seed(1)

        x = torch.randn(2, small_model_config.seq_len, small_model_config.n_features, device=cpu_device)
        target = torch.randn(2, small_model_config.forecast_horizon, device=cpu_device)

        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for parameter: {name}"

    def test_scale_params_are_learnable(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """The filtration scale parameters should be learnable and receive gradients."""
        model = TopologicalAttentionNetwork(small_model_config).to(cpu_device)
        model.train()
        torch.manual_seed(2)

        x = torch.randn(2, small_model_config.seq_len, small_model_config.n_features, device=cpu_device)
        target = torch.randn(2, small_model_config.forecast_horizon, device=cpu_device)

        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()

        assert model.scale_params.requires_grad
        assert model.scale_params.grad is not None
        assert model.scale_params.shape == (small_model_config.top_n_scales,)
