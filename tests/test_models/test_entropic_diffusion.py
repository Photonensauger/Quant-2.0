"""Tests for quant.models.entropic_diffusion -- Entropic Portfolio Diffusion."""

from __future__ import annotations

import torch

from quant.config.settings import ModelConfig
from quant.models.entropic_diffusion import EntropicPortfolioDiffusion


class TestEntropicPortfolioDiffusion:
    """Tests for EntropicPortfolioDiffusion."""

    def test_forward_shape(
        self,
        small_model_config: ModelConfig,
        sample_batch: torch.Tensor,
        cpu_device: torch.device,
    ) -> None:
        """Output shape must be [batch, forecast_horizon]."""
        model = EntropicPortfolioDiffusion(small_model_config).to(cpu_device)
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
        model = EntropicPortfolioDiffusion(small_model_config).to(cpu_device)
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
        model = EntropicPortfolioDiffusion(small_model_config).to(cpu_device)
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

    def test_aux_loss_computed(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """After forward pass, aux_loss should be a scalar tensor."""
        model = EntropicPortfolioDiffusion(small_model_config).to(cpu_device)
        model.eval()
        torch.manual_seed(4)

        x = torch.randn(2, small_model_config.seq_len, small_model_config.n_features, device=cpu_device)

        with torch.no_grad():
            model(x)

        assert model.aux_loss is not None
        assert model.aux_loss.dim() == 0  # scalar
        assert torch.isfinite(model.aux_loss)
