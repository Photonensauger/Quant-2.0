"""Tests for quant.models.adversarial_regime -- Adversarial Regime Model."""

from __future__ import annotations

import torch

from quant.config.settings import ModelConfig
from quant.models.adversarial_regime import AdversarialRegimeModel


class TestAdversarialRegimeModel:
    """Tests for AdversarialRegimeModel."""

    def test_forward_shape(
        self,
        small_model_config: ModelConfig,
        sample_batch: torch.Tensor,
        cpu_device: torch.device,
    ) -> None:
        """Output shape must be [batch, forecast_horizon]."""
        model = AdversarialRegimeModel(small_model_config).to(cpu_device)
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
        model = AdversarialRegimeModel(small_model_config).to(cpu_device)
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
        model = AdversarialRegimeModel(small_model_config).to(cpu_device)
        model.train()
        torch.manual_seed(1)

        x = torch.randn(2, small_model_config.seq_len, small_model_config.n_features, device=cpu_device)
        target = torch.randn(2, small_model_config.forecast_horizon, device=cpu_device)

        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)

        # Also backprop through discriminator and regime classifier
        disc_loss = model.discriminate(out).mean()
        regime_logits = model.classify_regime(x)
        regime_loss = regime_logits.mean()

        total_loss = loss + disc_loss + regime_loss
        total_loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for parameter: {name}"

    def test_discriminator_output_range(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """Discriminator output should be in [0, 1] due to sigmoid."""
        model = AdversarialRegimeModel(small_model_config).to(cpu_device)
        model.eval()
        torch.manual_seed(5)

        predictions = torch.randn(4, small_model_config.forecast_horizon, device=cpu_device)

        with torch.no_grad():
            disc_out = model.discriminate(predictions)

        assert disc_out.shape == (4, 1)
        assert (disc_out >= 0.0).all()
        assert (disc_out <= 1.0).all()
