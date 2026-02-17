"""Tests for quant.models.momentum_transformer -- MomentumTransformer."""

from __future__ import annotations

import torch
import pytest

from quant.config.settings import ModelConfig
from quant.models.momentum_transformer import MomentumTransformer


class TestMomentumTransformer:
    """Tests for MomentumTransformer."""

    def test_forward_shape(
        self,
        small_model_config: ModelConfig,
        sample_batch: torch.Tensor,
        cpu_device: torch.device,
    ) -> None:
        """Output shape must be [batch, 1] (position signal)."""
        model = MomentumTransformer(small_model_config).to(cpu_device)
        model.eval()

        with torch.no_grad():
            out = model(sample_batch)

        assert out.shape == (sample_batch.shape[0], 1), (
            f"Expected shape ({sample_batch.shape[0]}, 1), got {out.shape}"
        )

    def test_output_range_tanh(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """All output values must be in [-1, +1] due to the tanh activation."""
        model = MomentumTransformer(small_model_config).to(cpu_device)
        model.eval()
        torch.manual_seed(42)

        # Test with several random batches to increase coverage
        for seed in range(5):
            torch.manual_seed(seed)
            x = torch.randn(
                8,
                small_model_config.seq_len,
                small_model_config.n_features,
                device=cpu_device,
            )
            with torch.no_grad():
                out = model(x)

            assert out.min().item() >= -1.0, (
                f"Output below -1: {out.min().item()}"
            )
            assert out.max().item() <= 1.0, (
                f"Output above +1: {out.max().item()}"
            )

        # Also test with extreme inputs
        x_large = torch.randn(
            4,
            small_model_config.seq_len,
            small_model_config.n_features,
            device=cpu_device,
        ) * 100.0

        with torch.no_grad():
            out_large = model(x_large)

        assert out_large.min().item() >= -1.0, (
            f"Output below -1 with large input: {out_large.min().item()}"
        )
        assert out_large.max().item() <= 1.0, (
            f"Output above +1 with large input: {out_large.max().item()}"
        )

    def test_gradient_flows(
        self,
        small_model_config: ModelConfig,
        cpu_device: torch.device,
    ) -> None:
        """Gradients must flow from loss back to all model parameters."""
        model = MomentumTransformer(small_model_config).to(cpu_device)
        model.train()
        torch.manual_seed(1)

        x = torch.randn(
            2,
            small_model_config.seq_len,
            small_model_config.n_features,
            device=cpu_device,
        )
        # Target positions in [-1, 1]
        target = torch.tanh(torch.randn(2, 1, device=cpu_device))

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
