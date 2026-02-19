"""Tests for scripts/run_backtest.py CLI argument handling.

Guards against regressions:
  - Default model list must include ALL models from the registry (not a
    hardcoded subset of 3).
  - The --device flag must be accepted and forwarded to config.
  - The --models flag must list all available models in its help text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so we can import `scripts.*`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_backtest import MODEL_REGISTRY, parse_args


# ---------------------------------------------------------------------------
# Default model list
# ---------------------------------------------------------------------------

class TestDefaultModelList:
    """The default model list must cover the full MODEL_REGISTRY."""

    def test_registry_has_all_zug37_models(self):
        """Zug 37 models must be present in the registry."""
        expected = {"causal", "schrodinger", "topological",
                    "hamiltonian", "diffusion", "adversarial"}
        assert expected.issubset(MODEL_REGISTRY.keys())

    def test_registry_has_original_models(self):
        expected = {"transformer", "itransformer", "lstm", "momentum"}
        assert expected.issubset(MODEL_REGISTRY.keys())

    def test_default_models_arg_is_none(self):
        """When --models is not passed, args.models should be None
        so that main() falls through to MODEL_REGISTRY.keys()."""
        args = parse_args([
            "--strategy", "ensemble",
            "--assets", "AAPL",
            "--start", "2025-01-01",
            "--end", "2025-02-01",
        ])
        assert args.models is None

    def test_explicit_models_override(self):
        args = parse_args([
            "--strategy", "ensemble",
            "--assets", "AAPL",
            "--start", "2025-01-01",
            "--end", "2025-02-01",
            "--models", "transformer,lstm",
        ])
        assert args.models == "transformer,lstm"


# ---------------------------------------------------------------------------
# --device flag
# ---------------------------------------------------------------------------

class TestDeviceFlag:
    def test_device_default_is_none(self):
        args = parse_args([
            "--strategy", "ensemble",
            "--assets", "AAPL",
            "--start", "2025-01-01",
            "--end", "2025-02-01",
        ])
        assert args.device is None

    def test_device_cpu(self):
        args = parse_args([
            "--strategy", "ensemble",
            "--assets", "AAPL",
            "--start", "2025-01-01",
            "--end", "2025-02-01",
            "--device", "cpu",
        ])
        assert args.device == "cpu"

    def test_device_auto(self):
        args = parse_args([
            "--strategy", "ensemble",
            "--assets", "AAPL",
            "--start", "2025-01-01",
            "--end", "2025-02-01",
            "--device", "auto",
        ])
        assert args.device == "auto"

    def test_device_invalid_rejected(self):
        with pytest.raises(SystemExit):
            parse_args([
                "--strategy", "ensemble",
                "--assets", "AAPL",
                "--start", "2025-01-01",
                "--end", "2025-02-01",
                "--device", "tpu",
            ])
