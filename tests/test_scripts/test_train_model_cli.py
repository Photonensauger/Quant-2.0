"""Tests for scripts/train_model.py CLI argument handling.

Guards against regression: the --device flag must be accepted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.train_model import ALL_MODEL_NAMES, parse_args


class TestDeviceFlag:
    def test_device_default_is_none(self):
        args = parse_args([
            "--model", "transformer",
            "--assets", "AAPL",
        ])
        assert args.device is None

    def test_device_cpu(self):
        args = parse_args([
            "--model", "transformer",
            "--assets", "AAPL",
            "--device", "cpu",
        ])
        assert args.device == "cpu"

    def test_device_mps(self):
        args = parse_args([
            "--model", "transformer",
            "--assets", "AAPL",
            "--device", "mps",
        ])
        assert args.device == "mps"

    def test_device_cuda(self):
        args = parse_args([
            "--model", "transformer",
            "--assets", "AAPL",
            "--device", "cuda",
        ])
        assert args.device == "cuda"

    def test_device_auto(self):
        args = parse_args([
            "--model", "transformer",
            "--assets", "AAPL",
            "--device", "auto",
        ])
        assert args.device == "auto"

    def test_device_invalid_rejected(self):
        with pytest.raises(SystemExit):
            parse_args([
                "--model", "transformer",
                "--assets", "AAPL",
                "--device", "tpu",
            ])


class TestAllModelNamesComplete:
    """ALL_MODEL_NAMES must include Zug 37 models."""

    def test_has_zug37(self):
        expected = {"causal", "schrodinger", "topological",
                    "hamiltonian", "diffusion", "adversarial"}
        assert expected.issubset(set(ALL_MODEL_NAMES))

    def test_has_original(self):
        expected = {"transformer", "itransformer", "lstm", "momentum"}
        assert expected.issubset(set(ALL_MODEL_NAMES))

    def test_all_accepted(self):
        """--model all must be a valid choice."""
        args = parse_args(["--model", "all", "--assets", "AAPL"])
        assert args.model == "all"
