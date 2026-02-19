"""Regression test: backtest and training registries must stay in sync.

If a model is added to one script but not the other, backtests will silently
skip models or training will miss new architectures.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_backtest import MODEL_REGISTRY as BT_REGISTRY
from scripts.train_model import MODEL_REGISTRY as TRAIN_REGISTRY


class TestRegistriesInSync:
    def test_same_keys(self):
        """Both registries must contain the exact same model keys."""
        assert set(BT_REGISTRY.keys()) == set(TRAIN_REGISTRY.keys())

    def test_same_classes(self):
        """Both registries must map to the same model classes."""
        for key in BT_REGISTRY:
            assert BT_REGISTRY[key] is TRAIN_REGISTRY[key], (
                f"Registry mismatch for '{key}': "
                f"backtest={BT_REGISTRY[key]}, train={TRAIN_REGISTRY[key]}"
            )

    def test_minimum_model_count(self):
        """Registries must contain at least 10 models (4 original + 6 Zug 37)."""
        assert len(BT_REGISTRY) >= 10
        assert len(TRAIN_REGISTRY) >= 10
