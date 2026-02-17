"""Tests for quant.core.state_manager -- SystemStateManager.

All tests use tmp_path (pytest built-in) for file system operations,
and torch.save/load under the hood.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from quant.core.state_manager import (
    MAX_BACKUPS,
    STATE_FILENAME,
    STATE_VERSION,
    SystemStateManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_state(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid state dict for testing."""
    state: dict[str, Any] = {
        "bar_counter": 42,
        "models": {"m1": {"state_dict": {"w": torch.tensor([1.0, 2.0])}}},
        "performance": {"equity_history": [100_000.0, 100_100.0]},
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """State saved and immediately loaded back is identical."""
        mgr = SystemStateManager(tmp_path)
        original = _sample_state(bar_counter=99)

        mgr.save(original)
        loaded = mgr.load()

        assert loaded is not None
        assert loaded["bar_counter"] == 99
        assert loaded["version"] == STATE_VERSION
        assert torch.equal(
            loaded["models"]["m1"]["state_dict"]["w"],
            original["models"]["m1"]["state_dict"]["w"],
        )

    def test_save_load_preserves_nested_dicts(self, tmp_path: Path) -> None:
        """Nested dicts survive the round-trip."""
        mgr = SystemStateManager(tmp_path)
        state = _sample_state(
            nested={"a": {"b": [1, 2, 3]}},
        )

        mgr.save(state)
        loaded = mgr.load()

        assert loaded is not None
        assert loaded["nested"]["a"]["b"] == [1, 2, 3]


class TestAtomicWrite:

    def test_atomic_write(self, tmp_path: Path) -> None:
        """The state file exists after save (atomic rename succeeded)."""
        mgr = SystemStateManager(tmp_path)
        mgr.save(_sample_state())

        state_path = tmp_path / STATE_FILENAME
        assert state_path.exists()

    def test_no_temp_files_left(self, tmp_path: Path) -> None:
        """After a successful save, no .tmp files remain in the directory."""
        mgr = SystemStateManager(tmp_path)
        mgr.save(_sample_state())

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Leftover temp files: {tmp_files}"


class TestBackupRotation:

    def test_backup_rotation(self, tmp_path: Path) -> None:
        """Saving multiple times creates and rotates .bak.N files."""
        mgr = SystemStateManager(tmp_path)
        state_path = tmp_path / STATE_FILENAME

        # Save MAX_BACKUPS + 2 times to ensure rotation kicks in
        for i in range(MAX_BACKUPS + 2):
            mgr.save(_sample_state(bar_counter=i))

        # The primary file should exist
        assert state_path.exists()

        # .bak.1 through .bak.MAX_BACKUPS should exist
        for k in range(1, MAX_BACKUPS + 1):
            bak = Path(f"{state_path}.bak.{k}")
            assert bak.exists(), f"Expected backup {bak} to exist"

        # .bak.(MAX_BACKUPS+1) should NOT exist (oldest is deleted)
        overflow = Path(f"{state_path}.bak.{MAX_BACKUPS + 1}")
        assert not overflow.exists()

    def test_backup_content_is_previous_state(self, tmp_path: Path) -> None:
        """The most recent backup (.bak.1) contains the previous save's data."""
        mgr = SystemStateManager(tmp_path)

        mgr.save(_sample_state(bar_counter=10))
        mgr.save(_sample_state(bar_counter=20))

        bak1 = Path(f"{tmp_path / STATE_FILENAME}.bak.1")
        loaded_bak = torch.load(str(bak1), map_location="cpu", weights_only=False)

        assert loaded_bak["bar_counter"] == 10


class TestCorruptStateFallback:

    def test_corrupt_state_fallback(self, tmp_path: Path) -> None:
        """When the primary file is corrupt, load() falls back to a backup."""
        mgr = SystemStateManager(tmp_path)
        state_path = tmp_path / STATE_FILENAME

        # Save valid state twice (creates .bak.1)
        mgr.save(_sample_state(bar_counter=100))
        mgr.save(_sample_state(bar_counter=200))

        # Corrupt the primary file
        state_path.write_text("this is not a valid torch file")

        loaded = mgr.load()

        assert loaded is not None
        # Should have loaded from .bak.1 which was the previous save (bar_counter=100)
        assert loaded["bar_counter"] == 100


class TestLoadNoStateReturnsNone:

    def test_load_no_state_returns_none(self, tmp_path: Path) -> None:
        """Loading from an empty directory returns None."""
        mgr = SystemStateManager(tmp_path)
        result = mgr.load()
        assert result is None

    def test_load_all_corrupt_returns_none(self, tmp_path: Path) -> None:
        """If primary and all backups are corrupt, load returns None."""
        mgr = SystemStateManager(tmp_path)
        state_path = tmp_path / STATE_FILENAME

        # Create corrupt primary and all backup files
        state_path.write_text("corrupt_primary")
        for i in range(1, MAX_BACKUPS + 1):
            Path(f"{state_path}.bak.{i}").write_text(f"corrupt_bak_{i}")

        result = mgr.load()
        assert result is None


class TestVersionInjected:

    def test_version_injected(self, tmp_path: Path) -> None:
        """The save method injects STATE_VERSION into the state dict."""
        mgr = SystemStateManager(tmp_path)
        state = _sample_state()
        assert "version" not in state  # not set beforehand

        mgr.save(state)
        loaded = mgr.load()

        assert loaded is not None
        assert loaded["version"] == STATE_VERSION

    def test_version_overwritten(self, tmp_path: Path) -> None:
        """Even if state already has a version, it gets overwritten."""
        mgr = SystemStateManager(tmp_path)
        state = _sample_state(version="0.0.0")

        mgr.save(state)
        loaded = mgr.load()

        assert loaded is not None
        assert loaded["version"] == STATE_VERSION
