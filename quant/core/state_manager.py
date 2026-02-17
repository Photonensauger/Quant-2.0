"""Atomic state persistence with backup rotation.

The :class:`SystemStateManager` provides crash-safe checkpointing by writing
to a temporary file and atomically renaming it into place.  The last three
backups are kept so that a corrupt current file can always be recovered from
a recent backup.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from loguru import logger

STATE_VERSION = "2.0.0"
STATE_FILENAME = "system_state.pt"
MAX_BACKUPS = 3


class SystemStateManager:
    """Atomic state save / load with backup rotation.

    Parameters
    ----------
    state_dir : Path | str
        Directory where state files are written.  Created automatically
        if it does not exist.
    """

    def __init__(self, state_dir: Path | str) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.state_dir / STATE_FILENAME
        logger.info(
            "SystemStateManager initialised | dir={} | version={}",
            self.state_dir,
            STATE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, state: dict[str, Any]) -> None:
        """Atomically write *state* to disk and rotate backups.

        The write is performed into a temporary file in the same directory,
        then ``os.rename`` (atomic on POSIX) replaces the live file.
        Before the rename, existing backups are rotated so the last
        ``MAX_BACKUPS`` versions are preserved.

        Parameters
        ----------
        state : dict
            Full system state dictionary.  A ``"version"`` key is
            injected automatically.
        """
        state["version"] = STATE_VERSION

        # 1. Rotate existing backups  .bak.3 -> .bak.4 (deleted), .bak.2 -> .bak.3, ...
        self._rotate_backups()

        # 2. Current file becomes .bak.1
        if self._state_path.exists():
            bak1 = Path(f"{self._state_path}.bak.1")
            try:
                os.replace(str(self._state_path), str(bak1))
            except OSError as exc:
                logger.warning("Failed to rotate current state to backup: {}", exc)

        # 3. Write to a temp file in the same directory, then atomic rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.state_dir), suffix=".tmp", prefix="state_"
        )
        try:
            os.close(fd)
            torch.save(state, tmp_path)
            os.replace(tmp_path, str(self._state_path))
            logger.debug("State saved atomically to {}", self._state_path)
        except Exception:
            # Clean up the temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self) -> dict[str, Any] | None:
        """Load the latest state file, falling back to backups if corrupt.

        Returns
        -------
        dict | None
            The state dictionary, or ``None`` if no valid state file
            (or backup) could be found.
        """
        # Try the primary state file first
        state = self._try_load(self._state_path)
        if state is not None:
            logger.info(
                "State loaded from {} (version={})",
                self._state_path,
                state.get("version", "unknown"),
            )
            return state

        # Fall back to backups in order  .bak.1, .bak.2, .bak.3
        for i in range(1, MAX_BACKUPS + 1):
            bak_path = Path(f"{self._state_path}.bak.{i}")
            state = self._try_load(bak_path)
            if state is not None:
                logger.warning(
                    "Primary state corrupt/missing -- recovered from backup {}",
                    bak_path,
                )
                return state

        logger.info("No valid state file found in {}", self.state_dir)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rotate_backups(self) -> None:
        """Rotate backup files: delete oldest, shift others up by one."""
        # Delete the oldest backup beyond MAX_BACKUPS
        oldest = Path(f"{self._state_path}.bak.{MAX_BACKUPS}")
        if oldest.exists():
            try:
                oldest.unlink()
            except OSError as exc:
                logger.warning("Failed to delete oldest backup {}: {}", oldest, exc)

        # Shift  .bak.(n-1) -> .bak.n  down to  .bak.1 -> .bak.2
        for i in range(MAX_BACKUPS - 1, 0, -1):
            src = Path(f"{self._state_path}.bak.{i}")
            dst = Path(f"{self._state_path}.bak.{i + 1}")
            if src.exists():
                try:
                    os.replace(str(src), str(dst))
                except OSError as exc:
                    logger.warning("Failed to rotate backup {} -> {}: {}", src, dst, exc)

    @staticmethod
    def _try_load(path: Path) -> dict[str, Any] | None:
        """Attempt to load a state file; return None on any failure."""
        if not path.exists():
            return None
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=False)
            if not isinstance(state, dict):
                logger.warning("State file {} is not a dict -- ignoring", path)
                return None
            return state
        except Exception as exc:
            logger.warning("Failed to load state from {}: {}", path, exc)
            return None
