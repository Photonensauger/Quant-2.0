"""Structured logging configuration using *loguru*.

Call :func:`setup_logging` once at application startup to configure console
and file logging with consistent formatting, automatic rotation, and retention
policies.

Example
-------
>>> from quant.utils.logging import setup_logging
>>> setup_logging("DEBUG")
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from quant.config.settings import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_DIR: Path = PROJECT_ROOT / "logs"

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{module}:{function}:{line} - "
    "{message}"
)

# Rotation / retention defaults
_ROTATION = "50 MB"
_RETENTION = "30 days"
_COMPRESSION = "gz"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> "logger":
    """Configure application-wide logging with loguru.

    * Removes any previously installed handlers (idempotent).
    * Adds a **console** (stderr) handler with colourised output.
    * Adds a **file** handler under ``<project_root>/logs/`` with automatic
      rotation, retention, and compression.

    Parameters
    ----------
    level : str
        Minimum log level.  One of ``"TRACE"``, ``"DEBUG"``, ``"INFO"``,
        ``"SUCCESS"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``.

    Returns
    -------
    loguru.Logger
        The configured global ``logger`` instance (same singleton returned
        by ``from loguru import logger``).
    """
    level = level.upper()

    # Remove all existing handlers so repeated calls are safe.
    logger.remove()

    # ---- Console handler ------------------------------------------------
    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ---- File handler ---------------------------------------------------
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_DIR / "quant_{time:YYYY-MM-DD}.log"

    logger.add(
        str(log_file),
        format=_FILE_FORMAT,
        level=level,
        rotation=_ROTATION,
        retention=_RETENTION,
        compression=_COMPRESSION,
        backtrace=True,
        diagnose=True,
        enqueue=True,  # thread-safe writes
    )

    logger.info("Logging configured | level={} | log_dir={}", level, _LOG_DIR)
    return logger
