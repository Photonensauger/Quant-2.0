"""Cyclical time-of-day / calendar feature engineering.

Every temporal dimension is encoded as a *(sin, cos)* pair so that the
model can learn smooth, wrap-around periodicity (e.g. hour 23 is close to
hour 0).  No NaN values are produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def _sin_cos(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (sin, cos) encoding of *values* with a given *period*."""
    angle = 2.0 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cyclical calendar features from a DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``DatetimeIndex`` (or a column parseable to datetime that
        can be set as the index).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ~10 additional columns (sin/cos pairs for each
        temporal cycle).  No NaN values are introduced.

    Raises
    ------
    TypeError
        If the index is not a ``DatetimeIndex``.
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Expected DatetimeIndex, got {type(df.index).__name__}.  "
            "Set a datetime index before calling compute_time_features."
        )

    idx = df.index
    features = pd.DataFrame(index=idx)

    # ── Hour of day (period = 24) ─────────────────────────────────────────
    hour = idx.hour + idx.minute / 60.0
    sin_h, cos_h = _sin_cos(hour.values, 24.0)
    features["hour_sin"] = sin_h
    features["hour_cos"] = cos_h

    # ── Day of week (period = 7, Monday=0) ────────────────────────────────
    dow = idx.dayofweek.values.astype(np.float64)
    sin_d, cos_d = _sin_cos(dow, 7.0)
    features["dow_sin"] = sin_d
    features["dow_cos"] = cos_d

    # ── Day of month (period = ~31) ───────────────────────────────────────
    dom = idx.day.values.astype(np.float64)
    sin_dm, cos_dm = _sin_cos(dom, 31.0)
    features["dom_sin"] = sin_dm
    features["dom_cos"] = cos_dm

    # ── Month of year (period = 12) ──────────────────────────────────────
    month = idx.month.values.astype(np.float64)
    sin_m, cos_m = _sin_cos(month, 12.0)
    features["month_sin"] = sin_m
    features["month_cos"] = cos_m

    # ── Quarter (period = 4) ─────────────────────────────────────────────
    quarter = idx.quarter.values.astype(np.float64)
    sin_q, cos_q = _sin_cos(quarter, 4.0)
    features["quarter_sin"] = sin_q
    features["quarter_cos"] = cos_q

    n_cols = len(features.columns)
    logger.debug("Computed {} time features (all NaN-free)", n_cols)

    return pd.concat([df, features], axis=1)
