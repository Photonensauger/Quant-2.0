"""Information geometry features: Fisher information, Renyi entropy
spectrum, and KL divergence rate.

These features capture distributional properties of rolling return windows,
providing signals about regime complexity, distribution sharpness, and
the speed of distributional change.

All computation uses NumPy + SciPy only (no PyTorch dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats

from quant.config.settings import FeatureConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _histogram_density(
    values: np.ndarray,
    n_bins: int,
    eps: float = 1e-10,
    bin_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a normalised histogram density with an epsilon floor.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observations.
    n_bins : int
        Number of histogram bins.
    eps : float
        Minimum probability per bin to avoid log(0).
    bin_edges : np.ndarray | None
        Pre-computed bin edges.  If ``None``, edges are derived from *values*.

    Returns
    -------
    p : np.ndarray
        Probability vector (sums to 1, each entry >= eps).
    edges : np.ndarray
        Bin edges used.
    """
    if bin_edges is None:
        counts, edges = np.histogram(values, bins=n_bins)
    else:
        counts, edges = np.histogram(values, bins=bin_edges)

    p = counts.astype(np.float64)
    total = p.sum()
    if total > 0:
        p /= total
    else:
        p[:] = 1.0 / len(p)

    # Epsilon floor and renormalise
    p = np.maximum(p, eps)
    p /= p.sum()
    return p, edges


def _fisher_information(
    values: np.ndarray, n_bins: int, delta: float, eps: float = 1e-10,
) -> float:
    """Estimate Fisher information of a 1-D sample via histogram density.

    Uses central finite differences on the log-density with respect to a
    location shift *delta* of the data.

    Parameters
    ----------
    values : np.ndarray
        1-D observations from the rolling window.
    n_bins : int
        Number of histogram bins.
    delta : float
        Perturbation magnitude for numerical differentiation.
    eps : float
        Probability floor.

    Returns
    -------
    float
        Fisher information estimate (>= 0).
    """
    p_centre, edges = _histogram_density(values, n_bins, eps)
    p_plus, _ = _histogram_density(values + delta, n_bins, eps, bin_edges=edges)
    p_minus, _ = _histogram_density(values - delta, n_bins, eps, bin_edges=edges)

    log_p_plus = np.log(p_plus)
    log_p_minus = np.log(p_minus)
    score = (log_p_plus - log_p_minus) / (2.0 * delta)

    fisher = float(np.sum(p_centre * score ** 2))
    return max(fisher, 0.0)


def _renyi_entropy(p: np.ndarray, alpha: float) -> float:
    """Compute Renyi entropy of order *alpha*.

    For alpha == 1.0, returns Shannon entropy via scipy.

    Parameters
    ----------
    p : np.ndarray
        Probability vector (must sum to 1, all entries > 0).
    alpha : float
        Order parameter.

    Returns
    -------
    float
        Renyi entropy H_alpha >= 0.
    """
    if abs(alpha - 1.0) < 1e-12:
        return float(sp_stats.entropy(p))

    sum_p_alpha = np.sum(p ** alpha)
    if sum_p_alpha <= 0:
        return 0.0
    h = np.log(sum_p_alpha) / (1.0 - alpha)
    return max(float(h), 0.0)


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute KL(P || Q) with epsilon smoothing.

    Parameters
    ----------
    p, q : np.ndarray
        Probability vectors of equal length.
    eps : float
        Floor applied to *q* to avoid division by zero.

    Returns
    -------
    float
        KL divergence >= 0.
    """
    q_safe = np.maximum(q, eps)
    q_safe /= q_safe.sum()
    kl = float(np.sum(p * np.log(p / q_safe)))
    return max(kl, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_information_geometry_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute information geometry features from a DataFrame with a
    ``close`` column.

    Produces 7 new columns, all prefixed ``ig_``:

    1. ``ig_fisher_info``     -- Fisher information of the return distribution
    2. ``ig_renyi_h_0.5``     -- Renyi entropy (alpha=0.5)
    3. ``ig_renyi_h_2.0``     -- Renyi entropy (alpha=2.0)
    4. ``ig_renyi_h_10.0``    -- Renyi entropy (alpha=10.0)
    5. ``ig_renyi_slope``     -- Slope of H_alpha vs alpha (linear fit)
    6. ``ig_kl_rate_5``       -- KL divergence rate (lag=5)
    7. ``ig_kl_rate_20``      -- KL divergence rate (lag=20)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``close`` column (case-insensitive).
    config : FeatureConfig | None
        Configuration; uses defaults if ``None``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with 7 additional ``ig_`` columns appended.
        The first ~``ig_rolling_window`` rows will be NaN (warmup).
    """
    cfg = config or FeatureConfig()

    # Resolve close column (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close")
    if close_col is None:
        raise ValueError(
            "DataFrame must contain a 'close' column for information "
            "geometry features."
        )

    close = df[close_col].values.astype(np.float64)
    n = len(close)

    window = cfg.ig_rolling_window
    n_bins = cfg.ig_n_bins
    delta = cfg.ig_fisher_delta
    alphas = cfg.ig_renyi_alphas
    kl_lags = cfg.ig_kl_lags

    # Pre-allocate output arrays
    fisher_info = np.full(n, np.nan)
    renyi_h = {a: np.full(n, np.nan) for a in alphas}
    renyi_slope = np.full(n, np.nan)
    kl_rates = {lag: np.full(n, np.nan) for lag in kl_lags}

    # Compute log-returns
    log_returns = np.empty(n)
    log_returns[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns[1:] = np.log(close[1:] / close[:-1])

    # Rolling computation
    for t in range(window, n):
        rets = log_returns[t - window + 1 : t + 1]
        # Skip if any NaN in the window
        if np.any(np.isnan(rets)):
            continue

        # --- Fisher Information ---
        fisher_info[t] = _fisher_information(rets, n_bins, delta)

        # --- Renyi Entropy Spectrum ---
        p, _ = _histogram_density(rets, n_bins)
        h_values = []
        for a in alphas:
            h_a = _renyi_entropy(p, a)
            renyi_h[a][t] = h_a
            h_values.append(h_a)

        # --- Renyi Slope (linear regression of H vs alpha) ---
        alpha_arr = np.array(alphas)
        h_arr = np.array(h_values)
        if len(alpha_arr) >= 2:
            slope, _ = np.polyfit(alpha_arr, h_arr, 1)
            renyi_slope[t] = slope

        # --- KL Divergence Rate ---
        for lag in kl_lags:
            t_past = t - lag
            if t_past < window:
                continue
            rets_past = log_returns[t_past - window + 1 : t_past + 1]
            if np.any(np.isnan(rets_past)):
                continue

            # Common bin edges (union range)
            lo = min(rets.min(), rets_past.min())
            hi = max(rets.max(), rets_past.max())
            if hi - lo < 1e-12:
                kl_rates[lag][t] = 0.0
                continue
            common_edges = np.linspace(lo, hi, n_bins + 1)

            p_now, _ = _histogram_density(rets, n_bins, bin_edges=common_edges)
            p_past, _ = _histogram_density(rets_past, n_bins, bin_edges=common_edges)

            kl = _kl_divergence(p_now, p_past)
            kl_rates[lag][t] = kl / lag

    # Assemble output DataFrame
    result = df.copy()
    result["ig_fisher_info"] = fisher_info
    for a in alphas:
        result[f"ig_renyi_h_{a}"] = renyi_h[a]
    result["ig_renyi_slope"] = renyi_slope
    for lag in kl_lags:
        result[f"ig_kl_rate_{lag}"] = kl_rates[lag]

    n_ig_cols = 1 + len(alphas) + 1 + len(kl_lags)
    logger.debug(
        "Computed {} information geometry features; warmup={} rows",
        n_ig_cols,
        window,
    )
    return result
