"""Portfolio optimization with multiple strategy support.

Provides mean-variance, minimum-variance, risk-parity, and equal-weight
allocation strategies.  All optimizers guarantee fully-invested, long-only
portfolios (sum(weights) == 1.0, weights >= 0).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from scipy.optimize import minimize

from quant.config.settings import TradingConfig


class OptimizationMethod(str, Enum):
    """Supported portfolio optimization methods."""

    MEAN_VARIANCE = "mean_variance"
    MINIMUM_VARIANCE = "minimum_variance"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"


class PortfolioOptimizer:
    """Long-only portfolio optimizer using scipy SLSQP.

    Parameters
    ----------
    config : TradingConfig
        Trading configuration (used for position limits).
    method : OptimizationMethod
        Default optimization method.  Can be overridden per call.
    risk_aversion : float
        Risk-aversion parameter for mean-variance (higher = more conservative).
    """

    def __init__(
        self,
        config: TradingConfig | None = None,
        method: OptimizationMethod = OptimizationMethod.MINIMUM_VARIANCE,
        risk_aversion: float = 1.0,
    ) -> None:
        self.config = config or TradingConfig()
        self.method = method
        self.risk_aversion = risk_aversion
        logger.info(
            "PortfolioOptimizer initialised | method={} | risk_aversion={}",
            self.method.value,
            self.risk_aversion,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
        method: OptimizationMethod | None = None,
    ) -> NDArray[np.float64]:
        """Compute optimal portfolio weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical return matrix (rows = time, columns = assets).
        expected_returns : pd.Series, optional
            Forward-looking expected returns per asset.  Required only for
            mean-variance; estimated from *returns* if not supplied.
        method : OptimizationMethod, optional
            Override the default optimization method for this call.

        Returns
        -------
        np.ndarray
            1-D array of portfolio weights with
            ``len(weights) == returns.shape[1]``.

        Guarantees
        ----------
        * ``np.isclose(weights.sum(), 1.0)``
        * ``(weights >= 0).all()``
        """
        active_method = method or self.method
        n_assets = returns.shape[1]

        if n_assets == 0:
            logger.warning("Empty returns DataFrame; returning empty weights.")
            return np.array([], dtype=np.float64)

        if n_assets == 1:
            logger.debug("Single asset; returning weight [1.0].")
            return np.array([1.0], dtype=np.float64)

        logger.debug(
            "Optimising {} assets with method={}",
            n_assets,
            active_method.value,
        )

        dispatch = {
            OptimizationMethod.MEAN_VARIANCE: self._mean_variance,
            OptimizationMethod.MINIMUM_VARIANCE: self._minimum_variance,
            OptimizationMethod.RISK_PARITY: self._risk_parity,
            OptimizationMethod.EQUAL_WEIGHT: self._equal_weight,
        }

        weights = dispatch[active_method](returns, expected_returns)
        weights = self._enforce_constraints(weights, n_assets)

        logger.debug("Optimised weights: {}", np.round(weights, 4))
        return weights

    # ------------------------------------------------------------------
    # Optimization strategies
    # ------------------------------------------------------------------

    def _mean_variance(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[pd.Series],
    ) -> NDArray[np.float64]:
        """Maximise expected return minus risk-aversion * variance."""
        n = returns.shape[1]
        cov = returns.cov().values
        mu = (
            expected_returns.values
            if expected_returns is not None
            else returns.mean().values
        )

        def objective(w: NDArray) -> float:
            port_return = w @ mu
            port_var = w @ cov @ w
            # Minimise negative utility
            return -(port_return - 0.5 * self.risk_aversion * port_var)

        weights = self._solve(objective, n)
        return weights

    def _minimum_variance(
        self,
        returns: pd.DataFrame,
        _expected_returns: Optional[pd.Series],
    ) -> NDArray[np.float64]:
        """Minimise portfolio variance (ignores expected returns)."""
        n = returns.shape[1]
        cov = returns.cov().values

        def objective(w: NDArray) -> float:
            return float(w @ cov @ w)

        weights = self._solve(objective, n)
        return weights

    def _risk_parity(
        self,
        returns: pd.DataFrame,
        _expected_returns: Optional[pd.Series],
    ) -> NDArray[np.float64]:
        """Equalise marginal risk contribution across assets.

        Minimises ``sum_i (w_i * (Sigma @ w)_i / sigma_p - 1/n)^2``.
        """
        n = returns.shape[1]
        cov = returns.cov().values
        target_risk = 1.0 / n

        def objective(w: NDArray) -> float:
            port_var = w @ cov @ w
            if port_var < 1e-12:
                return 0.0
            port_std = np.sqrt(port_var)
            marginal = cov @ w
            risk_contrib = w * marginal / port_std
            total_rc = risk_contrib.sum()
            if total_rc < 1e-12:
                return 0.0
            rc_pct = risk_contrib / total_rc
            return float(np.sum((rc_pct - target_risk) ** 2))

        weights = self._solve(objective, n)
        return weights

    @staticmethod
    def _equal_weight(
        returns: pd.DataFrame,
        _expected_returns: Optional[pd.Series],
    ) -> NDArray[np.float64]:
        """1/N allocation (no optimisation needed)."""
        n = returns.shape[1]
        return np.full(n, 1.0 / n, dtype=np.float64)

    # ------------------------------------------------------------------
    # Solver helper
    # ------------------------------------------------------------------

    @staticmethod
    def _solve(
        objective,
        n_assets: int,
        max_iter: int = 1000,
    ) -> NDArray[np.float64]:
        """Run SLSQP with long-only, fully-invested constraints."""
        w0 = np.full(n_assets, 1.0 / n_assets)
        bounds = [(0.0, 1.0)] * n_assets
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-12},
        )

        if not result.success:
            logger.warning(
                "Optimiser did not converge: {}. Using last iterate.",
                result.message,
            )

        return result.x

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _enforce_constraints(
        weights: NDArray[np.float64],
        n_assets: int,
    ) -> NDArray[np.float64]:
        """Hard-clamp weights to satisfy sum == 1 and >= 0."""
        # Clip negatives (numerical noise)
        w = np.clip(weights, 0.0, None)

        total = w.sum()
        if total < 1e-12:
            # Degenerate: fall back to equal weight
            logger.warning("All weights near zero after clipping; using 1/N.")
            w = np.full(n_assets, 1.0 / n_assets)
        else:
            w /= total

        return w.astype(np.float64)
