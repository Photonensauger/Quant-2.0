"""Tests for quant.portfolio.optimizer – PortfolioOptimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import TradingConfig
from quant.portfolio.optimizer import OptimizationMethod, PortfolioOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n_periods: int = 200, n_assets: int = 4, seed: int = 42) -> pd.DataFrame:
    """Synthetic returns matrix with realistic correlations."""
    rng = np.random.default_rng(seed)
    # Create correlated returns via Cholesky factor
    raw = rng.standard_normal((n_periods, n_assets))
    corr = np.eye(n_assets) * 0.5 + 0.5  # moderate correlation
    L = np.linalg.cholesky(corr)
    returns = (raw @ L.T) * 0.01  # daily ~1% scale
    columns = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=columns)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def returns() -> pd.DataFrame:
    return _make_returns()


@pytest.fixture()
def optimizer() -> PortfolioOptimizer:
    return PortfolioOptimizer(config=TradingConfig())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWeightsSumToOne:
    """All methods must produce weights that sum to 1."""

    @pytest.mark.parametrize("method", list(OptimizationMethod))
    def test_weights_sum_to_one(
        self, optimizer: PortfolioOptimizer, returns: pd.DataFrame, method: OptimizationMethod
    ) -> None:
        weights = optimizer.optimize(returns, method=method)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6), (
            f"Weights sum={weights.sum()} for method={method.value}"
        )


class TestWeightsNonNegative:
    """All methods should produce long-only (non-negative) weights."""

    @pytest.mark.parametrize("method", list(OptimizationMethod))
    def test_weights_non_negative(
        self, optimizer: PortfolioOptimizer, returns: pd.DataFrame, method: OptimizationMethod
    ) -> None:
        weights = optimizer.optimize(returns, method=method)
        assert (weights >= -1e-8).all(), (
            f"Negative weights found for method={method.value}: {weights}"
        )


class TestMeanVariance:
    """Mean-variance optimizer should use expected returns."""

    def test_mean_variance(self, optimizer: PortfolioOptimizer, returns: pd.DataFrame) -> None:
        weights = optimizer.optimize(returns, method=OptimizationMethod.MEAN_VARIANCE)
        assert weights.shape == (returns.shape[1],)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)

    def test_mean_variance_with_explicit_mu(
        self, optimizer: PortfolioOptimizer, returns: pd.DataFrame
    ) -> None:
        mu = pd.Series(
            [0.10, 0.01, 0.01, 0.01], index=returns.columns
        )
        weights = optimizer.optimize(
            returns, expected_returns=mu, method=OptimizationMethod.MEAN_VARIANCE
        )
        # The asset with the highest expected return should get a significant allocation
        assert weights[0] > 0.1, "High expected-return asset should have non-trivial weight"


class TestMinimumVariance:
    """Minimum-variance should produce a valid weight vector."""

    def test_minimum_variance(self, optimizer: PortfolioOptimizer, returns: pd.DataFrame) -> None:
        weights = optimizer.optimize(returns, method=OptimizationMethod.MINIMUM_VARIANCE)
        assert weights.shape == (returns.shape[1],)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)


class TestRiskParity:
    """Risk-parity should approximately equalise risk contributions."""

    def test_risk_parity(self, optimizer: PortfolioOptimizer, returns: pd.DataFrame) -> None:
        weights = optimizer.optimize(returns, method=OptimizationMethod.RISK_PARITY)
        assert weights.shape == (returns.shape[1],)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)
        # All weights should be meaningfully positive
        assert (weights > 0.01).all(), f"Risk-parity weights too skewed: {weights}"


class TestEqualWeight:
    """Equal-weight should give 1/N to every asset."""

    def test_equal_weight(self, optimizer: PortfolioOptimizer, returns: pd.DataFrame) -> None:
        weights = optimizer.optimize(returns, method=OptimizationMethod.EQUAL_WEIGHT)
        n = returns.shape[1]
        expected = np.full(n, 1.0 / n)
        np.testing.assert_allclose(weights, expected, atol=1e-10)


class TestSingleAsset:
    """A single-asset portfolio should always get weight 1.0."""

    def test_single_asset(self, optimizer: PortfolioOptimizer) -> None:
        single = _make_returns(n_periods=100, n_assets=1)
        weights = optimizer.optimize(single)
        np.testing.assert_allclose(weights, [1.0], atol=1e-10)
