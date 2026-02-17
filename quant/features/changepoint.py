"""Bayesian Online Changepoint Detection (BOCPD).

Implements the Adams & MacKay (2007) algorithm with a Normal-Inverse-Gamma
conjugate prior so that posterior updates are analytic.  The detector is
fully online: call :meth:`update` with each new scalar observation and
receive back ``(cp_score, severity)``.

State is fully serialisable via :meth:`get_state` / :meth:`load_state`
so the detector can survive process restarts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from scipy.special import gammaln

from quant.config.settings import FeatureConfig


class BayesianChangePointDetector:
    """Online changepoint detector using BOCPD with a Normal-Inverse-Gamma prior.

    Parameters
    ----------
    config : FeatureConfig, optional
        Provides ``bocpd_lambda``, ``bocpd_alpha0``, ``bocpd_beta0``,
        ``bocpd_kappa0``, ``bocpd_mu0``.  A default ``FeatureConfig()`` is
        used when *config* is ``None``.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        cfg = config or FeatureConfig()

        # Hazard function: constant hazard  H(r) = 1 / lambda
        self._hazard = 1.0 / cfg.bocpd_lambda

        # Prior hyper-parameters (Normal-Inverse-Gamma)
        self._alpha0 = cfg.bocpd_alpha0
        self._beta0 = cfg.bocpd_beta0
        self._kappa0 = cfg.bocpd_kappa0
        self._mu0 = cfg.bocpd_mu0

        # Sufficient statistics for each possible run length.
        # Index i corresponds to run length i.
        # We start with a single run length of 0.
        self._mu: np.ndarray = np.array([self._mu0])
        self._kappa: np.ndarray = np.array([self._kappa0])
        self._alpha: np.ndarray = np.array([self._alpha0])
        self._beta: np.ndarray = np.array([self._beta0])

        # Run-length posterior  P(r_t | x_{1:t})
        self._run_length_posterior: np.ndarray = np.array([1.0])

        self._t: int = 0  # number of observations processed
        self._max_run_len: int = int(3 * cfg.bocpd_lambda)  # truncation

        logger.debug(
            "BOCPD initialised: lambda={}, alpha0={}, beta0={}, kappa0={}, mu0={}",
            cfg.bocpd_lambda,
            self._alpha0,
            self._beta0,
            self._kappa0,
            self._mu0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, x: float) -> tuple[float, float]:
        """Incorporate a single observation and return changepoint score.

        Parameters
        ----------
        x : float
            New data point (e.g. a log-return).

        Returns
        -------
        cp_score : float
            Probability of a changepoint at the current time step, in [0, 1].
        severity : float
            Magnitude of the shift expressed as the absolute z-score of *x*
            under the current predictive distribution (>= 0).
        """
        self._t += 1

        # 1. Evaluate predictive probability  p(x_t | r_{t-1}, x_{1:t-1})
        #    under the Student-t posterior-predictive of the NIG model.
        pred_prob = self._predictive_prob(x)

        # 2. Growth probabilities  (staying in the same run)
        growth_prob = self._run_length_posterior * pred_prob * (1.0 - self._hazard)

        # 3. Changepoint probability  (new run starts)
        cp_prob_mass = np.sum(
            self._run_length_posterior * pred_prob * self._hazard
        )

        # 4. Assemble new joint distribution and normalise
        new_joint = np.empty(len(growth_prob) + 1)
        new_joint[0] = cp_prob_mass
        new_joint[1:] = growth_prob

        evidence = new_joint.sum()
        if evidence > 0:
            new_joint /= evidence
        else:
            # Degenerate case: reset to uniform
            new_joint[:] = 1.0 / len(new_joint)

        # 5. Update sufficient statistics
        self._update_suffstats(x)

        # 6. Store new posterior
        self._run_length_posterior = new_joint

        # 7. Truncate to keep arrays bounded
        if len(self._run_length_posterior) > self._max_run_len:
            self._truncate()

        # 8. Derive outputs
        cp_score = float(new_joint[0])  # P(r_t = 0)

        # Severity: |z-score| under the marginal Student-t with the longest
        # run's sufficient statistics (most "stable" estimate).
        severity = self._severity(x)

        return cp_score, severity

    def get_state(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the detector state."""
        return {
            "hazard": self._hazard,
            "alpha0": self._alpha0,
            "beta0": self._beta0,
            "kappa0": self._kappa0,
            "mu0": self._mu0,
            "mu": self._mu.tolist(),
            "kappa": self._kappa.tolist(),
            "alpha": self._alpha.tolist(),
            "beta": self._beta.tolist(),
            "run_length_posterior": self._run_length_posterior.tolist(),
            "t": self._t,
            "max_run_len": self._max_run_len,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore detector state from a previous :meth:`get_state` call."""
        self._hazard = state["hazard"]
        self._alpha0 = state["alpha0"]
        self._beta0 = state["beta0"]
        self._kappa0 = state["kappa0"]
        self._mu0 = state["mu0"]
        self._mu = np.array(state["mu"], dtype=np.float64)
        self._kappa = np.array(state["kappa"], dtype=np.float64)
        self._alpha = np.array(state["alpha"], dtype=np.float64)
        self._beta = np.array(state["beta"], dtype=np.float64)
        self._run_length_posterior = np.array(
            state["run_length_posterior"], dtype=np.float64
        )
        self._t = state["t"]
        self._max_run_len = state["max_run_len"]
        logger.debug("BOCPD state restored at t={}", self._t)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predictive_prob(self, x: float) -> np.ndarray:
        """Student-t predictive probability for each run length.

        Under the NIG model the posterior predictive is::

            t_{2 alpha}(mu, beta (kappa + 1) / (alpha kappa))

        We evaluate its density at *x* for every current run length.
        """
        df = 2.0 * self._alpha  # degrees of freedom
        mu = self._mu
        var = self._beta * (self._kappa + 1.0) / (self._alpha * self._kappa + 1e-30)
        scale = np.sqrt(var + 1e-30)

        z = (x - mu) / scale
        # Log pdf of Student-t
        log_pdf = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale + 1e-30)
            - ((df + 1.0) / 2.0) * np.log1p(z ** 2 / df)
        )
        return np.exp(log_pdf)

    def _update_suffstats(self, x: float) -> None:
        """Extend sufficient statistics for the new observation.

        After seeing *x*, the run-length vector grows by 1.  Index 0
        corresponds to a fresh run (prior hyper-parameters).
        """
        mu_new = (self._kappa * self._mu + x) / (self._kappa + 1.0)
        kappa_new = self._kappa + 1.0
        alpha_new = self._alpha + 0.5
        beta_new = (
            self._beta
            + 0.5 * self._kappa * (x - self._mu) ** 2 / (self._kappa + 1.0)
        )

        # Prepend prior parameters for the new run (r=0)
        self._mu = np.concatenate([[self._mu0], mu_new])
        self._kappa = np.concatenate([[self._kappa0], kappa_new])
        self._alpha = np.concatenate([[self._alpha0], alpha_new])
        self._beta = np.concatenate([[self._beta0], beta_new])

    def _truncate(self) -> None:
        """Truncate run-length arrays to ``_max_run_len`` to bound memory."""
        n = self._max_run_len
        self._run_length_posterior = self._run_length_posterior[:n]
        # Re-normalise after truncation
        total = self._run_length_posterior.sum()
        if total > 0:
            self._run_length_posterior /= total
        self._mu = self._mu[:n]
        self._kappa = self._kappa[:n]
        self._alpha = self._alpha[:n]
        self._beta = self._beta[:n]

    def _severity(self, x: float) -> float:
        """Absolute z-score of *x* under the most probable non-zero run."""
        if len(self._mu) < 2:
            return 0.0
        # Use the run-length with highest posterior (excluding r=0)
        idx = int(np.argmax(self._run_length_posterior[1:])) + 1
        mu = self._mu[idx]
        var = self._beta[idx] * (self._kappa[idx] + 1.0) / (
            self._alpha[idx] * self._kappa[idx] + 1e-30
        )
        std = np.sqrt(var + 1e-30)
        return float(np.abs(x - mu) / (std + 1e-30))
