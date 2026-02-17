"""Tests for quant.features.changepoint -- Bayesian Online Changepoint Detection."""

from __future__ import annotations

import numpy as np
import pytest

from quant.config.settings import FeatureConfig
from quant.features.changepoint import BayesianChangePointDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def detector() -> BayesianChangePointDetector:
    """Return a freshly initialised BOCPD detector with default config."""
    return BayesianChangePointDetector(FeatureConfig())


def _feed_series(
    detector: BayesianChangePointDetector, data: np.ndarray
) -> list[tuple[float, float]]:
    """Feed an array through the detector and collect (cp_score, severity)."""
    return [detector.update(float(x)) for x in data]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBayesianChangePointDetector:
    """Tests for the BOCPD detector."""

    def test_bocpd_update_returns_tuple(
        self, detector: BayesianChangePointDetector
    ) -> None:
        """update() must return a (cp_score, severity) tuple of floats."""
        result = detector.update(0.01)

        assert isinstance(result, tuple)
        assert len(result) == 2

        cp_score, severity = result
        assert isinstance(cp_score, float)
        assert isinstance(severity, float)

    def test_bocpd_cp_score_range(
        self, detector: BayesianChangePointDetector
    ) -> None:
        """Changepoint score must be in [0, 1] for every observation."""
        np.random.seed(42)
        data = np.random.randn(200) * 0.01
        results = _feed_series(detector, data)

        for i, (cp_score, _) in enumerate(results):
            assert 0.0 <= cp_score <= 1.0, (
                f"cp_score={cp_score} out of [0,1] at step {i}"
            )

    def test_bocpd_detects_synthetic_changepoint(
        self, detector: BayesianChangePointDetector
    ) -> None:
        """Detector should show elevated cp_score after a mean shift."""
        np.random.seed(123)

        # Phase 1: 100 observations around mean 0
        phase1 = np.random.randn(100) * 0.01

        # Phase 2: abrupt shift to mean=2
        phase2 = 2.0 + np.random.randn(100) * 0.01

        data = np.concatenate([phase1, phase2])
        results = _feed_series(detector, data)

        # cp_score around the changepoint (indices 100-110) should be
        # noticeably higher than the baseline (indices 50-90).
        baseline_scores = [results[i][0] for i in range(50, 90)]
        transition_scores = [results[i][0] for i in range(100, 115)]

        baseline_mean = np.mean(baseline_scores)
        transition_max = np.max(transition_scores)

        assert transition_max > baseline_mean, (
            f"Expected elevated cp_score near changepoint. "
            f"baseline mean={baseline_mean:.4f}, "
            f"transition max={transition_max:.4f}"
        )

    def test_bocpd_state_roundtrip(
        self, detector: BayesianChangePointDetector
    ) -> None:
        """get_state -> load_state must produce identical subsequent outputs."""
        np.random.seed(7)
        warmup_data = np.random.randn(50) * 0.01

        # Feed some data to build up state
        for x in warmup_data:
            detector.update(float(x))

        # Snapshot state
        state = detector.get_state()

        # Continue the original detector with fresh data
        np.random.seed(99)
        test_data = np.random.randn(10) * 0.01
        original_results = [detector.update(float(x)) for x in test_data]

        # Restore a new detector from the snapshot
        restored = BayesianChangePointDetector()
        restored.load_state(state)
        restored_results = [restored.update(float(x)) for x in test_data]

        for i, (orig, rest) in enumerate(
            zip(original_results, restored_results)
        ):
            np.testing.assert_allclose(
                orig,
                rest,
                atol=1e-12,
                err_msg=f"Mismatch at step {i} after state roundtrip",
            )

    def test_bocpd_initial_state(
        self, detector: BayesianChangePointDetector
    ) -> None:
        """Initial state should have t=0 and a single run-length entry."""
        state = detector.get_state()

        assert state["t"] == 0
        assert len(state["run_length_posterior"]) == 1
        assert state["run_length_posterior"][0] == 1.0

    def test_bocpd_severity_nonnegative(
        self, detector: BayesianChangePointDetector
    ) -> None:
        """Severity (absolute z-score) must be >= 0 for every observation."""
        np.random.seed(55)
        data = np.random.randn(100) * 0.05
        results = _feed_series(detector, data)

        for i, (_, severity) in enumerate(results):
            assert severity >= 0.0, (
                f"severity={severity} is negative at step {i}"
            )
