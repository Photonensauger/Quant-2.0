"""Tests for quant.utils.viz -- PerformanceVisualizer.

All tests use synthetic backtest results. Charts are written to tmp_path.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant.utils.viz import PerformanceVisualizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backtest_result(
    n_bars: int = 200,
    n_trades: int = 20,
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """Create a synthetic backtest result dict for visualization testing."""
    rng = np.random.RandomState(42)

    # Equity curve
    returns = rng.normal(0.0005, 0.01, size=n_bars)
    equity = initial_capital * np.exp(np.cumsum(returns))
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")

    equity_series = pd.Series(equity, index=dates)
    returns_series = equity_series.pct_change().dropna()

    # Trade log
    trades: list[dict[str, Any]] = []
    for i in range(n_trades):
        pnl = rng.normal(50.0, 200.0)
        entry_time = dates[i * (n_bars // n_trades)]
        exit_time = entry_time + timedelta(hours=rng.randint(1, 48))
        trades.append(
            {
                "pnl": pnl,
                "side": "long" if pnl >= 0 else "short",
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": 100.0 + rng.randn(),
                "exit_price": 100.0 + rng.randn() + pnl / 100.0,
                "qty": 10.0,
                "symbol": "SYNTH",
            }
        )

    # Metrics
    metrics = {
        "total_return": float(equity[-1] / initial_capital - 1),
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.08,
        "win_rate": 0.55,
        "total_trades": float(n_trades),
        "profit_factor": 1.3,
        "calmar_ratio": 2.1,
        "sortino_ratio": 2.0,
    }

    return {
        "equity_curve": equity_series,
        "returns": returns_series,
        "trades": pd.DataFrame(trades),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateDashboardCreatesHtml:

    def test_generate_dashboard_creates_html(self, tmp_path: Path) -> None:
        """generate_dashboard writes a valid HTML file to disk."""
        viz = PerformanceVisualizer()
        result = _make_backtest_result()
        output = tmp_path / "dashboard.html"

        returned_path = viz.generate_dashboard(result, output)

        assert Path(returned_path).exists()
        content = Path(returned_path).read_text()
        assert "<html" in content.lower() or "<!doctype" in content.lower() or "plotly" in content.lower()
        assert len(content) > 1000  # non-trivial HTML

    def test_generate_dashboard_creates_parent_dirs(self, tmp_path: Path) -> None:
        """generate_dashboard creates intermediate directories if needed."""
        viz = PerformanceVisualizer()
        result = _make_backtest_result()
        output = tmp_path / "deep" / "nested" / "dir" / "dash.html"

        returned_path = viz.generate_dashboard(result, output)
        assert Path(returned_path).exists()

    def test_generate_dashboard_with_list_equity(self, tmp_path: Path) -> None:
        """generate_dashboard works when equity_curve is a plain list."""
        viz = PerformanceVisualizer()
        result = _make_backtest_result()
        # Convert equity_curve to a plain list (no DatetimeIndex)
        result["equity_curve"] = list(result["equity_curve"].values)
        result["returns"] = list(result["returns"].values)

        output = tmp_path / "list_equity.html"
        returned_path = viz.generate_dashboard(result, output)
        assert Path(returned_path).exists()

    def test_generate_dashboard_without_trades(self, tmp_path: Path) -> None:
        """Dashboard renders even if no trades are present."""
        viz = PerformanceVisualizer()
        result = _make_backtest_result(n_trades=0)
        result["trades"] = None

        output = tmp_path / "no_trades.html"
        returned_path = viz.generate_dashboard(result, output)
        assert Path(returned_path).exists()

    def test_generate_dashboard_without_metrics(self, tmp_path: Path) -> None:
        """Dashboard renders even if metrics are absent."""
        viz = PerformanceVisualizer()
        result = _make_backtest_result()
        del result["metrics"]

        output = tmp_path / "no_metrics.html"
        returned_path = viz.generate_dashboard(result, output)
        assert Path(returned_path).exists()


class TestGenerateSummaryTable:

    def test_generate_summary_table(self) -> None:
        """generate_summary_table returns an HTML string containing a <table>."""
        viz = PerformanceVisualizer()
        metrics = {
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.08,
            "win_rate": 0.55,
            "total_trades": 42,
            "profit_factor": 1.3,
        }

        html = viz.generate_summary_table(metrics)

        assert "<table" in html
        assert "</table>" in html
        assert "Performance Summary" in html

        # Each metric key (title-cased) should appear somewhere
        assert "Total Return" in html
        assert "Sharpe Ratio" in html
        assert "Max Drawdown" in html

    def test_generate_summary_table_empty_metrics(self) -> None:
        """Empty metrics dict still produces a valid table structure."""
        viz = PerformanceVisualizer()
        html = viz.generate_summary_table({})

        assert "<table" in html
        assert "</table>" in html

    def test_generate_summary_table_format_values(self) -> None:
        """Values are formatted according to their type and magnitude."""
        viz = PerformanceVisualizer()
        metrics: dict[str, Any] = {
            "small_float": 0.1234,
            "large_float": 12345.6789,
            "integer_val": 42,
            "negative": -0.05,
        }

        html = viz.generate_summary_table(metrics)

        # Float formatting should be present
        assert "0.1234" in html or "+0.1234" in html
        assert "+12,346" in html or "12,346" in html or "12345" in html
        # Integer should have comma formatting
        assert "42" in html
