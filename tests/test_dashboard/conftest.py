"""Fixtures for dashboard tests."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import dash
import numpy as np
import pandas as pd
import pytest

# Create a minimal Dash app so that register_page() calls in page modules
# succeed when they are imported during test collection.
dash.Dash(__name__, use_pages=True, pages_folder="")


@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Create a temp directory with sample parquet files."""
    data_dir = tmp_path / "data" / "cache"
    data_dir.mkdir(parents=True)

    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame({
        "open": close - np.random.rand(100) * 0.5,
        "high": close + np.random.rand(100) * 1.0,
        "low": close - np.random.rand(100) * 1.0,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, size=100),
    }, index=dates)
    df.to_parquet(data_dir / "AAPL_1d.parquet")
    df.to_parquet(data_dir / "MSFT_1d.parquet")

    return data_dir


@pytest.fixture()
def tmp_backtest_dir(tmp_path):
    """Create a temp directory with sample backtest JSON files."""
    bt_dir = tmp_path / "data" / "backtest"
    bt_dir.mkdir(parents=True)

    bt_result = {
        "metrics": {
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "calmar_ratio": 1.2,
            "max_drawdown": -0.12,
            "volatility": 0.18,
            "max_drawdown_duration": 30,
        },
        "returns": (np.random.randn(100) * 0.01).tolist(),
        "equity_curve": (100 + np.cumsum(np.random.randn(100) * 0.5)).tolist(),
        "trades": [
            {
                "asset": "AAPL", "side": "long",
                "entry_time": "2024-01-10", "exit_time": "2024-01-15",
                "qty": 10, "entry_price": 150.0, "exit_price": 155.0, "pnl": 50.0,
            },
            {
                "asset": "MSFT", "side": "short",
                "entry_time": "2024-02-01", "exit_time": "2024-02-10",
                "qty": 5, "entry_price": 400.0, "exit_price": 395.0, "pnl": 25.0,
            },
        ],
    }

    with open(bt_dir / "test_run.json", "w") as f:
        json.dump(bt_result, f)

    with open(bt_dir / "second_run.json", "w") as f:
        json.dump(bt_result, f)

    return bt_dir


# ---------------------------------------------------------------------------
# Shared fixtures for component and page tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_equity_curve():
    """100-bar equity curve starting at 10_000."""
    np.random.seed(42)
    return (10_000 + np.cumsum(np.random.randn(100) * 20)).tolist()


@pytest.fixture()
def sample_returns():
    """100 daily return floats."""
    np.random.seed(42)
    return (np.random.randn(100) * 0.01).tolist()


@pytest.fixture()
def sample_returns_series():
    """pd.Series of 100 daily returns with DatetimeIndex (for heatmap)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    return pd.Series(np.random.randn(100) * 0.01, index=dates)


@pytest.fixture()
def sample_trade_log():
    """Two trades — one profit, one loss."""
    return [
        {
            "asset": "AAPL", "side": "long",
            "entry_time": "2024-01-10", "exit_time": "2024-01-15",
            "qty": 10, "entry_price": 150.0, "exit_price": 155.0, "pnl": 50.0,
        },
        {
            "asset": "MSFT", "side": "short",
            "entry_time": "2024-02-01", "exit_time": "2024-02-10",
            "qty": 5, "entry_price": 400.0, "exit_price": 405.0, "pnl": -25.0,
        },
    ]


@pytest.fixture()
def sample_bt_data(sample_equity_curve, sample_returns, sample_trade_log):
    """Complete backtest result dict."""
    timestamps = pd.date_range("2024-01-01", periods=100, freq="B").strftime("%Y-%m-%d").tolist()
    return {
        "metrics": {
            "total_return": 0.15,
            "annualized_return": 0.30,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "calmar_ratio": 1.2,
            "max_drawdown": -0.12,
            "volatility": 0.18,
            "max_drawdown_duration": 30,
            "total_trades": 2,
            "win_rate": 0.5,
            "avg_win": 50.0,
            "avg_loss": -25.0,
            "profit_factor": 2.0,
            "expectancy": 12.5,
            "initial_capital": 10000,
            "final_equity": 11500,
        },
        "equity_curve": sample_equity_curve,
        "returns": sample_returns,
        "trades": sample_trade_log,
        "timestamps": timestamps,
    }


@pytest.fixture()
def mock_loader(sample_bt_data):
    """MagicMock satisfying every DashboardDataLoader call used by pages."""
    mock = MagicMock()
    mock.list_backtest_results.return_value = [
        Path("data/backtest/test_run.json"),
        Path("data/backtest/second_run.json"),
    ]
    mock.load_backtest_result.return_value = sample_bt_data
    mock.load_all_backtest_results.return_value = {
        "test_run": sample_bt_data,
        "second_run": sample_bt_data,
    }
    return mock
