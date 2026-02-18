"""Fixtures for dashboard tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
