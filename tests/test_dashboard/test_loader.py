"""Tests for dashboard.data.loader.DashboardDataLoader."""

import pandas as pd

from dashboard.data.loader import DashboardDataLoader


def test_list_symbols(tmp_data_dir, tmp_backtest_dir):
    loader = DashboardDataLoader(data_dir=tmp_data_dir, backtest_dir=tmp_backtest_dir)
    symbols = loader.list_symbols()
    assert len(symbols) == 2
    names = {s.symbol for s in symbols}
    assert names == {"AAPL", "MSFT"}


def test_load_market_data(tmp_data_dir, tmp_backtest_dir):
    loader = DashboardDataLoader(data_dir=tmp_data_dir, backtest_dir=tmp_backtest_dir)
    df = loader.load_market_data("AAPL", "1d")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "close" in df.columns
    assert len(df) == 100


def test_load_market_data_cached(tmp_data_dir, tmp_backtest_dir):
    loader = DashboardDataLoader(data_dir=tmp_data_dir, backtest_dir=tmp_backtest_dir)
    df1 = loader.load_market_data("AAPL", "1d")
    df2 = loader.load_market_data("AAPL", "1d")
    # Same object from cache
    assert df1 is df2


def test_load_market_data_missing(tmp_data_dir, tmp_backtest_dir):
    loader = DashboardDataLoader(data_dir=tmp_data_dir, backtest_dir=tmp_backtest_dir)
    df = loader.load_market_data("NONEXISTENT", "1d")
    assert df.empty


def test_load_backtest_result(tmp_data_dir, tmp_backtest_dir):
    loader = DashboardDataLoader(data_dir=tmp_data_dir, backtest_dir=tmp_backtest_dir)
    result = loader.load_backtest_result("test_run")
    assert isinstance(result, dict)
    assert "metrics" in result
    assert result["metrics"]["sharpe_ratio"] == 1.5
    assert len(result["trades"]) == 2


def test_load_all_backtest_results(tmp_data_dir, tmp_backtest_dir):
    loader = DashboardDataLoader(data_dir=tmp_data_dir, backtest_dir=tmp_backtest_dir)
    results = loader.load_all_backtest_results()
    assert len(results) == 2
    assert "test_run" in results
    assert "second_run" in results


def test_clear_cache(tmp_data_dir, tmp_backtest_dir):
    loader = DashboardDataLoader(data_dir=tmp_data_dir, backtest_dir=tmp_backtest_dir)
    df1 = loader.load_market_data("AAPL", "1d")
    loader.clear_cache()
    df2 = loader.load_market_data("AAPL", "1d")
    # After clear, a new object is loaded from disk
    assert df1 is not df2
    # But data should be equivalent
    pd.testing.assert_frame_equal(df1, df2)
