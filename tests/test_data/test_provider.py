"""Tests for quant.data.provider data providers and validation."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import DataConfig
from quant.data.provider import (
    OHLCV_COLUMNS,
    CryptoProvider,
    DataProvider,
    ForexProvider,
    YFinanceProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 1, 2, tzinfo=timezone.utc)


def _make_raw_ohlcv(n: int = 100, tz: str | None = "UTC") -> pd.DataFrame:
    """Create a raw OHLCV DataFrame (may need validation)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz=tz)
    np.random.seed(0)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.05,
            "high": close + np.abs(np.random.randn(n)) * 0.5,
            "low": close - np.abs(np.random.randn(n)) * 0.5,
            "close": close,
            "volume": np.random.lognormal(15, 1, n),
        },
        index=dates,
    )


def _make_yf_history_df(n: int = 100) -> pd.DataFrame:
    """Create a DataFrame mimicking yfinance Ticker.history() output."""
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="America/New_York")
    np.random.seed(1)
    close = 150.0 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.05,
            "High": close + np.abs(np.random.randn(n)) * 0.5,
            "Low": close - np.abs(np.random.randn(n)) * 0.5,
            "Close": close,
            "Volume": np.random.lognormal(15, 1, n),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# validate_dataframe tests
# ---------------------------------------------------------------------------
class TestValidateDataframe:
    """Tests for the DataProvider.validate_dataframe method."""

    def _get_provider(self) -> DataProvider:
        """Return a concrete provider to access validate_dataframe."""
        with patch("quant.data.provider.yf"):
            return YFinanceProvider()

    def test_validate_dataframe_fixes_types(self) -> None:
        provider = self._get_provider()
        df = _make_raw_ohlcv(50)
        # Intentionally set columns to int to test type coercion
        df["volume"] = df["volume"].astype(int)
        result = provider.validate_dataframe(df)
        for col in OHLCV_COLUMNS:
            assert result[col].dtype == np.float64, f"{col} should be float64"
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_validate_dataframe_drops_nan_ohlc(self) -> None:
        provider = self._get_provider()
        df = _make_raw_ohlcv(50)
        # Inject NaN in OHLC columns (rows should be dropped)
        df.iloc[5, df.columns.get_loc("close")] = np.nan
        df.iloc[10, df.columns.get_loc("open")] = np.nan
        # Inject NaN in volume (should be filled with 0, not dropped)
        df.iloc[15, df.columns.get_loc("volume")] = np.nan

        result = provider.validate_dataframe(df)
        assert len(result) == 48  # 50 - 2 dropped OHLC NaN rows
        assert result["volume"].isna().sum() == 0

    def test_validate_dataframe_enforces_high_low(self) -> None:
        provider = self._get_provider()
        df = _make_raw_ohlcv(50)
        # Force high below close (should be clipped up)
        df.iloc[3, df.columns.get_loc("high")] = df.iloc[3]["close"] - 10
        # Force low above open (should be clipped down)
        df.iloc[7, df.columns.get_loc("low")] = df.iloc[7]["open"] + 10

        result = provider.validate_dataframe(df)
        for i in range(len(result)):
            row = result.iloc[i]
            assert row["high"] >= max(row["open"], row["close"])
            assert row["low"] <= min(row["open"], row["close"])

    def test_validate_removes_duplicates(self) -> None:
        provider = self._get_provider()
        df = _make_raw_ohlcv(50)
        # Duplicate the first 5 rows (same index)
        dup = df.iloc[:5].copy()
        df_with_dups = pd.concat([df, dup])
        assert len(df_with_dups) == 55

        result = provider.validate_dataframe(df_with_dups)
        assert len(result) == 50  # duplicates removed
        assert result.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# YFinanceProvider tests
# ---------------------------------------------------------------------------
class TestYFinanceProvider:
    def test_yfinance_provider_returns_correct_schema(self) -> None:
        mock_history_df = _make_yf_history_df(100)

        with patch("quant.data.provider.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_history_df
            mock_yf.Ticker.return_value = mock_ticker

            provider = YFinanceProvider()
            result = provider.fetch_ohlcv("AAPL", "5m", START, END)

        assert list(result.columns) == OHLCV_COLUMNS
        assert isinstance(result.index, pd.DatetimeIndex)
        assert str(result.index.tz) == "UTC"
        for col in OHLCV_COLUMNS:
            assert result[col].dtype == np.float64


# ---------------------------------------------------------------------------
# CryptoProvider tests
# ---------------------------------------------------------------------------
class TestCryptoProvider:
    def test_crypto_provider_returns_correct_schema(self) -> None:
        # Build fake ccxt candle data: [timestamp_ms, open, high, low, close, volume]
        n = 100
        np.random.seed(2)
        base_ts = int(START.timestamp() * 1000)
        candles = []
        close = 40000.0
        for i in range(n):
            ts = base_ts + i * 300_000  # 5-min intervals in ms
            o = close + np.random.randn() * 10
            h = max(o, close) + abs(np.random.randn()) * 20
            lo = min(o, close) - abs(np.random.randn()) * 20
            vol = abs(np.random.randn()) * 100
            candles.append([ts, o, h, lo, close, vol])
            close = close + np.random.randn() * 10

        with patch("quant.data.provider.ccxt") as mock_ccxt:
            mock_exchange_cls = MagicMock()
            mock_exchange = MagicMock()
            # Return candles on first call, empty on second to break the while loop
            mock_exchange.fetch_ohlcv.side_effect = [candles, []]
            mock_exchange_cls.return_value = mock_exchange
            mock_ccxt.binance = mock_exchange_cls

            provider = CryptoProvider.__new__(CryptoProvider)
            provider.config = DataConfig()
            provider.exchange = mock_exchange

            result = provider.fetch_ohlcv("BTC/USDT", "5m", START, END)

        assert list(result.columns) == OHLCV_COLUMNS
        assert isinstance(result.index, pd.DatetimeIndex)
        for col in OHLCV_COLUMNS:
            assert result[col].dtype == np.float64


# ---------------------------------------------------------------------------
# ForexProvider tests
# ---------------------------------------------------------------------------
class TestForexProvider:
    def test_forex_provider_returns_correct_schema(self) -> None:
        mock_history_df = _make_yf_history_df(80)

        with patch("quant.data.provider.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_history_df
            mock_yf.Ticker.return_value = mock_ticker

            provider = ForexProvider()
            result = provider.fetch_ohlcv("EURUSD=X", "1h", START, END)

        assert list(result.columns) == OHLCV_COLUMNS
        assert isinstance(result.index, pd.DatetimeIndex)
        assert str(result.index.tz) == "UTC"


# ---------------------------------------------------------------------------
# Retry and empty-response tests
# ---------------------------------------------------------------------------
class TestProviderRetry:
    def test_provider_retry_on_failure(self) -> None:
        """Provider retries the configured number of times before raising."""
        config = DataConfig(max_retries=3, retry_backoff_base=0.0)  # instant retries

        with patch("quant.data.provider.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.side_effect = ConnectionError("network error")
            mock_yf.Ticker.return_value = mock_ticker

            provider = YFinanceProvider(config=config)
            # YFinanceProvider catches the exception after retry exhaustion
            # and returns an empty DataFrame
            result = provider.fetch_ohlcv("AAPL", "5m", START, END)

        assert result.empty
        assert mock_ticker.history.call_count == 3

    def test_provider_empty_response(self) -> None:
        """Provider returns empty DataFrame when API returns no data."""
        with patch("quant.data.provider.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_yf.Ticker.return_value = mock_ticker

            provider = YFinanceProvider()
            result = provider.fetch_ohlcv("INVALID", "5m", START, END)

        assert result.empty
        assert list(result.columns) == OHLCV_COLUMNS
