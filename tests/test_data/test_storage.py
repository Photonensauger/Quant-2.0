"""Tests for quant.data.storage Parquet caching layer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from quant.config.settings import DataConfig
from quant.data.provider import OHLCV_COLUMNS
from quant.data.storage import DataStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Use naive datetimes: storage.get_or_fetch adds tz="UTC" internally via
# pd.Timestamp(start, tz="UTC"), which raises if tzinfo is already set.
START = datetime(2024, 1, 1)
END = datetime(2024, 1, 2)


def _make_ohlcv_df(n: int = 100) -> pd.DataFrame:
    """Build a valid OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    df = pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.05,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.lognormal(15, 1, n),
        },
        index=dates,
    )
    df.index.name = "datetime"
    return df


def _make_storage(tmp_path: Path) -> DataStorage:
    """Create a DataStorage pointed at a tmp directory."""
    config = DataConfig(cache_dir=tmp_path / "cache")
    return DataStorage(config=config)


def _mock_provider(df: pd.DataFrame) -> MagicMock:
    """Create a mock DataProvider that returns df from fetch_ohlcv."""
    provider = MagicMock()
    provider.fetch_ohlcv.return_value = df
    return provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCacheWriteReadRoundtrip:
    def test_cache_write_read_roundtrip(self, tmp_dir: Path) -> None:
        """Data written to cache should be readable with identical content."""
        storage = _make_storage(tmp_dir)
        df = _make_ohlcv_df(50)

        # Write through internal method
        path = storage._cache_path("AAPL", "5m")
        storage._write_parquet(df, path)

        # Read back
        result = storage._read_parquet(path)
        assert result is not None
        assert len(result) == len(df)
        pd.testing.assert_frame_equal(result, df, check_freq=False)


class TestGetOrFetchCachesData:
    def test_get_or_fetch_caches_data(self, tmp_dir: Path) -> None:
        """get_or_fetch should cache data and not call provider on second request."""
        storage = _make_storage(tmp_dir)
        df = _make_ohlcv_df(100)
        provider = _mock_provider(df)

        # Use a time window that falls within the data range.
        # Data: 100 rows x 5min from 2024-01-01 00:00 -> last row at 2024-01-01 08:15.
        fetch_start = datetime(2024, 1, 1, 0, 0)
        fetch_end = datetime(2024, 1, 1, 4, 0)

        # First call: cache miss -> provider called
        result1 = storage.get_or_fetch(provider, "AAPL", "5m", fetch_start, fetch_end)
        assert provider.fetch_ohlcv.call_count == 1
        assert not result1.empty

        # Second call: full cache hit -> provider NOT called again
        result2 = storage.get_or_fetch(provider, "AAPL", "5m", fetch_start, fetch_end)
        assert provider.fetch_ohlcv.call_count == 1  # still 1
        assert len(result2) == len(result1)


class TestCorruptParquetFallback:
    def test_corrupt_parquet_fallback(self, tmp_dir: Path) -> None:
        """A corrupt Parquet file should be deleted and data re-fetched."""
        storage = _make_storage(tmp_dir)
        df = _make_ohlcv_df(50)

        # Write corrupt data to the cache path
        path = storage._cache_path("AAPL", "5m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"this is not valid parquet data")

        # _read_parquet should handle corruption gracefully
        result = storage._read_parquet(path)
        assert result is None
        assert not path.exists()  # corrupt file deleted

        # get_or_fetch should fetch from provider when cache is corrupt
        provider = _mock_provider(df)
        result = storage.get_or_fetch(provider, "AAPL", "5m", START, END)
        assert not result.empty
        assert provider.fetch_ohlcv.call_count == 1


class TestInvalidateCache:
    def test_invalidate_cache(self, tmp_dir: Path) -> None:
        """invalidate should delete the cached Parquet file."""
        storage = _make_storage(tmp_dir)
        df = _make_ohlcv_df(50)

        # Populate cache
        path = storage._cache_path("AAPL", "5m")
        storage._write_parquet(df, path)
        assert path.exists()

        # Invalidate
        storage.invalidate("AAPL", "5m")
        assert not path.exists()


class TestCachePathSafeChars:
    def test_cache_path_safe_chars(self, tmp_dir: Path) -> None:
        """Cache paths should handle symbols with / and = characters."""
        storage = _make_storage(tmp_dir)

        # Crypto symbol with /
        path_crypto = storage._cache_path("BTC/USDT", "5m")
        assert "/" not in path_crypto.name  # slash replaced
        assert "BTC-USDT_5m.parquet" == path_crypto.name

        # Forex symbol with =
        path_forex = storage._cache_path("EURUSD=X", "1h")
        assert "=" not in path_forex.name  # equals replaced
        assert "EURUSD_X_1h.parquet" == path_forex.name
