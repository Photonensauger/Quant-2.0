"""Data providers for fetching OHLCV data from various sources."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from quant.config.settings import DataConfig

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataProvider(ABC):
    """Abstract base class for all market data providers."""

    _INTERVAL_MAP: ClassVar[dict[str, str]] = {}

    def __init__(self, config: DataConfig | None = None) -> None:
        self.config = config or DataConfig()

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Returns DataFrame with DatetimeIndex (UTC) and columns
        [open, high, low, close, volume], all float64.
        """

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and repair an OHLCV DataFrame.

        Guarantees:
        - DatetimeIndex in UTC, monotonic, no duplicates
        - Columns are exactly OHLCV_COLUMNS, all float64
        - No NaN in OHLC (rows with NaN OHLC are dropped; volume NaN -> 0)
        - high >= max(open, close), low <= min(open, close)
        """
        if df.empty:
            return df

        df = df.copy()

        # Ensure correct columns exist
        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        df = df[OHLCV_COLUMNS]

        # Cast dtypes
        df = df.astype(np.float64)

        # Ensure UTC DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "datetime"

        # Remove duplicates, sort
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        # Drop rows with NaN in OHLC, fill volume NaN with 0
        df = df.dropna(subset=["open", "high", "low", "close"])
        df["volume"] = df["volume"].fillna(0.0)

        # Fix high/low consistency
        row_max = df[["open", "close"]].max(axis=1)
        row_min = df[["open", "close"]].min(axis=1)
        df["high"] = df["high"].clip(lower=row_max)
        df["low"] = df["low"].clip(upper=row_min)

        return df

    def _retry_fetch(self, fetch_fn, symbol: str) -> pd.DataFrame:
        """Execute fetch_fn with exponential backoff retries."""
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                return fetch_fn()
            except Exception as exc:
                last_exc = exc
                wait = self.config.retry_backoff_base * (2 ** attempt)
                logger.warning(
                    "Fetch attempt {}/{} for {} failed: {}. Retrying in {:.1f}s",
                    attempt + 1,
                    self.config.max_retries,
                    symbol,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error("All {} retries exhausted for {}: {}", self.config.max_retries, symbol, last_exc)
        raise last_exc  # type: ignore[misc]


class YFinanceProvider(DataProvider):
    """Provider for equities and ETFs via Yahoo Finance."""

    _INTERVAL_MAP: ClassVar[dict[str, str]] = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "1d": "1d",
    }

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        logger.info("YFinance fetching {} {} from {} to {}", symbol, interval, start, end)
        yf_interval = self._INTERVAL_MAP.get(interval, interval)

        def _fetch() -> pd.DataFrame:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                interval=yf_interval,
                start=start,
                end=end,
                auto_adjust=True,
            )
            return df

        try:
            df = self._retry_fetch(_fetch, symbol)
        except Exception:
            logger.warning("YFinance returned no data for {} after retries", symbol)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if df.empty:
            logger.warning("YFinance returned empty DataFrame for {}", symbol)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df.columns = df.columns.str.lower()
        df = df.rename(columns={"stock splits": "stock_splits"})
        df = df[[c for c in OHLCV_COLUMNS if c in df.columns]]

        return self.validate_dataframe(df)


class CryptoProvider(DataProvider):
    """Provider for cryptocurrency pairs via ccxt (Binance default)."""

    _INTERVAL_MAP: ClassVar[dict[str, str]] = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "1d": "1d",
    }

    def __init__(
        self,
        config: DataConfig | None = None,
        exchange_id: str = "binance",
    ) -> None:
        super().__init__(config)
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange: ccxt.Exchange = exchange_class({"enableRateLimit": True})

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        logger.info("ccxt fetching {} {} from {} to {}", symbol, interval, start, end)
        timeframe = self._INTERVAL_MAP.get(interval, interval)
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_candles: list[list] = []

        def _fetch_page() -> list[list]:
            nonlocal since_ms
            candles = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=1000,
            )
            return candles

        try:
            while since_ms < end_ms:
                candles = self._retry_fetch(_fetch_page, symbol)
                if not candles:
                    break
                all_candles.extend(candles)
                since_ms = candles[-1][0] + 1
        except Exception:
            logger.warning("ccxt fetch failed for {} after retries", symbol)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if not all_candles:
            logger.warning("ccxt returned no candles for {}", symbol)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.DataFrame(all_candles, columns=["timestamp"] + OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df.index.name = "datetime"

        # Trim to requested range
        df = df.loc[start:end]

        return self.validate_dataframe(df)


class ForexProvider(DataProvider):
    """Provider for forex pairs via Yahoo Finance (e.g. EURUSD=X)."""

    _INTERVAL_MAP: ClassVar[dict[str, str]] = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "1d": "1d",
    }

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        logger.info("Forex fetching {} {} from {} to {}", symbol, interval, start, end)
        yf_interval = self._INTERVAL_MAP.get(interval, interval)

        def _fetch() -> pd.DataFrame:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                interval=yf_interval,
                start=start,
                end=end,
                auto_adjust=True,
            )
            return df

        try:
            df = self._retry_fetch(_fetch, symbol)
        except Exception:
            logger.warning("Forex fetch failed for {} after retries", symbol)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if df.empty:
            logger.warning("Forex returned empty DataFrame for {}", symbol)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df.columns = df.columns.str.lower()
        df = df[[c for c in OHLCV_COLUMNS if c in df.columns]]

        # Forex volume is often 0; keep as-is
        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        return self.validate_dataframe(df)
