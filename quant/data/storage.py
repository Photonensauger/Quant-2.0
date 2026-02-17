"""Parquet-based caching layer for market data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from quant.config.settings import DataConfig
from quant.data.provider import DataProvider, OHLCV_COLUMNS


class DataStorage:
    """Transparent cache that sits between callers and DataProviders.

    Data is stored as Parquet files keyed by (symbol, interval).
    """

    def __init__(self, config: DataConfig | None = None) -> None:
        self.config = config or DataConfig()
        self.cache_dir: Path = self.config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, interval: str) -> Path:
        safe_symbol = symbol.replace("/", "-").replace("=", "_")
        return self.cache_dir / f"{safe_symbol}_{interval}.parquet"

    def _read_parquet(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            return df
        except Exception as exc:
            logger.warning("Corrupt Parquet file {}: {}. Deleting.", path, exc)
            path.unlink(missing_ok=True)
            return None

    def _write_parquet(self, df: pd.DataFrame, path: Path) -> None:
        if df.empty:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=True)
        logger.debug("Cached {} rows to {}", len(df), path)

    def get_or_fetch(
        self,
        provider: DataProvider,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return cached data if it covers [start, end], otherwise fetch and merge."""
        path = self._cache_path(symbol, interval)
        cached = self._read_parquet(path)

        start_ts = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start).tz_convert("UTC")
        end_ts = pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end).tz_convert("UTC")

        if cached is not None and not cached.empty:
            cache_start = cached.index.min()
            cache_end = cached.index.max()

            if cache_start <= start_ts and cache_end >= end_ts:
                logger.debug("Full cache hit for {} {}", symbol, interval)
                return cached.loc[start_ts:end_ts].copy()

            # Partial cache: fetch only the missing ranges and merge
            frames = [cached]

            if cache_start > start_ts:
                logger.debug("Fetching pre-cache gap for {} {}", symbol, interval)
                pre = provider.fetch_ohlcv(symbol, interval, start, cache_start.to_pydatetime())
                if not pre.empty:
                    frames.append(pre)

            if cache_end < end_ts:
                logger.debug("Fetching post-cache gap for {} {}", symbol, interval)
                post = provider.fetch_ohlcv(symbol, interval, cache_end.to_pydatetime(), end)
                if not post.empty:
                    frames.append(post)

            merged = pd.concat(frames)
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()
            self._write_parquet(merged, path)
            return merged.loc[start_ts:end_ts].copy()

        # No cache at all: full fetch
        logger.debug("Cache miss for {} {}, fetching from provider", symbol, interval)
        df = provider.fetch_ohlcv(symbol, interval, start, end)

        if df.empty:
            logger.warning("Provider returned empty data for {} {}", symbol, interval)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        self._write_parquet(df, path)
        return df.loc[start_ts:end_ts].copy()

    def invalidate(self, symbol: str, interval: str) -> None:
        """Delete cached file for a given symbol/interval pair."""
        path = self._cache_path(symbol, interval)
        if path.exists():
            path.unlink()
            logger.info("Invalidated cache for {} {}", symbol, interval)

    def invalidate_all(self) -> None:
        """Delete all cached Parquet files."""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        logger.info("Invalidated all cached data")
