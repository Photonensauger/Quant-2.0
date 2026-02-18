"""Dashboard data loader — reads Parquet market data and backtest results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from dashboard.data.cache import TTLCache

# ---------------------------------------------------------------------------
# Resolve directories from env or defaults
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DATA_DIR = _PROJECT_ROOT / os.getenv("QUANT_DATA_DIR", "data/cache")
_BACKTEST_DIR = _PROJECT_ROOT / os.getenv("QUANT_BACKTEST_DIR", "data/backtest")


@dataclass(frozen=True)
class SymbolInfo:
    """Parsed symbol/interval pair extracted from a Parquet filename."""

    symbol: str
    interval: str
    path: Path


class DashboardDataLoader:
    """Reads Parquet market data and JSON backtest results for the dashboard.

    Parameters
    ----------
    data_dir : Path | None
        Directory containing ``{SYMBOL}_{interval}.parquet`` files.
        Defaults to ``$QUANT_DATA_DIR`` or ``<project>/data/cache``.
    backtest_dir : Path | None
        Directory containing backtest result JSON files.
        Defaults to ``$QUANT_BACKTEST_DIR`` or ``<project>/data/backtest``.
    cache_ttl : int
        Seconds to cache loaded data in memory.  Default 60.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        backtest_dir: Path | None = None,
        cache_ttl: int = 60,
        backtest_cache_ttl: int = 300,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self.backtest_dir = Path(backtest_dir) if backtest_dir else _BACKTEST_DIR
        self._cache = TTLCache(ttl_seconds=cache_ttl)
        self._bt_cache = TTLCache(ttl_seconds=backtest_cache_ttl)

    # ------------------------------------------------------------------
    # Symbol discovery
    # ------------------------------------------------------------------

    def list_symbols(self) -> list[SymbolInfo]:
        """Return all available symbols parsed from Parquet filenames.

        Filenames are expected to follow ``{SYMBOL}_{interval}.parquet``.
        """
        cached = self._cache.get("symbols")
        if cached is not None:
            return cached

        symbols: list[SymbolInfo] = []
        if not self.data_dir.exists():
            logger.warning("Data directory does not exist: {}", self.data_dir)
            return symbols

        for path in sorted(self.data_dir.glob("*.parquet")):
            parts = path.stem.rsplit("_", maxsplit=1)
            if len(parts) != 2:
                logger.warning("Skipping file with unexpected name: {}", path.name)
                continue
            symbol, interval = parts
            symbols.append(SymbolInfo(symbol=symbol, interval=interval, path=path))

        self._cache.set("symbols", symbols)
        logger.info("Discovered {} symbol/interval pairs", len(symbols))
        return symbols

    def available_symbols(self) -> list[str]:
        """Return deduplicated sorted list of symbol tickers."""
        return sorted({s.symbol for s in self.list_symbols()})

    def available_intervals(self, symbol: str) -> list[str]:
        """Return intervals available for a given symbol."""
        return sorted(
            s.interval for s in self.list_symbols() if s.symbol == symbol
        )

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def load_market_data(
        self, symbol: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Load OHLCV data for *symbol*/*interval* as a DataFrame.

        Returns an empty DataFrame if the file is not found.
        """
        cache_key = f"market:{symbol}:{interval}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        path = self.data_dir / f"{symbol}_{interval}.parquet"
        if not path.exists():
            logger.warning("Parquet file not found: {}", path)
            return pd.DataFrame()

        df = pd.read_parquet(path)
        self._cache.set(cache_key, df)
        logger.debug("Loaded {} rows for {}_{}", len(df), symbol, interval)
        return df

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def get_summary(self, symbol: str, interval: str = "1d") -> dict[str, Any]:
        """Compute quick summary stats from market data.

        Returns a dict with keys: ``symbol``, ``interval``, ``latest_price``,
        ``daily_change``, ``daily_change_pct``, ``high_52w``, ``low_52w``,
        ``avg_volume``, ``last_updated``.
        """
        df = self.load_market_data(symbol, interval)
        if df.empty:
            return {"symbol": symbol, "interval": interval, "error": "no data"}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        daily_change = latest["close"] - prev["close"]
        daily_change_pct = (daily_change / prev["close"] * 100) if prev["close"] else 0.0

        # 52-week window (approx 252 trading days for daily data)
        window = df.tail(252)

        return {
            "symbol": symbol,
            "interval": interval,
            "latest_price": round(float(latest["close"]), 2),
            "daily_change": round(float(daily_change), 2),
            "daily_change_pct": round(float(daily_change_pct), 2),
            "high_52w": round(float(window["high"].max()), 2),
            "low_52w": round(float(window["low"].min()), 2),
            "avg_volume": int(window["volume"].mean()),
            "last_updated": str(df.index[-1]),
        }

    # ------------------------------------------------------------------
    # Backtest results
    # ------------------------------------------------------------------

    def list_backtest_results(self) -> list[Path]:
        """Return paths of all JSON backtest result files."""
        if not self.backtest_dir.exists():
            logger.warning("Backtest directory does not exist: {}", self.backtest_dir)
            return []
        return sorted(self.backtest_dir.glob("*.json"))

    def load_backtest_result(self, name: str) -> dict[str, Any]:
        """Load a single backtest result JSON by filename (with or without .json).

        Returns the parsed dict, or an empty dict on failure.
        """
        if not name.endswith(".json"):
            name = f"{name}.json"

        cache_key = f"backtest:{name}"
        cached = self._bt_cache.get(cache_key)
        if cached is not None:
            return cached

        path = self.backtest_dir / name
        if not path.exists():
            logger.warning("Backtest result not found: {}", path)
            return {}

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load backtest result {}: {}", path, exc)
            return {}

        self._bt_cache.set(cache_key, data)
        logger.debug("Loaded backtest result: {}", name)
        return data

    def load_all_backtest_results(self) -> dict[str, dict[str, Any]]:
        """Load all backtest result JSONs. Keys are filenames (without .json)."""
        results: dict[str, dict[str, Any]] = {}
        for path in self.list_backtest_results():
            data = self.load_backtest_result(path.stem)
            if data:
                results[path.stem] = data
        return results

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Flush the in-memory TTL cache (forces re-read from disk)."""
        self._cache.clear()
        self._bt_cache.clear()
        logger.debug("Dashboard data cache cleared")
