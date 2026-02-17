#!/usr/bin/env python3
"""CLI for downloading and caching market data.

Usage
-----
# Download 30 days of 5-minute bars for stocks
python scripts/download_data.py --assets AAPL,MSFT --interval 5m --days 30

# Download crypto data
python scripts/download_data.py --assets BTC/USDT,ETH/USDT --interval 1h --days 60

# Download forex data
python scripts/download_data.py --assets EURUSD=X --interval 1d --days 365

# Force re-download (invalidate cache)
python scripts/download_data.py --assets AAPL --interval 5m --days 30 --force
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from quant.config.settings import SystemConfig
from quant.data.provider import CryptoProvider, ForexProvider, YFinanceProvider
from quant.data.storage import DataStorage
from quant.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_symbol(symbol: str) -> str:
    """Heuristic to classify a symbol as stock, crypto, or forex.

    Rules:
        - Contains "/" -> crypto  (e.g. BTC/USDT)
        - Ends with "=X" -> forex (e.g. EURUSD=X)
        - Otherwise -> stock
    """
    if "/" in symbol:
        return "crypto"
    if symbol.upper().endswith("=X"):
        return "forex"
    return "stock"


def create_provider(symbol_type: str, config: SystemConfig):
    """Instantiate the appropriate DataProvider for a symbol type."""
    if symbol_type == "crypto":
        return CryptoProvider(config.data)
    if symbol_type == "forex":
        return ForexProvider(config.data)
    return YFinanceProvider(config.data)


def format_size(n_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and cache OHLCV market data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_data.py --assets AAPL,MSFT --interval 5m --days 30\n"
            "  python scripts/download_data.py --assets BTC/USDT --interval 1h --days 60\n"
            "  python scripts/download_data.py --assets EURUSD=X --interval 1d --days 365\n"
        ),
    )
    parser.add_argument(
        "--assets",
        type=str,
        required=True,
        help="Comma-separated list of symbols (e.g. AAPL,MSFT or BTC/USDT).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=["1m", "5m", "15m", "1h", "1d"],
        help="Bar interval (default: 5m).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of calendar days of history to download (default: 30).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Invalidate cached data and re-download from scratch.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- Configuration -------------------------------------------------------
    config = SystemConfig()
    setup_logging(args.log_level)

    symbols = [s.strip() for s in args.assets.split(",") if s.strip()]
    if not symbols:
        print("ERROR: No symbols provided.", file=sys.stderr)
        return 1

    interval = args.interval
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.days)

    storage = DataStorage(config.data)

    print(f"\n{'=' * 60}")
    print(f"  Market Data Downloader")
    print(f"{'=' * 60}")
    print(f"  Symbols:   {', '.join(symbols)}")
    print(f"  Interval:  {interval}")
    print(f"  Period:    {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')} ({args.days} days)")
    print(f"  Cache dir: {config.data.cache_dir}")
    print(f"  Force:     {args.force}")
    print(f"{'=' * 60}\n")

    # --- Download each symbol ------------------------------------------------
    results: list[dict] = []
    for symbol in symbols:
        symbol_type = classify_symbol(symbol)
        provider = create_provider(symbol_type, config)

        print(f"  [{symbol_type.upper():6s}] {symbol} ... ", end="", flush=True)

        # Optionally invalidate cache
        if args.force:
            storage.invalidate(symbol, interval)

        try:
            df = storage.get_or_fetch(provider, symbol, interval, start, end)
        except Exception as exc:
            print(f"FAILED ({exc})")
            results.append({
                "symbol": symbol,
                "type": symbol_type,
                "rows": 0,
                "status": "FAILED",
                "error": str(exc),
            })
            continue

        n_rows = len(df)
        if n_rows == 0:
            print("EMPTY (no data returned)")
            results.append({
                "symbol": symbol,
                "type": symbol_type,
                "rows": 0,
                "status": "EMPTY",
            })
            continue

        # Gather stats
        date_range = f"{df.index.min().strftime('%Y-%m-%d %H:%M')} -> {df.index.max().strftime('%Y-%m-%d %H:%M')}"
        cache_path = storage._cache_path(symbol, interval)
        file_size = cache_path.stat().st_size if cache_path.exists() else 0

        print(f"OK  {n_rows:>7,} rows  |  {date_range}  |  {format_size(file_size)}")
        results.append({
            "symbol": symbol,
            "type": symbol_type,
            "rows": n_rows,
            "status": "OK",
            "start": df.index.min(),
            "end": df.index.max(),
            "cache_size": file_size,
        })

    # --- Summary -------------------------------------------------------------
    total_rows = sum(r["rows"] for r in results)
    ok_count = sum(1 for r in results if r["status"] == "OK")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    empty_count = sum(1 for r in results if r["status"] == "EMPTY")

    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  Total symbols: {len(symbols)}")
    print(f"  Successful:    {ok_count}")
    print(f"  Empty:         {empty_count}")
    print(f"  Failed:        {failed_count}")
    print(f"  Total rows:    {total_rows:,}")
    print(f"{'=' * 60}\n")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
