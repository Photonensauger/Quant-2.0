#!/usr/bin/env python3
"""CLI for live / paper trading with the self-training loop.

Usage
-----
# Paper trading (simulated)
python scripts/live_trade.py --strategy ensemble --broker paper --assets AAPL --interval 5m

# Paper trading with custom capital
python scripts/live_trade.py --strategy ensemble --broker paper --assets AAPL --interval 5m --capital 50000

# Live trading (requires Alpaca API credentials in .env)
python scripts/live_trade.py --strategy ensemble --broker live --assets AAPL --interval 5m

State is automatically saved on Ctrl+C (SIGINT) and SIGTERM,
and restored on restart.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant.config.settings import SystemConfig
from quant.core.self_trainer import SelfTrainer
from quant.data.provider import YFinanceProvider
from quant.data.storage import DataStorage
from quant.execution.live import LiveExecutor
from quant.execution.paper import PaperExecutor
from quant.features.pipeline import FeaturePipeline
from quant.models import (
    AttentionLSTM,
    CombinedLoss,
    DecoderTransformer,
    ITransformer,
    MomentumTransformer,
    Trainer,
)
from quant.portfolio.position import PositionSizer
from quant.portfolio.risk import RiskManager
from quant.strategies.ensemble import EnsembleStrategy
from quant.strategies.ml_signal import MLSignalStrategy
from quant.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type] = {
    "transformer": DecoderTransformer,
    "itransformer": ITransformer,
    "lstm": AttentionLSTM,
    "momentum": MomentumTransformer,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trained_model(
    model_name: str,
    config: SystemConfig,
) -> torch.nn.Module | None:
    """Load a trained model from checkpoints.

    Looks for ``checkpoints/<model_name>_model.pt`` first, then
    ``checkpoints/<model_name>_latest.pt`` (trainer checkpoint).

    Returns None if no checkpoint is found.
    """
    checkpoint_dir = config.checkpoint_dir
    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        print(f"  WARNING: Unknown model name '{model_name}'.", file=sys.stderr)
        return None

    # Try model checkpoint first
    model_path = checkpoint_dir / f"{model_name}_model.pt"
    if model_path.exists():
        try:
            model, ckpt = model_cls.load(model_path, device=config.device)
            epoch = ckpt.get("epoch", "?")
            val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"  Loaded {model_name} from {model_path} (epoch={epoch}, val_loss={val_loss:.6f})")
            model.eval()
            return model
        except Exception as exc:
            print(f"  WARNING: Failed to load {model_path}: {exc}", file=sys.stderr)

    # Try trainer checkpoint
    trainer_path = checkpoint_dir / f"{model_name}_latest.pt"
    if trainer_path.exists():
        try:
            model = model_cls(config.model)
            state = torch.load(trainer_path, map_location=config.device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.to(config.device)
            model.eval()
            epoch = state.get("epoch", "?")
            val_loss = state.get("best_val_loss", float("inf"))
            print(f"  Loaded {model_name} from {trainer_path} (epoch={epoch}, val_loss={val_loss:.6f})")
            return model
        except Exception as exc:
            print(f"  WARNING: Failed to load {trainer_path}: {exc}", file=sys.stderr)

    print(f"  WARNING: No checkpoint found for {model_name} in {checkpoint_dir}.")
    return None


def create_trainers(
    models: dict[str, torch.nn.Module],
    config: SystemConfig,
) -> dict[str, Trainer]:
    """Create a Trainer instance for each model (for online retraining)."""
    trainers: dict[str, Trainer] = {}
    for name, model in models.items():
        criterion = CombinedLoss(mse_weight=1.0, directional_weight=0.5)
        trainer = Trainer(
            model=model,
            criterion=criterion,
            model_config=config.model,
            training_config=config.training,
            device=config.device,
        )
        trainers[name] = trainer
    return trainers


def load_recent_data(
    symbols: list[str],
    interval: str,
    lookback_days: int,
    config: SystemConfig,
) -> dict[str, pd.DataFrame]:
    """Fetch recent market data for the given symbols."""
    storage = DataStorage(config.data)
    provider = YFinanceProvider(config.data)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=lookback_days)

    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = storage.get_or_fetch(provider, symbol, interval, start, end)
        if not df.empty:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars loaded")
        else:
            print(f"  WARNING: No data for {symbol}.", file=sys.stderr)
    return data


def interval_to_seconds(interval: str) -> float:
    """Convert an interval string to seconds."""
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "1d": 86400,
    }
    return float(mapping.get(interval, 300))


# ---------------------------------------------------------------------------
# Paper trading simulation loop
# ---------------------------------------------------------------------------

def run_paper_simulation(
    self_trainer: SelfTrainer,
    data: dict[str, pd.DataFrame],
    config: SystemConfig,
    interval: str,
    speed: float,
) -> None:
    """Simulate a live data feed by replaying historical bars.

    Iterates through the available data bar by bar, feeding each bar to
    the SelfTrainer's ``process_bar`` method.  A configurable ``speed``
    multiplier controls the sleep between bars (0 = no delay).
    """
    # Use the first symbol's data as the reference timeline
    ref_symbol = list(data.keys())[0]
    ref_df = data[ref_symbol]

    # Skip the warmup period (seq_len bars are needed for the first prediction)
    seq_len = config.model.seq_len
    start_bar = max(seq_len + config.features.warmup_period, 0)
    n_bars = len(ref_df)

    if start_bar >= n_bars:
        print("  ERROR: Not enough data bars for the configured seq_len and warmup.")
        return

    bar_interval_seconds = interval_to_seconds(interval)
    sleep_time = bar_interval_seconds / speed if speed > 0 else 0

    print(f"\n  Simulating {n_bars - start_bar} bars (speed={speed}x) ...")
    print(f"  Press Ctrl+C to stop.\n")

    for bar_idx in range(start_bar, n_bars):
        # Build the lookback window for this bar
        lookback_start = max(0, bar_idx - seq_len - config.features.warmup_period)
        window = ref_df.iloc[lookback_start : bar_idx + 1].copy()

        try:
            result = self_trainer.process_bar(window)
        except RuntimeError as exc:
            if "not running" in str(exc).lower():
                break
            raise

        # Progress logging every 100 bars
        bar_num = bar_idx - start_bar + 1
        total_bars = n_bars - start_bar
        if bar_num % 100 == 0 or bar_num == total_bars:
            n_signals = len(result.get("signals", []))
            n_trades = len(result.get("trades", []))
            retrained = result.get("retrained", False)
            ts = ref_df.index[bar_idx]
            print(
                f"  Bar {bar_num:>6,}/{total_bars:,}  |  {ts}  |  "
                f"signals={n_signals}  trades={n_trades}  "
                f"retrained={'Y' if retrained else 'N'}"
            )

        # Simulated delay between bars
        if sleep_time > 0 and bar_idx < n_bars - 1:
            time.sleep(sleep_time)

    print("\n  Paper simulation complete.")


# ---------------------------------------------------------------------------
# Live trading loop
# ---------------------------------------------------------------------------

def run_live_loop(
    self_trainer: SelfTrainer,
    symbols: list[str],
    interval: str,
    config: SystemConfig,
) -> None:
    """Continuously fetch the latest bar and process it.

    Runs indefinitely until interrupted by SIGINT or SIGTERM.
    """
    storage = DataStorage(config.data)
    provider = YFinanceProvider(config.data)
    bar_interval_seconds = interval_to_seconds(interval)
    seq_len = config.model.seq_len
    warmup = config.features.warmup_period
    lookback_bars = seq_len + warmup + 10  # extra margin

    print(f"\n  Live trading loop started.")
    print(f"  Polling every {bar_interval_seconds:.0f}s for new bars.")
    print(f"  Press Ctrl+C to stop.\n")

    bar_count = 0
    while True:
        try:
            # Fetch recent data for the primary symbol
            ref_symbol = symbols[0]
            lookback_days = max(int(lookback_bars * bar_interval_seconds / 86400) + 2, 2)
            end = datetime.now(tz=timezone.utc)
            start = end - timedelta(days=lookback_days)

            df = storage.get_or_fetch(provider, ref_symbol, interval, start, end)

            if df.empty or len(df) < seq_len:
                print(f"  Waiting for data ({len(df) if not df.empty else 0} bars available)...")
                time.sleep(bar_interval_seconds)
                continue

            # Use the last seq_len + warmup bars as the window
            window = df.iloc[-(seq_len + warmup):].copy()

            result = self_trainer.process_bar(window)

            bar_count += 1
            n_signals = len(result.get("signals", []))
            n_trades = len(result.get("trades", []))
            retrained = result.get("retrained", False)
            ts = df.index[-1]

            print(
                f"  Bar {bar_count:>6,}  |  {ts}  |  "
                f"signals={n_signals}  trades={n_trades}  "
                f"retrained={'Y' if retrained else 'N'}"
            )

            # Sleep until the next bar
            time.sleep(bar_interval_seconds)

        except KeyboardInterrupt:
            print("\n  Keyboard interrupt received. Shutting down...")
            break
        except Exception as exc:
            print(f"  ERROR in live loop: {exc}", file=sys.stderr)
            time.sleep(bar_interval_seconds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live or paper trading with the self-training loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/live_trade.py --strategy ensemble --broker paper --assets AAPL --interval 5m\n"
            "  python scripts/live_trade.py --strategy ensemble --broker live --assets AAPL --interval 5m\n"
        ),
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["ml", "ensemble"],
        help="Strategy type: 'ml' (single model) or 'ensemble' (all models).",
    )
    parser.add_argument(
        "--broker",
        type=str,
        required=True,
        choices=["paper", "live"],
        help="Execution mode: 'paper' (simulated) or 'live' (Alpaca).",
    )
    parser.add_argument(
        "--assets",
        type=str,
        required=True,
        help="Comma-separated list of symbols (e.g. AAPL,MSFT).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=["1m", "5m", "15m", "1h", "1d"],
        help="Bar interval (default: 5m).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial paper trading capital (default: 100000). Ignored for live.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Days of historical data to load for feature pipeline warmup (default: 30).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help=(
            "Simulation speed multiplier for paper mode. "
            "0 = no delay (fastest), 1 = real-time, 10 = 10x faster. "
            "Only used in paper mode (default: 0)."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated model names to load (default: all available). "
            "Options: transformer, itransformer, lstm, momentum."
        ),
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

    # Determine which models to load
    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_names = list(MODEL_REGISTRY.keys())

    print(f"\n{'=' * 60}")
    print(f"  Live/Paper Trading System")
    print(f"{'=' * 60}")
    print(f"  Strategy:  {args.strategy}")
    print(f"  Broker:    {args.broker}")
    print(f"  Assets:    {', '.join(symbols)}")
    print(f"  Interval:  {args.interval}")
    print(f"  Capital:   ${args.capital:,.2f}")
    print(f"  Lookback:  {args.lookback_days} days")
    print(f"  Models:    {', '.join(model_names)}")
    print(f"  State dir: {config.state_dir}")
    print(f"  Device:    {config.device}")
    if args.broker == "paper":
        print(f"  Speed:     {args.speed}x")
    print(f"{'=' * 60}")

    # --- Load recent data for feature pipeline warmup ------------------------
    print("\n  Loading recent data for warmup...")
    data = load_recent_data(symbols, args.interval, args.lookback_days, config)
    if not data:
        print("ERROR: No data loaded. Run scripts/download_data.py first.", file=sys.stderr)
        return 1

    # --- Feature pipeline (fit on historical data) ---------------------------
    print("\n  Fitting feature pipeline...")
    pipeline = FeaturePipeline(config.features)
    ref_symbol = list(data.keys())[0]
    features, targets = pipeline.fit_transform(data[ref_symbol])

    if features.shape[0] == 0:
        print("ERROR: Feature pipeline produced zero valid rows.", file=sys.stderr)
        return 1

    n_features = features.shape[1]
    config.model.n_features = n_features
    print(f"  Features: {n_features} dimensions, {features.shape[0]:,} valid rows")

    # --- Load trained models -------------------------------------------------
    print("\n  Loading trained models...")
    models: dict[str, torch.nn.Module] = {}
    for name in model_names:
        model = load_trained_model(name, config)
        if model is not None:
            models[name] = model

    if not models:
        print("\n  WARNING: No trained models found.")
        print("  The system will run but cannot generate model-based signals.")
        print("  Run scripts/train_model.py first to train models.\n")

    # --- Create trainers for online retraining -------------------------------
    trainers = create_trainers(models, config)

    # --- Build ensemble strategy ---------------------------------------------
    print("\n  Building strategy...")
    if args.strategy == "ensemble":
        strategies: dict[str, MLSignalStrategy] = {}
        for name in models:
            strategies[name] = MLSignalStrategy(config.trading, name=f"ml_{name}")

        if not strategies:
            strategies["default"] = MLSignalStrategy(config.trading, name="ml_default")

        ensemble = EnsembleStrategy(
            config=config.trading,
            strategies=strategies,
            name="ensemble",
        )
    else:
        # Single ML strategy
        model_name = next(iter(models), "default")
        single_strategy = MLSignalStrategy(config.trading, name=f"ml_{model_name}")
        # Wrap in an ensemble with a single strategy for compatibility
        ensemble = EnsembleStrategy(
            config=config.trading,
            strategies={model_name: single_strategy},
            name="ml_single",
        )

    print(f"  Strategy: {ensemble.name}")

    # --- Create executor -----------------------------------------------------
    print("\n  Setting up execution engine...")
    if args.broker == "paper":
        executor = PaperExecutor(
            config=config.trading,
            initial_capital=args.capital,
        )
        print(f"  Mode: PAPER (simulated) | Capital: ${args.capital:,.2f}")
    else:
        executor = LiveExecutor(config=config.trading)
        print("  Mode: LIVE (Alpaca)")
        print("  Connecting to broker...")
        try:
            executor.connect()
        except Exception as exc:
            print(f"  ERROR: Could not connect to Alpaca: {exc}", file=sys.stderr)
            return 1

    # --- Create risk manager and position sizer ------------------------------
    risk_manager = RiskManager(config.trading)
    position_sizer = PositionSizer(config.trading)

    # --- Build SelfTrainer ---------------------------------------------------
    print("\n  Initialising SelfTrainer...")
    self_trainer = SelfTrainer(
        config=config,
        models=models,
        trainers=trainers,
        feature_pipeline=pipeline,
        ensemble=ensemble,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        executor=executor,
    )

    # Start the self-trainer (loads saved state if available, registers
    # shutdown hooks for Ctrl+C and SIGTERM)
    self_trainer.start()

    # --- Main loop -----------------------------------------------------------
    try:
        if args.broker == "paper":
            run_paper_simulation(
                self_trainer=self_trainer,
                data=data,
                config=config,
                interval=args.interval,
                speed=args.speed,
            )
        else:
            run_live_loop(
                self_trainer=self_trainer,
                symbols=symbols,
                interval=args.interval,
                config=config,
            )
    except KeyboardInterrupt:
        print("\n  Ctrl+C received. Shutting down gracefully...")
    except Exception as exc:
        print(f"\n  FATAL ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # --- Final summary -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Session Summary")
    print(f"{'=' * 60}")
    print(f"  Bars processed:  {self_trainer.bar_counter}")
    print(f"  State saved to:  {config.state_dir}")

    if args.broker == "paper" and hasattr(executor, "get_portfolio_state"):
        state = executor.get_portfolio_state()
        print(f"  Final equity:    ${state.get('equity', 0):,.2f}")
        print(f"  Cash remaining:  ${state.get('cash', 0):,.2f}")
        print(f"  Open positions:  {len(state.get('positions', {}))}")

    if args.broker == "live":
        try:
            executor.disconnect()
            print("  Disconnected from broker.")
        except Exception:
            pass

    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
