#!/usr/bin/env python3
"""CLI for running backtests with trained models.

Usage
-----
# Backtest ensemble strategy on AAPL
python scripts/run_backtest.py --strategy ensemble --assets AAPL --start 2025-01-01 --end 2025-02-01

# Backtest a single ML strategy
python scripts/run_backtest.py --strategy ml --assets AAPL,MSFT --start 2025-01-01 --end 2025-02-01

# Custom output directory and initial capital
python scripts/run_backtest.py --strategy ensemble --assets AAPL --start 2025-01-01 --end 2025-02-01 \\
    --capital 50000 --output reports/my_backtest.html
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant.backtest.engine import BacktestEngine, BacktestResult
from quant.config.settings import SystemConfig
from quant.data.provider import YFinanceProvider
from quant.data.storage import DataStorage
from quant.features.pipeline import FeaturePipeline
from quant.models import (
    AdversarialRegimeModel,
    AttentionLSTM,
    CausalDiscoveryTransformer,
    CombinedLoss,
    DecoderTransformer,
    EntropicPortfolioDiffusion,
    HamiltonianNeuralODE,
    ITransformer,
    MomentumTransformer,
    SchrodingerTransformer,
    TopologicalAttentionNetwork,
    Trainer,
)
from quant.strategies.base import BaseStrategy
from quant.strategies.ensemble import EnsembleStrategy
from quant.strategies.ml_signal import MLSignalStrategy
from quant.utils.logging import setup_logging
from quant.utils.viz import PerformanceVisualizer


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type] = {
    "transformer": DecoderTransformer,
    "itransformer": ITransformer,
    "lstm": AttentionLSTM,
    "momentum": MomentumTransformer,
    # Zug 37 Models
    "causal": CausalDiscoveryTransformer,
    "schrodinger": SchrodingerTransformer,
    "topological": TopologicalAttentionNetwork,
    "hamiltonian": HamiltonianNeuralODE,
    "diffusion": EntropicPortfolioDiffusion,
    "adversarial": AdversarialRegimeModel,
}


# ---------------------------------------------------------------------------
# Model loading
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


class ModelWrapper:
    """Wraps a PyTorch model to expose a ``predict(features)`` method.

    The BacktestEngine expects ``model.predict(features) -> np.ndarray``.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, seq_len: int) -> None:
        self.model = model
        self.device = device
        self.seq_len = seq_len

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference and return predictions as numpy array."""
        self.model.eval()

        # Pad if features are shorter than seq_len
        if features.shape[0] < self.seq_len:
            pad_len = self.seq_len - features.shape[0]
            padding = np.zeros((pad_len, features.shape[1]))
            features = np.concatenate([padding, features], axis=0)

        # Take the last seq_len rows
        seq = features[-self.seq_len:]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        pred = self.model(x)
        return pred.cpu().numpy().squeeze()


# ---------------------------------------------------------------------------
# Strategy construction
# ---------------------------------------------------------------------------

def build_strategy(
    strategy_name: str,
    config: SystemConfig,
    models: dict[str, torch.nn.Module],
) -> BaseStrategy:
    """Build the specified strategy.

    For "ensemble", creates an MLSignalStrategy per loaded model and wraps
    them in an EnsembleStrategy.  For "ml", uses the first available model
    with a single MLSignalStrategy.
    """
    if strategy_name == "ensemble":
        if not models:
            print("  WARNING: No models loaded. Creating a single MLSignalStrategy.", file=sys.stderr)
            return MLSignalStrategy(config.trading, name="ml_fallback")

        strategies: dict[str, BaseStrategy] = {}
        for name in models:
            strategies[name] = MLSignalStrategy(config.trading, name=f"ml_{name}")

        # Load accuracy-based weights from training metrics if available
        initial_weights = None
        metrics_path = config.checkpoint_dir / "training_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    training_metrics = json.load(f)
                # Filter to models actually used in this ensemble
                relevant = {k: v for k, v in training_metrics.items() if k in models}
                if relevant:
                    total_acc = sum(relevant.values())
                    if total_acc > 0:
                        initial_weights = {k: v / total_acc for k, v in relevant.items()}
                        print(f"  Loaded accuracy-based weights: {initial_weights}")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  WARNING: Could not load training metrics: {exc}", file=sys.stderr)

        return EnsembleStrategy(
            config=config.trading,
            strategies=strategies,
            name="ensemble",
            initial_weights=initial_weights,
        )

    elif strategy_name == "ml":
        model_name = next(iter(models), "default")
        return MLSignalStrategy(config.trading, name=f"ml_{model_name}")

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a backtest with trained models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_backtest.py --strategy ensemble --assets AAPL --start 2025-01-01 --end 2025-02-01\n"
            "  python scripts/run_backtest.py --strategy ml --assets AAPL,MSFT --start 2025-01-01 --end 2025-02-01\n"
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
        "--assets",
        type=str,
        default="AAPL",
        help="Comma-separated list of symbols to backtest (default: AAPL).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=["1m", "5m", "15m", "1h", "1d"],
        help="Bar interval (default: 5m).",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="Backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial capital (default: 100000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/dashboard.html",
        help="Output path for the HTML dashboard (default: reports/dashboard.html).",
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

    config.backtest.initial_capital = args.capital

    symbols = [s.strip() for s in args.assets.split(",") if s.strip()]
    if not symbols:
        print("ERROR: No symbols provided.", file=sys.stderr)
        return 1

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if end_date <= start_date:
        print("ERROR: End date must be after start date.", file=sys.stderr)
        return 1

    # Determine which models to load
    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_names = ["transformer", "itransformer", "lstm"]

    print(f"\n{'=' * 60}")
    print(f"  Backtesting Engine")
    print(f"{'=' * 60}")
    print(f"  Strategy:  {args.strategy}")
    print(f"  Assets:    {', '.join(symbols)}")
    print(f"  Interval:  {args.interval}")
    print(f"  Period:    {args.start} -> {args.end}")
    print(f"  Capital:   ${args.capital:,.2f}")
    print(f"  Models:    {', '.join(model_names)}")
    print(f"  Output:    {args.output}")
    print(f"{'=' * 60}")

    # --- Load data -----------------------------------------------------------
    print("\n  Loading market data...")
    storage = DataStorage(config.data)
    provider = YFinanceProvider(config.data)

    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df = storage.get_or_fetch(provider, symbol, args.interval, start_date, end_date)
        if df.empty:
            print(f"  WARNING: No data for {symbol}. Skipping.", file=sys.stderr)
            continue
        data[symbol] = df
        print(f"  {symbol}: {len(df):,} bars ({df.index.min()} -> {df.index.max()})")

    if not data:
        print("\nERROR: No data available. Run scripts/download_data.py first.", file=sys.stderr)
        return 1

    # --- Feature pipeline (reuse training state if available) ----------------
    print("\n  Running feature pipeline...")
    pipeline = FeaturePipeline(config.features)

    pipeline_state_path = config.checkpoint_dir / "feature_pipeline_state.json"
    if pipeline_state_path.exists():
        with open(pipeline_state_path) as f:
            pipeline.load_state(json.load(f))
        ref_symbol = list(data.keys())[0]
        features, targets = pipeline.transform(data[ref_symbol])
        print(f"  Loaded pipeline state from {pipeline_state_path}")
    else:
        print("  WARNING: No saved pipeline state found. Fitting on backtest data (may cause dimension mismatch).")
        ref_symbol = list(data.keys())[0]
        features, targets = pipeline.fit_transform(data[ref_symbol])

    n_features = pipeline.n_features
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
        print("  WARNING: No trained models found. Backtest will run without model predictions.")
        print("  (Signals will not be generated. Consider running scripts/train_model.py first.)")

    # Wrap models for the backtest engine
    wrapped_model = None
    if models:
        # For the backtest engine, use a single wrapped model.
        # For ensemble, the strategy handles multiple models internally.
        first_model_name = next(iter(models))
        wrapped_model = ModelWrapper(
            models[first_model_name],
            config.device,
            config.model.seq_len,
        )

    # --- Build strategy ------------------------------------------------------
    print("\n  Building strategy...")
    strategy = build_strategy(args.strategy, config, models)
    print(f"  Strategy: {strategy.name}")

    # --- Run backtest --------------------------------------------------------
    print("\n  Running backtest...")
    engine = BacktestEngine(
        config=config,
        checkpoint_dir=config.checkpoint_dir / "backtest",
    )

    t0 = time.time()
    result: BacktestResult = engine.run(
        data=data,
        model=wrapped_model,
        feature_pipeline=pipeline,
        strategy=strategy,
    )
    elapsed = time.time() - t0

    # --- Print key metrics ---------------------------------------------------
    metrics = result.metrics

    print(f"\n{'=' * 60}")
    print(f"  Backtest Results")
    print(f"{'=' * 60}")
    print(f"  Duration:             {elapsed:.1f}s")
    print(f"  Initial Capital:      ${metrics.get('initial_capital', 0):,.2f}")
    print(f"  Final Equity:         ${metrics.get('final_equity', 0):,.2f}")
    print(f"  Total Return:         {metrics.get('total_return', 0):.2%}")
    print(f"  Annualized Return:    {metrics.get('annualized_return', 0):.2%}")
    print(f"  Volatility:           {metrics.get('volatility', 0):.2%}")
    print(f"  Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio:        {metrics.get('sortino_ratio', 0):.2f}")
    print(f"  Calmar Ratio:         {metrics.get('calmar_ratio', 0):.2f}")
    print(f"  Max Drawdown:         {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Max DD Duration:      {metrics.get('max_drawdown_duration', 0):.0f} bars")
    print(f"  Total Trades:         {int(metrics.get('total_trades', 0))}")
    print(f"  Win Rate:             {metrics.get('win_rate', 0):.2%}")
    print(f"  Profit Factor:        {metrics.get('profit_factor', 0):.2f}")
    print(f"  Avg Win:              ${metrics.get('avg_win', 0):,.2f}")
    print(f"  Avg Loss:             ${metrics.get('avg_loss', 0):,.2f}")
    print(f"  Expectancy:           ${metrics.get('expectancy', 0):,.2f}")
    print(f"{'=' * 60}")

    # --- Export JSON for dashboard -------------------------------------------
    json_dir = Path(config.data.cache_dir).parent / "backtest"
    json_dir.mkdir(parents=True, exist_ok=True)
    assets_label = args.assets.replace(",", "_") if hasattr(args, "assets") else "multi"
    json_name = f"{args.strategy}_{assets_label}_{args.interval}.json"
    json_export = {
        "equity_curve": result.equity_curve,
        "returns": result.returns,
        "trade_log": result.trade_log,
        "metrics": result.metrics,
        "timestamps": [str(t) for t in result.timestamps],
    }
    json_path = json_dir / json_name
    with open(json_path, "w") as jf:
        json.dump(json_export, jf, indent=2, default=str)
    print(f"\n  JSON result saved to: {json_path}")

    # --- Generate dashboard --------------------------------------------------
    print(f"\n  Generating performance dashboard...")
    viz = PerformanceVisualizer()

    # Convert BacktestResult to the dict format expected by the visualizer
    viz_data: dict = {
        "equity_curve": result.equity_curve,
        "returns": result.returns,
        "trades": result.trade_log,
        "metrics": result.metrics,
    }

    # Attach timestamps if available
    if result.timestamps:
        equity_series = pd.Series(
            result.equity_curve,
            index=pd.DatetimeIndex(result.timestamps[: len(result.equity_curve)]),
        )
        viz_data["equity_curve"] = equity_series

        returns_series = pd.Series(
            result.returns,
            index=pd.DatetimeIndex(result.timestamps[: len(result.returns)]),
        )
        viz_data["returns"] = returns_series

    try:
        output_path = viz.generate_dashboard(viz_data, args.output)
        print(f"  Dashboard saved to: {output_path}")
    except Exception as exc:
        print(f"  WARNING: Dashboard generation failed: {exc}", file=sys.stderr)

    print(f"\n{'=' * 60}")
    print(f"  Backtest complete.")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
