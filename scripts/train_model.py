#!/usr/bin/env python3
"""CLI for training forecasting models on downloaded market data.

Usage
-----
# Train all models on AAPL 5m data for 50 epochs
python scripts/train_model.py --model all --assets AAPL --interval 5m --epochs 50

# Train a specific model
python scripts/train_model.py --model transformer --assets AAPL --interval 5m --epochs 100

# Train on multiple assets (data is concatenated)
python scripts/train_model.py --model lstm --assets AAPL,MSFT --interval 5m --epochs 50

# Customize walk-forward splits
python scripts/train_model.py --model itransformer --assets AAPL --epochs 30 --splits 3
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
from torch.utils.data import DataLoader, Subset

from quant.config.settings import SystemConfig
from quant.data.dataset import TimeSeriesDataset, WalkForwardSplitter
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
from quant.utils.logging import setup_logging


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

ALL_MODEL_NAMES = list(MODEL_REGISTRY.keys())


def build_model(name: str, config: SystemConfig):
    """Instantiate a model by name and return (model, criterion)."""
    model_cls = MODEL_REGISTRY[name]
    model = model_cls(config.model)
    criterion = CombinedLoss(mse_weight=1.0, directional_weight=0.5)
    return model, criterion


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    symbols: list[str],
    interval: str,
    days: int,
    config: SystemConfig,
) -> pd.DataFrame:
    """Load and concatenate OHLCV data for all symbols from the cache.

    Falls back to fetching from the provider if the cache is empty.
    """
    storage = DataStorage(config.data)
    provider = YFinanceProvider(config.data)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        df = storage.get_or_fetch(provider, symbol, interval, start, end)
        if df.empty:
            print(f"  WARNING: No data for {symbol}. Skipping.", file=sys.stderr)
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No data available for any of the requested symbols. "
            "Run scripts/download_data.py first."
        )

    # For multi-asset training, concatenate along the time axis.
    # Each asset's data is independent; the model treats it as additional
    # training samples during walk-forward splitting.
    combined = pd.concat(frames, axis=0).sort_index()
    # Remove duplicate timestamps that may occur from concatenation
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_single_model(
    model_name: str,
    features: np.ndarray,
    targets: np.ndarray,
    config: SystemConfig,
    max_epochs: int,
    n_splits: int,
    batch_size: int,
) -> dict:
    """Train a single model with walk-forward cross-validation.

    Returns a summary dict with training results.
    """
    print(f"\n{'─' * 50}")
    print(f"  Training: {model_name}")
    print(f"{'─' * 50}")

    model, criterion = build_model(model_name, config)
    device = config.device

    print(f"  Model parameters: {model.count_parameters():,}")
    print(f"  Device: {device}")
    print(f"  Features shape: {features.shape}")
    print(f"  Targets shape: {targets.shape}")

    # Create full dataset
    dataset = TimeSeriesDataset(
        features=features,
        target=targets,
        seq_len=config.model.seq_len,
        forecast_horizon=config.model.forecast_horizon,
    )

    n_samples = len(dataset)
    if n_samples == 0:
        print("  ERROR: Not enough data for the configured seq_len and forecast_horizon.")
        return {"model": model_name, "status": "FAILED", "reason": "insufficient_data"}

    print(f"  Dataset samples: {n_samples:,}")

    # Walk-forward splitting
    splitter = WalkForwardSplitter(
        n_splits=n_splits,
        test_ratio=config.training.test_ratio,
        gap_size=config.training.gap_size,
    )
    splits = splitter.split(n_samples)

    if not splits:
        print("  ERROR: No valid walk-forward splits could be generated.")
        return {"model": model_name, "status": "FAILED", "reason": "no_valid_splits"}

    print(f"  Walk-forward splits: {len(splits)}")

    # Train on the last (largest) split for the final model
    train_range, test_range = splits[-1]
    train_indices = list(train_range)
    test_indices = list(test_range)

    print(f"  Train samples: {len(train_indices):,}")
    print(f"  Test samples:  {len(test_indices):,}")

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        model_config=config.model,
        training_config=config.training,
        device=device,
    )

    # Train
    t0 = time.time()
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs,
        patience=config.training.early_stopping_patience,
    )
    elapsed = time.time() - t0

    # Save checkpoint
    checkpoint_dir = config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_latest.pt"
    trainer.save_checkpoint(checkpoint_path)

    # Also save the model itself for easy loading
    model_path = checkpoint_dir / f"{model_name}_model.pt"
    model.save(
        model_path,
        optimizer=trainer.optimizer,
        epoch=trainer.epoch,
        best_val_loss=trainer.best_val_loss,
    )

    # Print results
    best_epoch = result.get("best_epoch", "?")
    best_val_loss = result.get("best_val_loss", float("inf"))
    dir_acc = result.get("directional_accuracy", 0.0)

    print(f"\n  Results for {model_name}:")
    print(f"    Best epoch:              {best_epoch}")
    print(f"    Best validation loss:    {best_val_loss:.6f}")
    print(f"    Directional accuracy:    {dir_acc:.2%}")
    print(f"    Training time:           {elapsed:.1f}s")
    print(f"    Checkpoint saved:        {checkpoint_path}")
    print(f"    Model saved:             {model_path}")

    return {
        "model": model_name,
        "status": "OK",
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "directional_accuracy": dir_acc,
        "elapsed": elapsed,
        "checkpoint_path": str(checkpoint_path),
        "model_path": str(model_path),
        "n_params": model.count_parameters(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train forecasting models on market data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/train_model.py --model all --assets AAPL --interval 5m --epochs 50\n"
            "  python scripts/train_model.py --model transformer --assets AAPL,MSFT --epochs 100\n"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODEL_NAMES + ["all"],
        help=(
            "Model to train. Options: "
            + ", ".join(ALL_MODEL_NAMES)
            + ", all. Use 'all' to train every model."
        ),
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
        help="Bar interval of the training data (default: 5m).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of calendar days of data to use for training (default: 30).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs per model (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of walk-forward cross-validation splits (default: 5).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config default).",
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

    # Override training config from CLI args
    config.training.max_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.n_splits = args.splits
    if args.lr is not None:
        config.training.learning_rate = args.lr

    symbols = [s.strip() for s in args.assets.split(",") if s.strip()]
    if not symbols:
        print("ERROR: No symbols provided.", file=sys.stderr)
        return 1

    # Determine which models to train
    if args.model == "all":
        model_names = ALL_MODEL_NAMES
    else:
        model_names = [args.model]

    print(f"\n{'=' * 60}")
    print(f"  Model Training Pipeline")
    print(f"{'=' * 60}")
    print(f"  Models:    {', '.join(model_names)}")
    print(f"  Assets:    {', '.join(symbols)}")
    print(f"  Interval:  {args.interval}")
    print(f"  Days:      {args.days}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  Splits:    {args.splits}")
    print(f"  Device:    {config.device}")
    print(f"  Ckpt dir:  {config.checkpoint_dir}")
    print(f"{'=' * 60}")

    # --- Load data -----------------------------------------------------------
    print("\n  Loading data...")
    try:
        raw_data = load_data(symbols, args.interval, args.days, config)
    except RuntimeError as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"  Loaded {len(raw_data):,} bars")

    # --- Feature pipeline ----------------------------------------------------
    print("  Running feature pipeline...")
    pipeline = FeaturePipeline(config.features)
    features, targets = pipeline.fit_transform(raw_data)

    # Save pipeline state so backtest can use the same feature set
    pipeline_state_path = config.checkpoint_dir / "feature_pipeline_state.json"
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(pipeline_state_path, "w") as f:
        json.dump(pipeline.get_state(), f)
    print(f"  Pipeline state saved to {pipeline_state_path}")

    if features.shape[0] == 0:
        print("  ERROR: Feature pipeline produced zero valid rows.", file=sys.stderr)
        return 1

    n_features = features.shape[1]
    config.model.n_features = n_features

    print(f"  Features: {features.shape[0]:,} rows x {n_features} features")
    print(f"  Targets:  {targets.shape[0]:,} rows")

    # --- Train each model ----------------------------------------------------
    all_results: list[dict] = []
    t_total_start = time.time()

    for model_name in model_names:
        try:
            result = train_single_model(
                model_name=model_name,
                features=features,
                targets=targets,
                config=config,
                max_epochs=args.epochs,
                n_splits=args.splits,
                batch_size=args.batch_size,
            )
            all_results.append(result)
        except Exception as exc:
            print(f"\n  ERROR training {model_name}: {exc}", file=sys.stderr)
            all_results.append({
                "model": model_name,
                "status": "FAILED",
                "reason": str(exc),
            })

    total_elapsed = time.time() - t_total_start

    # --- Save training metrics for accuracy-based ensemble weighting ---------
    training_metrics = {
        r["model"]: r["directional_accuracy"]
        for r in all_results
        if r.get("status") == "OK"
    }
    if training_metrics:
        metrics_path = config.checkpoint_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(training_metrics, f, indent=2)
        print(f"\n  Training metrics saved to {metrics_path}")

    # --- Summary -------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Training Summary")
    print(f"{'=' * 60}")
    print(f"  {'Model':<18s} {'Status':<8s} {'Val Loss':<12s} {'Dir Acc':<10s} {'Time':<10s} {'Params':<12s}")
    print(f"  {'─' * 18} {'─' * 8} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 12}")

    for r in all_results:
        name = r.get("model", "?")
        status = r.get("status", "?")
        if status == "OK":
            val_loss = f"{r['best_val_loss']:.6f}"
            dir_acc = f"{r['directional_accuracy']:.2%}"
            elapsed = f"{r['elapsed']:.1f}s"
            params = f"{r['n_params']:,}"
        else:
            val_loss = "---"
            dir_acc = "---"
            elapsed = "---"
            params = "---"
        print(f"  {name:<18s} {status:<8s} {val_loss:<12s} {dir_acc:<10s} {elapsed:<10s} {params:<12s}")

    ok_count = sum(1 for r in all_results if r.get("status") == "OK")
    failed_count = sum(1 for r in all_results if r.get("status") == "FAILED")

    print(f"\n  Total time: {total_elapsed:.1f}s | OK: {ok_count} | Failed: {failed_count}")
    print(f"  Checkpoints: {config.checkpoint_dir}")
    print(f"{'=' * 60}\n")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
