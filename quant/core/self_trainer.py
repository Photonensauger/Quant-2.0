"""Self-training orchestration loop.

The :class:`SelfTrainer` is the main run-loop of the system.  It receives
market bars one at a time, transforms features, obtains model predictions,
generates ensemble signals, runs risk checks, executes approved trades,
and periodically retrains models online when performance degrades or a
regime change is detected.

State is checkpointed periodically and on shutdown so the system can
resume from the exact point it left off.
"""

from __future__ import annotations

import atexit
import copy
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from loguru import logger

from quant.config.settings import SystemConfig, TrainingConfig
from quant.core.state_manager import SystemStateManager, STATE_VERSION


class SelfTrainer:
    """Main orchestration loop with online self-training.

    Parameters
    ----------
    config : SystemConfig
        Top-level system configuration.
    models : dict[str, Any]
        Mapping of ``model_name -> model_instance``.  Each model must
        support ``.state_dict()``, ``.load_state_dict()``, and be callable
        (forward pass) with a batched tensor.
    trainers : dict[str, Any]
        Mapping of ``model_name -> Trainer`` instance.  Used for online
        retraining.
    feature_pipeline : FeaturePipeline
        Fitted feature engineering pipeline.
    ensemble : EnsembleStrategy
        Ensemble strategy that aggregates model signals.
    risk_manager : RiskManager
        Portfolio-level risk manager.
    position_sizer : PositionSizer
        Position sizing calculator.
    executor : Any
        Execution backend (e.g. ``PaperExecutor`` or ``LiveExecutor``).
        Must expose ``execute(signal, qty, ...) -> dict``,
        ``close_all_positions() -> list``, and ``get_portfolio_state() -> dict``.
    data_buffer : list[dict] | None
        Optional pre-populated buffer of recent bars for retraining.
    """

    def __init__(
        self,
        config: SystemConfig,
        models: dict[str, Any],
        trainers: dict[str, Any],
        feature_pipeline: Any,
        ensemble: Any,
        risk_manager: Any,
        position_sizer: Any,
        executor: Any,
        data_buffer: list[dict] | None = None,
    ) -> None:
        self.config = config
        self.training_config: TrainingConfig = config.training
        self.models = models
        self.trainers = trainers
        self.feature_pipeline = feature_pipeline
        self.ensemble = ensemble
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.executor = executor

        # State manager for checkpointing
        self._state_manager = SystemStateManager(config.state_dir)

        # Runtime counters
        self.bar_counter: int = 0
        self._running: bool = False
        self._shutdown_requested: bool = False

        # Rolling performance tracking
        self._equity_history: list[float] = []
        self._returns_history: list[float] = []
        self._trade_history: list[dict[str, Any]] = []
        self._rolling_sharpe: float = 0.0

        # Training tracking
        self._last_retrain_bar: int = 0
        self._retrain_count: int = 0
        self._samples_since_retrain: int = 0

        # Data buffer for retraining (recent bars)
        self._data_buffer: list[dict] = data_buffer or []

        # Per-model rolling Sharpe for ensemble weight updates
        self._model_rolling_sharpes: dict[str, float] = {
            name: 0.0 for name in self.models
        }

        # Latest changepoint score (set during feature transform)
        self._latest_cp_score: float = 0.0

        logger.info(
            "SelfTrainer initialised | models={} | checkpoint_interval={} | "
            "retrain_interval={}",
            list(self.models.keys()),
            self.training_config.checkpoint_interval,
            self.training_config.retrain_interval,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register shutdown hooks, load or initialise state, enter main loop.

        This method registers ``atexit`` and signal handlers for SIGINT
        and SIGTERM to ensure state is saved on shutdown.  It then loads
        the most recent checkpoint (if available) and sets the ``_running``
        flag so that :meth:`process_bar` can be called in a loop.
        """
        # Register shutdown hooks
        atexit.register(self._shutdown)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("Shutdown hooks registered (atexit, SIGINT, SIGTERM)")

        # Load existing state or start fresh
        state = self._state_manager.load()
        if state is not None:
            self.load_full_state(state)
            logger.info(
                "Resumed from checkpoint at bar_counter={}",
                self.bar_counter,
            )
        else:
            logger.info("Starting fresh -- no previous state found")

        self._running = True
        self._shutdown_requested = False
        logger.info("SelfTrainer started | bar_counter={}", self.bar_counter)

    def process_bar(self, data: Any) -> dict[str, Any]:
        """Process a single market bar through the full pipeline.

        This is the core per-bar logic:

        1. Increment bar counter.
        2. Transform features via the feature pipeline.
        3. Obtain model predictions.
        4. Generate ensemble strategy signals.
        5. Run risk checks on proposed trades.
        6. Execute approved signals.
        7. Update rolling performance metrics.
        8. Check retrain triggers and retrain if needed.
        9. Update ensemble weights.
        10. Checkpoint if the interval has been reached.

        Parameters
        ----------
        data : Any
            A single bar of market data.  Typically a ``pd.DataFrame``
            with one or more rows (including lookback) containing OHLCV
            and feature columns.

        Returns
        -------
        dict
            Summary of what happened this bar, including signals produced,
            trades executed, and whether retraining occurred.
        """
        if not self._running:
            raise RuntimeError(
                "SelfTrainer is not running. Call start() first."
            )

        if self._shutdown_requested:
            logger.warning("Shutdown requested -- skipping bar processing")
            return {"skipped": True, "reason": "shutdown_requested"}

        result: dict[str, Any] = {
            "bar_counter": 0,
            "signals": [],
            "trades": [],
            "retrained": False,
            "checkpointed": False,
            "risk_violations": [],
        }

        # --- 1. Increment bar counter ------------------------------------
        self.bar_counter += 1
        self._samples_since_retrain += 1
        result["bar_counter"] = self.bar_counter

        # Buffer raw data for potential retraining
        self._buffer_bar(data)

        # --- 2. Feature pipeline transform --------------------------------
        try:
            features, targets = self.feature_pipeline.transform(data)
        except Exception as exc:
            logger.error(
                "Feature pipeline failed at bar {}: {}", self.bar_counter, exc
            )
            return result

        # Extract changepoint score if available
        self._update_cp_score(data)

        if features.shape[0] == 0:
            logger.debug("No valid feature rows at bar {}", self.bar_counter)
            return result

        # --- 3. Model predictions -----------------------------------------
        predictions: dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            try:
                pred = self._predict(model, features)
                predictions[name] = pred
            except Exception as exc:
                logger.error(
                    "Model '{}' prediction failed at bar {}: {}",
                    name,
                    self.bar_counter,
                    exc,
                )

        if not predictions:
            logger.debug("No predictions at bar {}", self.bar_counter)
            return result

        # --- 4. Strategy signals ------------------------------------------
        try:
            # Pass the average prediction to the ensemble (individual
            # strategies may use their own prediction internally)
            avg_pred = np.mean(list(predictions.values()), axis=0)
            signals = self.ensemble.generate_signals(data, avg_pred)
            result["signals"] = signals
        except Exception as exc:
            logger.error(
                "Ensemble signal generation failed at bar {}: {}",
                self.bar_counter,
                exc,
            )
            signals = []

        # --- 5. Risk check ------------------------------------------------
        portfolio_state = self._get_portfolio_state()
        risk_ok, violations = self.risk_manager.check_all(portfolio_state)
        result["risk_violations"] = violations

        # --- 6. Execute if approved ---------------------------------------
        if signals and risk_ok:
            for sig in signals:
                try:
                    trade = self._execute_signal(sig, data, portfolio_state)
                    if trade is not None:
                        result["trades"].append(trade)
                        self._trade_history.append(trade)
                except Exception as exc:
                    logger.error(
                        "Execution failed for signal {}: {}",
                        sig,
                        exc,
                    )
        elif not risk_ok:
            logger.warning(
                "Bar {} | Risk violations -- skipping execution: {}",
                self.bar_counter,
                violations,
            )

        # --- 7. Update rolling performance --------------------------------
        self._update_performance(portfolio_state)

        # --- 8. Check retrain triggers ------------------------------------
        if self._check_retrain():
            result["retrained"] = True
            for model_name in list(self.models.keys()):
                try:
                    self._retrain_model(model_name)
                except Exception as exc:
                    logger.error(
                        "Retrain failed for model '{}': {}", model_name, exc
                    )

        # --- 9. Update ensemble weights -----------------------------------
        try:
            self.ensemble.update_weights(self._model_rolling_sharpes)
        except Exception as exc:
            logger.error("Ensemble weight update failed: {}", exc)

        # --- 10. Checkpoint if interval reached ---------------------------
        if self.bar_counter % self.training_config.checkpoint_interval == 0:
            try:
                full_state = self.get_full_state()
                self._state_manager.save(full_state)
                result["checkpointed"] = True
                logger.info(
                    "Checkpoint saved at bar {}", self.bar_counter
                )
            except Exception as exc:
                logger.error("Checkpoint failed at bar {}: {}", self.bar_counter, exc)

        logger.debug(
            "Bar {} processed | signals={} trades={} retrained={} "
            "sharpe={:.3f}",
            self.bar_counter,
            len(result["signals"]),
            len(result["trades"]),
            result["retrained"],
            self._rolling_sharpe,
        )

        return result

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def get_full_state(self) -> dict[str, Any]:
        """Collect the complete system state from all components.

        Returns
        -------
        dict
            Full state dictionary matching the canonical schema.
        """
        # Model states
        model_states: dict[str, dict[str, Any]] = {}
        for name, model in self.models.items():
            trainer = self.trainers.get(name)
            model_state: dict[str, Any] = {
                "state_dict": model.state_dict(),
                "config": {},
                "optimizer_state": None,
                "epoch": 0,
                "best_val_loss": float("inf"),
            }
            if trainer is not None:
                trainer_state = trainer.get_state()
                model_state["state_dict"] = trainer_state.get(
                    "model_state_dict", model.state_dict()
                )
                model_state["config"] = trainer_state.get("model_config", {})
                model_state["optimizer_state"] = trainer_state.get(
                    "optimizer_state_dict"
                )
                model_state["epoch"] = trainer_state.get("epoch", 0)
                model_state["best_val_loss"] = trainer_state.get(
                    "best_val_loss", float("inf")
                )
            model_states[name] = model_state

        # Feature pipeline state
        pipeline_state = self.feature_pipeline.get_state()
        feature_pipeline_state = {
            "feature_names": pipeline_state.get("feature_names", []),
            "dropped_features": pipeline_state.get("dropped_features", []),
            "rolling_means": (
                pipeline_state.get("rolling_mean") or {}
            ),
            "rolling_stds": (
                pipeline_state.get("rolling_std") or {}
            ),
            "rolling_counts": {},
        }

        # BOCPD state
        bocpd_state = pipeline_state.get("cpd_state", {})

        # Ensemble state
        ensemble_raw = self.ensemble.get_state()
        ensemble_state = {
            "weights": ensemble_raw.get("weights", {}),
            "rolling_sharpes": dict(self._model_rolling_sharpes),
        }

        # Performance state
        performance_state = {
            "equity_history": list(self._equity_history),
            "returns_history": list(self._returns_history),
            "trade_history": list(self._trade_history),
            "rolling_sharpe": self._rolling_sharpe,
        }

        # Training state
        training_state = {
            "last_retrain_bar": self._last_retrain_bar,
            "retrain_count": self._retrain_count,
            "samples_since_retrain": self._samples_since_retrain,
        }

        return {
            "version": STATE_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bar_counter": self.bar_counter,
            "models": model_states,
            "feature_pipeline": feature_pipeline_state,
            "bocpd": bocpd_state,
            "ensemble": ensemble_state,
            "performance": performance_state,
            "training": training_state,
        }

    def load_full_state(self, state: dict[str, Any]) -> None:
        """Distribute a saved state dictionary to all components.

        Parameters
        ----------
        state : dict
            State dictionary produced by :meth:`get_full_state`.
        """
        version = state.get("version", "unknown")
        if version != STATE_VERSION:
            logger.warning(
                "State version mismatch: expected {}, got {}",
                STATE_VERSION,
                version,
            )

        # Restore bar counter
        self.bar_counter = state.get("bar_counter", 0)

        # Restore model states
        model_states = state.get("models", {})
        for name, mstate in model_states.items():
            if name in self.models:
                try:
                    self.models[name].load_state_dict(mstate["state_dict"])
                    logger.debug("Restored model state for '{}'", name)
                except Exception as exc:
                    logger.error(
                        "Failed to restore model '{}': {}", name, exc
                    )

            if name in self.trainers:
                try:
                    trainer_state = {
                        "model_state_dict": mstate["state_dict"],
                        "optimizer_state_dict": mstate.get("optimizer_state"),
                        "scheduler_state_dict": {},
                        "epoch": mstate.get("epoch", 0),
                        "best_val_loss": mstate.get(
                            "best_val_loss", float("inf")
                        ),
                        "model_config": mstate.get("config", {}),
                    }
                    # Only restore optimizer if state exists
                    if mstate.get("optimizer_state") is not None:
                        self.trainers[name].load_state(trainer_state)
                    else:
                        # At minimum restore model weights and epoch
                        self.trainers[name].epoch = mstate.get("epoch", 0)
                        self.trainers[name].best_val_loss = mstate.get(
                            "best_val_loss", float("inf")
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to fully restore trainer '{}': {}", name, exc
                    )

        # Restore feature pipeline state
        fp_state = state.get("feature_pipeline", {})
        try:
            pipeline_restore = {
                "feature_names": fp_state.get("feature_names", []),
                "dropped_features": fp_state.get("dropped_features", []),
                "rolling_mean": fp_state.get("rolling_means"),
                "rolling_std": fp_state.get("rolling_stds"),
                "fitted": bool(fp_state.get("feature_names")),
                "cpd_state": state.get("bocpd", {}),
            }
            self.feature_pipeline.load_state(pipeline_restore)
            logger.debug("Restored feature pipeline state")
        except Exception as exc:
            logger.error("Failed to restore feature pipeline: {}", exc)

        # Restore ensemble state
        ens_state = state.get("ensemble", {})
        try:
            saved_weights = ens_state.get("weights", {})
            if saved_weights:
                self.ensemble.load_state({"weights": saved_weights})
            self._model_rolling_sharpes = ens_state.get(
                "rolling_sharpes", {name: 0.0 for name in self.models}
            )
            logger.debug("Restored ensemble state")
        except Exception as exc:
            logger.error("Failed to restore ensemble state: {}", exc)

        # Restore performance state
        perf = state.get("performance", {})
        self._equity_history = perf.get("equity_history", [])
        self._returns_history = perf.get("returns_history", [])
        self._trade_history = perf.get("trade_history", [])
        self._rolling_sharpe = perf.get("rolling_sharpe", 0.0)

        # Restore training state
        train = state.get("training", {})
        self._last_retrain_bar = train.get("last_retrain_bar", 0)
        self._retrain_count = train.get("retrain_count", 0)
        self._samples_since_retrain = train.get("samples_since_retrain", 0)

        logger.info(
            "Full state loaded | bar_counter={} | models={} | "
            "sharpe={:.3f} | retrain_count={}",
            self.bar_counter,
            list(model_states.keys()),
            self._rolling_sharpe,
            self._retrain_count,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGINT / SIGTERM by flagging shutdown."""
        sig_name = signal.Signals(signum).name
        logger.warning("Received {} -- initiating graceful shutdown", sig_name)
        self._shutdown_requested = True
        self._shutdown()

    def _shutdown(self) -> None:
        """Close all positions (if live), save full state, log completion.

        Safe to call multiple times -- subsequent calls are no-ops.
        """
        if not self._running:
            return

        self._running = False
        logger.info("Shutting down SelfTrainer (bar_counter={})", self.bar_counter)

        # Close all positions via the executor
        try:
            if hasattr(self.executor, "close_all_positions"):
                closed = self.executor.close_all_positions()
                logger.info("Closed {} positions on shutdown", len(closed) if closed else 0)
        except Exception as exc:
            logger.error("Failed to close positions on shutdown: {}", exc)

        # Save full state
        try:
            full_state = self.get_full_state()
            self._state_manager.save(full_state)
            logger.info("State saved")
        except Exception as exc:
            logger.error("Failed to save state on shutdown: {}", exc)

    # ------------------------------------------------------------------
    # Retrain logic
    # ------------------------------------------------------------------

    def _check_retrain(self) -> bool:
        """Check all four retrain triggers and return True if any fires.

        Triggers
        --------
        1. **Interval**: ``bar_counter - last_retrain_bar >= retrain_interval``
        2. **Min samples**: ``samples_since_retrain >= retrain_min_samples``
           (must be met in conjunction with another trigger)
        3. **Sharpe degradation**: ``rolling_sharpe < sharpe_threshold``
        4. **Changepoint**: ``cp_score > cp_score_threshold``
        """
        cfg = self.training_config

        # Minimum samples guard -- retraining without enough data is wasteful
        has_min_samples = self._samples_since_retrain >= cfg.retrain_min_samples

        if not has_min_samples:
            return False

        # Trigger 1: interval
        interval_trigger = (
            self.bar_counter - self._last_retrain_bar >= cfg.retrain_interval
        )

        # Trigger 3: Sharpe degradation
        sharpe_trigger = (
            len(self._returns_history) >= cfg.retrain_min_samples
            and self._rolling_sharpe < cfg.sharpe_threshold
        )

        # Trigger 4: changepoint score
        cp_trigger = self._latest_cp_score > cfg.cp_score_threshold

        should_retrain = interval_trigger or sharpe_trigger or cp_trigger

        if should_retrain:
            reasons = []
            if interval_trigger:
                reasons.append(
                    f"interval ({self.bar_counter - self._last_retrain_bar}"
                    f" >= {cfg.retrain_interval})"
                )
            if sharpe_trigger:
                reasons.append(
                    f"sharpe ({self._rolling_sharpe:.3f}"
                    f" < {cfg.sharpe_threshold})"
                )
            if cp_trigger:
                reasons.append(
                    f"changepoint ({self._latest_cp_score:.3f}"
                    f" > {cfg.cp_score_threshold})"
                )
            logger.info(
                "Retrain triggered at bar {} | reasons: {}",
                self.bar_counter,
                ", ".join(reasons),
            )

        return should_retrain

    def _retrain_model(self, model_name: str) -> None:
        """Clone a model, retrain on recent data, accept if improved.

        Steps
        -----
        1. Deep-copy the current model (safety net).
        2. Set learning rate to ``current_lr * retrain_lr_factor``.
        3. Train on the last N buffered bars.
        4. Validate on a hold-out split.
        5. Accept the new weights only if validation loss improved.
        6. Update training counters.

        Parameters
        ----------
        model_name : str
            Key into ``self.models`` and ``self.trainers``.
        """
        model = self.models.get(model_name)
        trainer = self.trainers.get(model_name)

        if model is None or trainer is None:
            logger.warning(
                "Cannot retrain '{}': model or trainer not found", model_name
            )
            return

        if len(self._data_buffer) < self.training_config.retrain_min_samples:
            logger.debug(
                "Insufficient buffered data for retraining '{}' ({} < {})",
                model_name,
                len(self._data_buffer),
                self.training_config.retrain_min_samples,
            )
            return

        logger.info(
            "Retraining model '{}' | buffer_size={} | lr_factor={}",
            model_name,
            len(self._data_buffer),
            self.training_config.retrain_lr_factor,
        )

        # 1. Clone current model state as a safety net
        original_state_dict = copy.deepcopy(model.state_dict())
        original_best_val_loss = trainer.best_val_loss

        # 2. Build train/val loaders from the data buffer
        try:
            train_loader, val_loader = self._build_retrain_loaders()
        except Exception as exc:
            logger.error(
                "Failed to build retrain loaders for '{}': {}", model_name, exc
            )
            return

        if train_loader is None:
            logger.warning("No training data available for '{}'", model_name)
            return

        # 3. Retrain with reduced learning rate
        try:
            retrain_result = trainer.continue_training(
                loader=train_loader,
                epochs=self.training_config.retrain_epochs,
                lr_factor=self.training_config.retrain_lr_factor,
                val_loader=val_loader,
            )
        except Exception as exc:
            logger.error("Retrain loop failed for '{}': {}", model_name, exc)
            # Restore original weights
            model.load_state_dict(original_state_dict)
            return

        # 4. Accept or reject
        improved = retrain_result.get("improved", False)
        if improved:
            logger.info(
                "Retrain '{}' ACCEPTED | val_loss: {:.6f} -> {:.6f}",
                model_name,
                retrain_result.get("val_loss_before", float("inf")),
                retrain_result.get("val_loss_after", float("inf")),
            )
        else:
            logger.info(
                "Retrain '{}' REJECTED (no improvement) | restoring original weights",
                model_name,
            )
            model.load_state_dict(original_state_dict)
            trainer.best_val_loss = original_best_val_loss

        # 5. Update counters
        self._last_retrain_bar = self.bar_counter
        self._retrain_count += 1
        self._samples_since_retrain = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict(self, model: Any, features: np.ndarray) -> np.ndarray:
        """Run a forward pass through a model and return numpy predictions.

        Parameters
        ----------
        model : nn.Module
            A model that accepts ``[batch, seq_len, n_features]`` tensors.
        features : np.ndarray
            Feature array of shape ``[T, n_features]``.

        Returns
        -------
        np.ndarray
            Model predictions.
        """
        model.eval()
        device = self.config.device
        seq_len = self.config.model.seq_len

        # If features are shorter than seq_len, pad with zeros at the front
        if features.shape[0] < seq_len:
            pad_len = seq_len - features.shape[0]
            padding = np.zeros((pad_len, features.shape[1]))
            features = np.concatenate([padding, features], axis=0)

        # Take the last seq_len rows
        seq = features[-seq_len:]
        # Shape: [1, seq_len, n_features]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        pred = model(x)
        return pred.cpu().numpy().squeeze()

    def _execute_signal(
        self,
        sig: Any,
        data: Any,
        portfolio_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Size and execute a single signal through the executor.

        Returns a trade summary dict or None if the signal was not
        executed.
        """
        # Compute position size
        equity = portfolio_state.get("equity", 0.0)
        if equity <= 0:
            logger.warning("Cannot execute: equity is zero or negative")
            return None

        # Extract current price from data
        try:
            if hasattr(data, "iloc"):
                price = float(data["close"].iloc[-1])
            elif isinstance(data, dict):
                price = float(data.get("close", 0.0))
            else:
                logger.warning("Cannot extract price from data type: {}", type(data))
                return None
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("Failed to extract price: {}", exc)
            return None

        if price <= 0:
            return None

        signal_dict = {
            "direction": sig.direction,
            "confidence": sig.confidence,
            "symbol": sig.symbol,
        }

        qty = self.position_sizer.calculate(
            capital=equity,
            price=price,
            signal=signal_dict,
            market_data=data,
        )

        if abs(qty) < 1e-10:
            logger.debug("Position size too small for signal: {}", sig.symbol)
            return None

        # Execute through the executor
        try:
            trade = self.executor.execute(signal=sig, qty=qty)
            return trade
        except Exception as exc:
            logger.error("Executor raised exception: {}", exc)
            return None

    def _get_portfolio_state(self) -> dict[str, Any]:
        """Retrieve the current portfolio state from the executor."""
        try:
            if hasattr(self.executor, "get_portfolio_state"):
                return self.executor.get_portfolio_state()
        except Exception as exc:
            logger.error("Failed to get portfolio state: {}", exc)

        # Return minimal defaults
        return {
            "positions": {},
            "cash": 0.0,
            "equity": 0.0,
            "equity_curve": self._equity_history,
            "daily_pnl": 0.0,
            "returns_history": np.array(self._returns_history),
        }

    def _update_performance(self, portfolio_state: dict[str, Any]) -> None:
        """Update rolling performance metrics after processing a bar."""
        equity = portfolio_state.get("equity", 0.0)
        self._equity_history.append(equity)

        # Compute bar return
        if len(self._equity_history) >= 2:
            prev = self._equity_history[-2]
            if prev > 0:
                bar_return = (equity - prev) / prev
            else:
                bar_return = 0.0
        else:
            bar_return = 0.0
        self._returns_history.append(bar_return)

        # Rolling Sharpe ratio (annualised, assuming 252 trading days)
        self._rolling_sharpe = self._compute_rolling_sharpe(
            self._returns_history, window=252
        )

        # Per-model Sharpe tracking (simplified: use system-level Sharpe
        # weighted by each model's recent contribution; in a full system
        # you'd track per-model attribution)
        for name in self._model_rolling_sharpes:
            self._model_rolling_sharpes[name] = self._rolling_sharpe

    @staticmethod
    def _compute_rolling_sharpe(
        returns: list[float],
        window: int = 252,
        annualise: bool = True,
    ) -> float:
        """Compute rolling Sharpe ratio from a list of returns."""
        if len(returns) < 2:
            return 0.0

        r = np.array(returns[-window:])
        mean_r = np.mean(r)
        std_r = np.std(r, ddof=1)

        if std_r < 1e-12:
            return 0.0

        sharpe = mean_r / std_r
        if annualise:
            sharpe *= np.sqrt(252)

        return float(sharpe)

    def _buffer_bar(self, data: Any) -> None:
        """Append the latest bar to the retraining data buffer.

        The buffer is capped at ``retrain_interval * 2`` bars to avoid
        unbounded memory growth.
        """
        max_buffer = self.training_config.retrain_interval * 2
        try:
            if hasattr(data, "to_dict"):
                # DataFrame -- store the last row as a dict
                bar = data.iloc[-1].to_dict() if hasattr(data, "iloc") else data.to_dict()
            elif isinstance(data, dict):
                bar = data
            else:
                bar = {"raw": data}
            self._data_buffer.append(bar)
            if len(self._data_buffer) > max_buffer:
                self._data_buffer = self._data_buffer[-max_buffer:]
        except Exception as exc:
            logger.debug("Failed to buffer bar: {}", exc)

    def _update_cp_score(self, data: Any) -> None:
        """Extract the latest changepoint score from data if available."""
        try:
            if hasattr(data, "columns") and "cp_score" in data.columns:
                self._latest_cp_score = float(data["cp_score"].iloc[-1])
            elif isinstance(data, dict) and "cp_score" in data:
                self._latest_cp_score = float(data["cp_score"])
        except (IndexError, KeyError, TypeError):
            pass  # Keep the previous value

    def _build_retrain_loaders(self) -> tuple[Any, Any]:
        """Build PyTorch DataLoaders from the data buffer for retraining.

        Returns ``(train_loader, val_loader)``.  Uses an 80/20 split on
        the buffered bars.

        Returns
        -------
        tuple
            ``(train_loader, val_loader)`` or ``(None, None)`` on failure.
        """
        import pandas as pd
        from torch.utils.data import DataLoader, TensorDataset

        if len(self._data_buffer) < self.training_config.retrain_min_samples:
            return None, None

        # Reconstruct a DataFrame from buffered dicts
        try:
            df = pd.DataFrame(self._data_buffer)
        except Exception as exc:
            logger.error("Failed to build DataFrame from buffer: {}", exc)
            return None, None

        # Run through feature pipeline
        try:
            features, targets = self.feature_pipeline.transform(df)
        except Exception as exc:
            logger.error("Feature transform failed during retrain prep: {}", exc)
            return None, None

        if features.shape[0] < self.training_config.retrain_min_samples:
            return None, None

        # Create sequences for the model
        seq_len = self.config.model.seq_len
        X_seqs, Y_seqs = [], []
        for i in range(seq_len, len(features)):
            X_seqs.append(features[i - seq_len : i])
            Y_seqs.append(targets[i] if i < len(targets) else 0.0)

        if len(X_seqs) == 0:
            return None, None

        X = torch.tensor(np.array(X_seqs), dtype=torch.float32)
        Y = torch.tensor(np.array(Y_seqs), dtype=torch.float32)

        # 80/20 split
        split = int(0.8 * len(X))
        if split < 1 or split >= len(X):
            return None, None

        train_ds = TensorDataset(X[:split], Y[:split])
        val_ds = TensorDataset(X[split:], Y[split:])

        batch_size = self.training_config.batch_size
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
