# Quant 2.0 -- System Architecture

> Self-learning Multi-Asset Quantitative Trading System
> Version 2.0.0

---

## Table of Contents

1. [System Layer Architecture Overview](#1-system-layer-architecture-overview)
2. [Component Network Diagram](#2-component-network-diagram)
3. [Data Flow Diagrams](#3-data-flow-diagrams)
4. [Interface Specifications](#4-interface-specifications)
5. [Dependency Graph & Build Order](#5-dependency-graph--build-order)
6. [Sequence Diagrams](#6-sequence-diagrams)
7. [State Persistence Schema](#7-state-persistence-schema)
8. [Error Handling & Edge Cases](#8-error-handling--edge-cases)
9. [Project File Structure](#9-project-file-structure)
10. [Implementation Order & Acceptance Criteria](#10-implementation-order--acceptance-criteria)
11. [End-to-End Verification](#11-end-to-end-verification)

---

## 1. System Layer Architecture Overview

The system is organized into eleven distinct layers, each with a single
responsibility.  Higher layers depend only on lower layers; no circular
dependencies exist.

```
+============================================================================+
|                          SCRIPTS LAYER  (CLI entry points)                  |
|   download_data.py  |  train_model.py  |  run_backtest.py  |  live_trade   |
+============================================================================+
          |                    |                   |                |
          v                    v                   v                v
+============================================================================+
|                         CORE LAYER  (orchestration)                        |
|              SelfTrainer  (main loop + online retraining)                   |
|              SystemStateManager  (atomic checkpoints)                       |
+============================================================================+
          |                    |                   |                |
          v                    v                   v                v
+--------------------+-----------------------+-------------------------+
| EXECUTION LAYER    | BACKTEST LAYER        | STRATEGIES LAYER        |
| PaperExecutor      | BacktestEngine        | BaseStrategy / Signal   |
| LiveExecutor       | BacktestResult        | MLSignalStrategy        |
|   (Alpaca API)     | compute_metrics()     | EnsembleStrategy        |
+--------------------+-----------------------+-------------------------+
          |                    |                   |
          v                    v                   v
+--------------------+-----------------------+-------------------------+
| PORTFOLIO LAYER                                                     |
| PortfolioOptimizer (mean-var, min-var, risk-parity, equal-weight)   |
| PositionSizer      (Kelly, volatility-adjusted, fixed-fraction)     |
| RiskManager        (VaR, drawdown, daily loss, concentration)       |
| Position           (dataclass tracking open trades)                 |
+---------------------------------------------------------------------+
          |
          v
+---------------------------------------------------------------------+
| MODELS LAYER                                                        |
| BaseModel           -- abstract with save/load checkpoints          |
| DecoderTransformer  -- causal self-attention, sinusoidal PE         |
| ITransformer        -- inverted: features as tokens (ICLR 2024)    |
| AttentionLSTM       -- multi-layer LSTM + multi-head attention      |
| MomentumTransformer -- outputs position in [-1,+1] via tanh        |
| PPOAgent            -- Actor-Critic for portfolio allocation        |
| Trainer             -- walk-forward + online retraining             |
| DifferentiableSharpeRatio / DirectionalLoss / CombinedLoss          |
+---------------------------------------------------------------------+
          |
          v
+---------------------------------------------------------------------+
| FEATURES LAYER                                                      |
| compute_technical_features()  -- ~41 indicators                     |
| compute_time_features()       -- 10 sin/cos calendar features       |
| BayesianChangePointDetector   -- BOCPD with NIG prior               |
| FeaturePipeline               -- orchestrator: build + norm + clean |
+---------------------------------------------------------------------+
          |
          v
+---------------------------------------------------------------------+
| DATA LAYER                                                          |
| DataProvider (ABC)  -> YFinanceProvider, CryptoProvider, ForexProv.  |
| DataStorage         -- Parquet cache with partial-fetch merge       |
| TimeSeriesDataset   -- sliding-window PyTorch Dataset               |
| WalkForwardSplitter -- expanding-window CV with gap                 |
+---------------------------------------------------------------------+
          |
          v
+---------------------------------------------------------------------+
| CONFIG LAYER                                                        |
| DataConfig | FeatureConfig | ModelConfig | TrainingConfig            |
| TradingConfig | BacktestConfig | SystemConfig (aggregator)          |
+---------------------------------------------------------------------+
          |
          v
+---------------------------------------------------------------------+
| UTILS LAYER                                                         |
| setup_logging()          -- loguru console + rotating file           |
| PerformanceVisualizer    -- Plotly HTML dashboard (6 panels)         |
+---------------------------------------------------------------------+
```

### Layer Summary Table

| # | Layer       | Package              | Key Classes / Functions                                    |
|---|-------------|----------------------|------------------------------------------------------------|
| 1 | Config      | `quant.config`       | `SystemConfig`, `DataConfig`, `FeatureConfig`, ...         |
| 2 | Data        | `quant.data`         | `DataProvider` (ABC), `DataStorage`, `TimeSeriesDataset`   |
| 3 | Features    | `quant.features`     | `FeaturePipeline`, `BayesianChangePointDetector`           |
| 4 | Models      | `quant.models`       | `DecoderTransformer`, `ITransformer`, `AttentionLSTM`, ... |
| 5 | Strategies  | `quant.strategies`   | `Signal`, `MLSignalStrategy`, `EnsembleStrategy`           |
| 6 | Portfolio   | `quant.portfolio`    | `PortfolioOptimizer`, `PositionSizer`, `RiskManager`       |
| 7 | Backtest    | `quant.backtest`     | `BacktestEngine`, `compute_metrics()`                      |
| 8 | Execution   | `quant.execution`    | `PaperExecutor`, `LiveExecutor`                            |
| 9 | Core        | `quant.core`         | `SelfTrainer`, `SystemStateManager`                        |
|10 | Utils       | `quant.utils`        | `setup_logging()`, `PerformanceVisualizer`                 |
|11 | Scripts     | `scripts/`           | `download_data`, `train_model`, `run_backtest`, `live_trade` |

---

## 2. Component Network Diagram

```
                     +-------------------+
                     |   CLI / Scripts   |
                     +-------------------+
                       |   |   |   |
            +----------+   |   |   +----------+
            |              |   |              |
            v              v   v              v
  +-------------+   +-----------+   +-------------+
  | download_   |   | train_    |   | run_        |
  | data.py     |   | model.py  |   | backtest.py |
  +------+------+   +-----+-----+   +------+------+
         |                |                 |
         |                v                 v
         |          +-----+------+   +------+------+     +-------------+
         |          | Trainer    |   | Backtest    |     | live_trade  |
         |          | (Trainer)  |   | Engine      |     |   .py       |
         |          +-----+------+   +------+------+     +------+------+
         |                |                 |                    |
         |                v                 v                    v
         |        +-------+--------+  +-----+------+     +------+------+
         |        | Models         |  | Strategy   |     | SelfTrainer |
         |        | (Transformer,  |  | (Ensemble, |     | (Core Loop) |
         |        |  LSTM, PPO...) |  |  ML)       |     +------+------+
         |        +-------+--------+  +-----+------+            |
         |                |                 |                    |
         v                v                 v                    v
  +------+------+  +------+------+  +------+------+     +------+------+
  | Data Layer  |  | Feature     |  | Portfolio   |     | Execution   |
  | (Provider,  |  | Pipeline    |  | (Optimizer, |     | (Paper /    |
  |  Storage)   |  | (Tech,Time, |  |  Risk, Pos) |     |  Live)      |
  +------+------+  |  BOCPD)     |  +-------------+     +------+------+
         |         +------+------+                              |
         |                |                                     |
         v                v                                     v
  +------+---------+------+------+                       +------+------+
  | Config Layer   | Utils Layer |                       | Alpaca API  |
  | (SystemConfig) | (Logging,   |                       | (optional)  |
  +----------------+  Viz)       |                       +-------------+
                    +-------------+
```

---

## 3. Data Flow Diagrams

### 3.1 Raw Data to Trade (Live Path)

```
  Market Data Source (YFinance / ccxt / Alpaca)
         |
         v
  +------+------+
  | DataProvider | -- fetch_ohlcv(symbol, interval, start, end)
  +------+------+     returns: DataFrame[open, high, low, close, volume]
         |              DatetimeIndex (UTC), float64, validated
         v
  +------+------+
  | DataStorage  | -- Parquet cache: get_or_fetch()
  +------+------+     partial fetch & merge if cache incomplete
         |
         v
  +------+----------+
  | FeaturePipeline  | -- fit_transform() or transform()
  |  1. technical    |    ~41 OHLCV indicators
  |  2. time_feats   |    10 sin/cos calendar features
  |  3. BOCPD        |    cp_score, cp_severity
  |  4. corr filter  |    drop |corr| > 0.95
  |  5. rolling z    |    (x - mu_roll) / (std_roll + eps)
  +------+----------+
         |  output: np.ndarray [T', n_features], np.ndarray [T'] (targets)
         v
  +------+------+
  | Model(s)     | -- forward(x: [B, seq_len, n_features])
  |  Transformer |    returns: [B, forecast_horizon] (log-return preds)
  |  iTransformer|
  |  LSTM        |
  |  Momentum    |
  +------+------+
         |
         v
  +------+----------+
  | Strategy Layer   | -- generate_signals(data, predictions)
  |  MLSignalStrategy|    confidence scaling, cooldown, confirmation
  |  EnsembleStrategy|    Sharpe-weighted vote across models
  +------+----------+
         |  output: list[Signal(timestamp, symbol, direction, confidence, ...)]
         v
  +------+----------+
  | Portfolio Layer  |
  |  PositionSizer   | -- Kelly / vol-adjusted / fixed-fraction
  |  RiskManager     | -- drawdown, VaR, daily loss, concentration checks
  +------+----------+
         |  output: signed qty, risk approval
         v
  +------+------+
  | Executor     | -- submit_order(symbol, side, qty, price)
  |  Paper / Live|    fill record + position tracking
  +------+------+
         |
         v
     Trade Executed
```

### 3.2 Backtest Loop

```
  +------------------------------------------------------------------+
  |  BacktestEngine.run(data, model, pipeline, strategy)             |
  |                                                                  |
  |  for bar_idx in range(n_bars):                                   |
  |    |                                                             |
  |    +-- 1. mark_to_market(positions, data[bar_idx].close)         |
  |    |                                                             |
  |    +-- 2. risk_check_positions()  (stop-loss / take-profit)      |
  |    |       if triggered -> close_position() + trade_log          |
  |    |                                                             |
  |    +-- 3. risk_check_portfolio()  (drawdown, VaR, daily loss)    |
  |    |       if breached -> set _risk_breached flag                |
  |    |                                                             |
  |    +-- 4. if bar_idx >= seq_len:                                 |
  |    |       window = data[bar_idx - seq_len + 1 : bar_idx + 1]   |
  |    |       features = pipeline.transform(window)                 |
  |    |       predictions = model.predict(features)                 |
  |    |       signals = strategy.generate_signals(window, preds)    |
  |    |                                                             |
  |    +-- 5. for each signal with direction != 0:                   |
  |    |       if !risk_breached && cooldown_ok && confidence_ok:    |
  |    |         qty = position_sizer.calculate(...)                 |
  |    |         apply slippage + commission                         |
  |    |         open position, deduct cash                          |
  |    |                                                             |
  |    +-- 6. equity = cash + margin_held + unrealised_pnl           |
  |    |       equity_curve.append(equity)                           |
  |    |       returns.append(bar_return)                            |
  |    |                                                             |
  |    +-- 7. checkpoint every N bars (JSON snapshot)                |
  |                                                                  |
  |  close_all_positions()  (end of backtest)                        |
  |  metrics = compute_metrics(equity_curve, trade_log, capital)     |
  |  return BacktestResult(equity_curve, returns, trade_log, ...)    |
  +------------------------------------------------------------------+
```

### 3.3 Self-Training Cycle

```
  +-----------------------------------------------------------------+
  |  SelfTrainer.process_bar(data)                                  |
  |                                                                 |
  |  1. bar_counter++                                               |
  |  2. buffer_bar(data)  -- ring buffer, max = 2 * retrain_interval|
  |  3. features, targets = pipeline.transform(data)                |
  |  4. for each model: predictions[name] = model(features)         |
  |  5. signals = ensemble.generate_signals(data, avg_prediction)   |
  |  6. risk_ok, violations = risk_manager.check_all(portfolio)     |
  |  7. if signals && risk_ok: execute via executor                 |
  |  8. update_performance(equity, returns, rolling Sharpe)         |
  |  9. if check_retrain():                                         |
  |       for each model:                                           |
  |         a. clone model (safety net)                             |
  |         b. build loaders from buffer (80/20 split)              |
  |         c. trainer.continue_training(lr * retrain_lr_factor)    |
  |         d. if val_loss_after < val_loss_before: ACCEPT          |
  |            else: REJECT (restore original weights)              |
  | 10. ensemble.update_weights(model_rolling_sharpes)              |
  | 11. if bar_counter % checkpoint_interval == 0:                  |
  |       state_manager.save(get_full_state())                      |
  |                                                                 |
  |  Retrain Triggers (all require min_samples guard):              |
  |    - interval: bars since last retrain >= retrain_interval      |
  |    - sharpe:   rolling_sharpe < sharpe_threshold                |
  |    - BOCPD:    cp_score > cp_score_threshold                    |
  +-----------------------------------------------------------------+
```

---

## 4. Interface Specifications

### 4.1 Config Layer Contract

```python
# All configs are frozen dataclasses (Python @dataclass).
# SystemConfig aggregates all sub-configs.

SystemConfig
  .data:     DataConfig       # cache_dir, intervals, retries, min_rows
  .features: FeatureConfig    # warmup, rolling_window, BOCPD params, horizon
  .model:    ModelConfig      # seq_len, d_model, n_heads, n_layers per arch
  .training: TrainingConfig   # lr, epochs, batch_size, walk-forward splits
  .trading:  TradingConfig    # position limits, risk limits, slippage, SL/TP
  .backtest: BacktestConfig   # initial_capital, slippage, commission, margin
  .device:   torch.device     # auto-detected: MPS > CUDA > CPU
  .checkpoint_dir: Path
  .state_dir: Path
  .log_level: str
```

### 4.2 Data Layer Contract

```python
class DataProvider(ABC):
    def fetch_ohlcv(symbol, interval, start, end) -> pd.DataFrame
    # Guarantees: DatetimeIndex (UTC), cols [open,high,low,close,volume], float64
    # high >= max(open,close), low <= min(open,close), no NaN in OHLC

class DataStorage:
    def get_or_fetch(provider, symbol, interval, start, end) -> pd.DataFrame
    # Cache-through: Parquet files keyed by (symbol, interval)
    # Partial-fetch: only downloads missing date ranges

class TimeSeriesDataset(Dataset):
    def __getitem__(idx) -> (x: [seq_len, n_features], y: [forecast_horizon])

class WalkForwardSplitter:
    def split(n_total) -> list[tuple[range, range]]
    # Guarantee: max(train_range) + gap_size < min(test_range)
```

### 4.3 Features Layer Contract

```python
class FeaturePipeline:
    def fit_transform(df: DataFrame) -> (np.ndarray [T', n_features], np.ndarray [T'])
    def transform(df: DataFrame) -> (np.ndarray [T', n_features], np.ndarray [T'])
    def get_state() -> dict    # JSON-serialisable
    def load_state(dict) -> None

class BayesianChangePointDetector:
    def update(x: float) -> (cp_score: float, severity: float)
    # cp_score in [0, 1], severity >= 0
    def get_state() -> dict
    def load_state(dict) -> None
```

### 4.4 Models Layer Contract

```python
class BaseModel(nn.Module, ABC):
    def forward(x: [batch, seq_len, n_features]) -> Tensor
    def save(path, optimizer=None, epoch=0, ...) -> Path
    @classmethod
    def load(path, device=None) -> (model, checkpoint_dict)
    def count_parameters() -> int

# Return-prediction models:
#   DecoderTransformer, ITransformer, AttentionLSTM
#   forward(x) -> [batch, forecast_horizon]

# Position-output model:
#   MomentumTransformer
#   forward(x) -> [batch, 1]  (values in [-1, +1])

# RL agent (not a BaseModel subclass):
class PPOAgent(nn.Module):
    def select_action(state) -> ActionResult(action, log_prob, value)
    def evaluate_actions(states, actions) -> EvalResult(log_probs, values, entropy)

class Trainer:
    def fit(train_loader, val_loader) -> dict
    def continue_training(loader, epochs, lr_factor, val_loader) -> dict
    def get_state() / load_state(dict) / save_checkpoint(path)
```

### 4.5 Strategies Layer Contract

```python
@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    direction: int     # -1, 0, +1
    confidence: float  # [0.0, 1.0]
    target_position: float  # [-1.0, 1.0]
    metadata: dict

class BaseStrategy(ABC):
    def generate_signals(data: DataFrame, predictions: ndarray | None) -> list[Signal]

class EnsembleStrategy(BaseStrategy):
    def update_weights(performance: dict[str, float]) -> None
    def get_state() / load_state(dict) / save_state(path)
```

### 4.6 Portfolio Layer Contract

```python
class PortfolioOptimizer:
    def optimize(returns: DataFrame, ...) -> ndarray  # weights summing to 1.0

class PositionSizer:
    def calculate(capital, price, signal, market_data) -> float  # signed qty
    # Guarantee: |qty * price| <= capital * max_position_pct

class RiskManager:
    def check_all(portfolio_state: dict) -> (bool, list[str])
    def check_stop_loss(position, current_price, atr) -> bool
    def check_take_profit(position, current_price, atr) -> bool
```

### 4.7 Backtest Layer Contract

```python
class BacktestEngine:
    def run(data: dict[str, DataFrame], model, pipeline, strategy) -> BacktestResult

@dataclass
class BacktestResult:
    equity_curve: list[float]
    returns: list[float]
    trade_log: list[dict]
    metrics: dict[str, float]
    positions_history: list[dict]
    timestamps: list[datetime]

def compute_metrics(equity_curve, trade_log, initial_capital) -> dict[str, float]
# Keys: total_return, annualized_return, volatility, sharpe_ratio,
#        sortino_ratio, calmar_ratio, max_drawdown, max_drawdown_duration,
#        total_trades, win_rate, profit_factor, avg_win, avg_loss, expectancy
```

### 4.8 Execution Layer Contract

```python
class PaperExecutor:
    def submit_order(symbol, side, qty, price) -> dict  # fill record
    def close_position(symbol, price=None) -> dict
    def close_all_positions(prices: dict) -> list[dict]
    def get_portfolio_state() -> dict
    def update_market_prices(prices: dict) -> None

class LiveExecutor:
    def connect() -> None    # with exponential-backoff retries
    def disconnect() -> None
    def submit_order(symbol, side, qty, order_type, ...) -> dict
    def close_position(symbol) -> dict
    def close_all_positions() -> list[dict]
    def get_portfolio_state() -> dict
    def get_order_status(order_id) -> dict
    def cancel_order(order_id) -> bool
    def cancel_all_orders() -> bool
```

### 4.9 Core Layer Contract

```python
class SystemStateManager:
    def save(state: dict) -> None     # atomic write + backup rotation (3 backups)
    def load() -> dict | None         # fallback chain: primary -> bak.1 -> bak.2 -> bak.3

class SelfTrainer:
    def start() -> None               # register shutdown hooks, load state
    def process_bar(data) -> dict      # single-bar pipeline execution
    def get_full_state() -> dict       # serialise all components
    def load_full_state(dict) -> None  # distribute state to all components
```

---

## 5. Dependency Graph & Build Order

Each layer depends only on layers listed to its right (or below).

```
Scripts -> Core -> Execution, Backtest, Strategies -> Portfolio
                                                        |
                                                        v
Backtest, Strategies -> Models -> Features -> Data -> Config
                                                        |
                                                        v
                                                      Utils
```

**Build order** (install/test from bottom up):

```
  1. Utils         (no internal deps)
  2. Config        (depends on: torch, dotenv)
  3. Data          (depends on: Config)
  4. Features      (depends on: Config, Data)
  5. Models        (depends on: Config)
  6. Strategies    (depends on: Config)
  7. Portfolio     (depends on: Config)
  8. Backtest      (depends on: Config, Features, Portfolio, Strategies)
  9. Execution     (depends on: Config, Portfolio)
 10. Core          (depends on: Config, Features, Models, Strategies, Portfolio, Execution)
 11. Scripts       (depends on: all layers)
```

---

## 6. Sequence Diagrams

### 6.1 Live Trading -- Single Bar

```
  User/Cron              SelfTrainer         FeaturePipeline      Model(s)
      |                      |                     |                  |
      |-- process_bar(data)->|                     |                  |
      |                      |-- transform(data)-->|                  |
      |                      |<-- (features, tgt)--|                  |
      |                      |                     |                  |
      |                      |-- forward(features)-------------------->|
      |                      |<-- predictions --------------------------|
      |                      |
      |                  EnsembleStrategy       RiskManager        Executor
      |                      |                     |                  |
      |                      |-- generate_signals->|                  |
      |                      |<-- signals ---------|                  |
      |                      |                     |                  |
      |                      |-- check_all()------>|                  |
      |                      |<-- (ok, violations)-|                  |
      |                      |                     |                  |
      |                      | [if ok && signals]  |                  |
      |                      |-- calculate(qty)--->|                  |
      |                      |<-- signed_qty ------|                  |
      |                      |                     |                  |
      |                      |-- submit_order(symbol, side, qty, price)->|
      |                      |<-- fill_record -----|                  |
      |                      |                     |                  |
      |                      |-- check_retrain()   |                  |
      |                      | [if triggered]      |                  |
      |                      |   clone model       |                  |
      |                      |   continue_training |                  |
      |                      |   accept/reject     |                  |
      |                      |                     |                  |
      |                      |-- update_weights()->|                  |
      |                      |                     |                  |
      |                      |-- checkpoint() [if interval reached]   |
      |<-- result dict ------|                     |                  |
```

### 6.2 State Recovery on Restart

```
  Process Start          SelfTrainer         StateManager
      |                      |                     |
      |-- start() --------->|                     |
      |                      |-- load() --------->|
      |                      |                     |-- try primary state.pt
      |                      |                     |   [if corrupt]
      |                      |                     |-- try state.pt.bak.1
      |                      |                     |   [if corrupt]
      |                      |                     |-- try state.pt.bak.2
      |                      |                     |   [if corrupt]
      |                      |                     |-- try state.pt.bak.3
      |                      |<-- state dict ------|
      |                      |                     |
      |                      |-- load_full_state() |
      |                      |   restore models    |
      |                      |   restore trainers  |
      |                      |   restore pipeline  |
      |                      |   restore ensemble  |
      |                      |   restore counters  |
      |                      |                     |
      |                      |-- register atexit   |
      |                      |-- register SIGINT   |
      |                      |-- register SIGTERM  |
      |                      |                     |
      |                      |-- _running = True   |
      |<-- ready ------------|                     |
      |                      |                     |
  [Normal operation: process_bar() loop]           |
      |                      |                     |
  SIGINT received            |                     |
      |                      |-- _shutdown()       |
      |                      |   close_all_positions()
      |                      |   get_full_state()  |
      |                      |-- save(state) ----->|
      |                      |                     |-- rotate backups
      |                      |                     |-- atomic write (tmp + rename)
      |<-- exit -------------|                     |
```

---

## 7. State Persistence Schema

The system state is serialised as a PyTorch checkpoint (`torch.save`)
containing the following top-level structure:

```python
{
    "version": "2.0.0",                 # str -- schema version
    "timestamp": "2026-02-17T...",      # str -- ISO 8601 UTC timestamp
    "bar_counter": 12345,               # int -- total bars processed

    "models": {
        "<model_name>": {
            "state_dict": OrderedDict,  # model weights
            "config": dict,             # ModelConfig as dict
            "optimizer_state": dict,    # AdamW state
            "epoch": int,
            "best_val_loss": float,
        },
        # ... one entry per model
    },

    "feature_pipeline": {
        "feature_names": ["log_return", "sma20_dist", ...],
        "dropped_features": ["stoch_slow_d", ...],
        "rolling_means": {"feature_name": float, ...},
        "rolling_stds": {"feature_name": float, ...},
        "rolling_counts": {},
    },

    "bocpd": {
        "hazard": float,
        "alpha0": float, "beta0": float,
        "kappa0": float, "mu0": float,
        "mu": [float, ...],            # per-run-length
        "kappa": [float, ...],
        "alpha": [float, ...],
        "beta": [float, ...],
        "run_length_posterior": [float, ...],
        "t": int,
        "max_run_len": int,
    },

    "ensemble": {
        "weights": {"transformer": 0.35, "lstm": 0.25, ...},
        "rolling_sharpes": {"transformer": 1.2, ...},
    },

    "performance": {
        "equity_history": [float, ...],
        "returns_history": [float, ...],
        "trade_history": [dict, ...],
        "rolling_sharpe": float,
    },

    "training": {
        "last_retrain_bar": int,
        "retrain_count": int,
        "samples_since_retrain": int,
    },
}
```

### Backup Rotation

| File                       | Description           |
|----------------------------|-----------------------|
| `state/system_state.pt`   | Current (latest)      |
| `state/system_state.pt.bak.1` | Previous save     |
| `state/system_state.pt.bak.2` | Two saves ago     |
| `state/system_state.pt.bak.3` | Three saves ago   |

Writes are **atomic**: data is written to a temporary file in the same
directory, then `os.replace()` swaps it into position (atomic on POSIX).

---

## 8. Error Handling & Edge Cases

### 8.1 Data Layer

| Scenario | Handling |
|---|---|
| API returns empty DataFrame | Return `pd.DataFrame(columns=OHLCV_COLUMNS)`; log warning |
| API timeout / network error | Exponential backoff: `base * 2^attempt`, up to `max_retries` |
| Corrupt Parquet cache file | Delete file, log warning, re-fetch from provider |
| Duplicate timestamps in data | De-duplicate keeping first occurrence |
| NaN in OHLC columns | Drop affected rows; fill volume NaN with 0.0 |
| `high < max(open,close)` | Clip high upward: `high = clip(high, lower=max(open,close))` |

### 8.2 Feature Layer

| Scenario | Handling |
|---|---|
| All-NaN feature column | Drop column, log warning |
| Correlation > 0.95 between features | Drop the second column in each pair |
| Rolling std = 0 (constant feature) | Add epsilon (1e-8) to denominator |
| Pipeline not fitted before transform | Raise `RuntimeError` |
| Missing features in new data | Fill with NaN, log warning |

### 8.3 Model Layer

| Scenario | Handling |
|---|---|
| NaN / Inf in loss | Skip gradient step, increment NaN counter, log error |
| MPS / CUDA out of memory | Fallback to CPU; re-create optimizer, scheduler |
| Gradient explosion | `clip_grad_norm_(max_norm=1.0)` |
| No valid walk-forward splits | Return error status, skip model |
| Checkpoint file not found | Return None, continue without model |

### 8.4 Strategy Layer

| Scenario | Handling |
|---|---|
| Model predictions are None | Return empty signal list |
| Signal confidence below threshold | Suppress signal |
| Cooldown period not elapsed | Suppress signal |
| Direction not confirmed for N bars | Suppress signal |
| Sub-strategy raises exception | Log exception, return empty list for that strategy |
| Weighted vote magnitude < 0.1 | Consensus direction = 0 (flat), no signal |

### 8.5 Portfolio Layer

| Scenario | Handling |
|---|---|
| Negative Kelly fraction | Floor at 0.0 (never bet negative) |
| ATR = 0 | Fall back to fixed-fraction sizing |
| Position notional > max_position_pct | Cap quantity to stay within limit |
| Equity <= 0 | Return 0 quantity, log warning |
| Optimizer does not converge (SLSQP) | Use last iterate, log warning |
| All weights near zero after clip | Fall back to equal-weight (1/N) |

### 8.6 Risk Layer

| Scenario | Handling |
|---|---|
| Max drawdown breached (>15%) | Flag risk breach; refuse new trades |
| Daily loss limit breached (>3%) | Flag risk breach; refuse new trades |
| VaR(95%) breached (>5%) | Flag risk breach; refuse new trades |
| Position concentration breached (>10%) | Report violation per position |
| Insufficient data for VaR (<20 returns) | Skip VaR check |
| Stop-loss triggered | Close position immediately, log info |
| Take-profit triggered | Close position immediately, log info |

### 8.7 Execution Layer

| Scenario | Handling |
|---|---|
| Insufficient cash for buy | Reduce quantity to fit available cash |
| Zero or negative price/qty | Raise ValueError / return 0 |
| Alpaca connection failure | Exponential backoff retries; emergency stop after max_retries |
| Alpaca API key not set | Raise RuntimeError on connect() |
| Order notional exceeds max_order_size | Raise ValueError (reject before sending) |
| Live order submission failure | Log error, return failed status dict |

### 8.8 Core Layer

| Scenario | Handling |
|---|---|
| Process killed (SIGINT/SIGTERM) | Graceful shutdown: close positions, save state |
| State file corrupt | Fall back to .bak.1, .bak.2, .bak.3 in order |
| State version mismatch | Log warning, attempt to load anyway |
| process_bar() called before start() | Raise RuntimeError |
| Retrain rejected (no improvement) | Restore original model weights |
| Feature pipeline fails during retrain | Log error, skip retrain for this cycle |

---

## 9. Project File Structure

```
Quant 2.0/
|-- pyproject.toml                    # Build system, dependencies, pytest config
|-- requirements.txt                  # Pinned dependencies
|-- .env.example                      # Environment variable template
|-- .gitignore
|
|-- quant/                            # Main package
|   |-- __init__.py
|   |
|   |-- config/
|   |   |-- __init__.py
|   |   +-- settings.py               # SystemConfig + all sub-configs
|   |
|   |-- data/
|   |   |-- __init__.py               # Exports: DataProvider, Storage, Dataset, ...
|   |   |-- provider.py               # YFinance, Crypto (ccxt), Forex providers
|   |   |-- storage.py                # Parquet cache with partial-fetch merge
|   |   +-- dataset.py                # TimeSeriesDataset, WalkForwardSplitter
|   |
|   |-- features/
|   |   |-- __init__.py               # Exports: technical, time, BOCPD, pipeline
|   |   |-- technical.py              # ~41 technical indicators (pandas/numpy)
|   |   |-- time_features.py          # 10 sin/cos calendar encodings
|   |   |-- changepoint.py            # BOCPD with Normal-Inverse-Gamma prior
|   |   +-- pipeline.py               # FeaturePipeline orchestrator
|   |
|   |-- models/
|   |   |-- __init__.py               # Exports all models, losses, trainer
|   |   |-- base.py                   # BaseModel ABC with save/load
|   |   |-- transformer.py            # DecoderTransformer (causal self-attention)
|   |   |-- itransformer.py           # ITransformer (features as tokens)
|   |   |-- lstm.py                   # AttentionLSTM
|   |   |-- momentum_transformer.py   # MomentumTransformer (tanh output)
|   |   |-- rl_portfolio.py           # PPOAgent (Actor-Critic)
|   |   |-- losses.py                 # DifferentiableSharpeRatio, CombinedLoss
|   |   +-- trainer.py                # Trainer with walk-forward + online retrain
|   |
|   |-- strategies/
|   |   |-- __init__.py               # Exports: Signal, BaseStrategy, ML, Ensemble
|   |   |-- base.py                   # Signal dataclass + BaseStrategy ABC
|   |   |-- ml_signal.py              # MLSignalStrategy (predictions -> signals)
|   |   +-- ensemble.py               # EnsembleStrategy (Sharpe-weighted voting)
|   |
|   |-- portfolio/
|   |   |-- __init__.py               # Exports: Optimizer, Sizer, Risk, Position
|   |   |-- optimizer.py              # Mean-var, min-var, risk-parity, equal-weight
|   |   |-- position.py               # Position dataclass + PositionSizer
|   |   +-- risk.py                   # RiskManager (drawdown, VaR, daily, conc.)
|   |
|   |-- backtest/
|   |   |-- __init__.py
|   |   |-- engine.py                 # BacktestEngine + BacktestResult
|   |   +-- metrics.py                # compute_metrics() -- 17 performance metrics
|   |
|   |-- execution/
|   |   |-- __init__.py
|   |   |-- paper.py                  # PaperExecutor (simulated fills)
|   |   +-- live.py                   # LiveExecutor (Alpaca API)
|   |
|   |-- core/
|   |   |-- __init__.py
|   |   |-- self_trainer.py           # SelfTrainer main loop + online retraining
|   |   +-- state_manager.py          # SystemStateManager (atomic save + backups)
|   |
|   +-- utils/
|       |-- __init__.py
|       |-- logging.py                # loguru setup: console + rotating file
|       +-- viz.py                    # PerformanceVisualizer (Plotly dashboard)
|
|-- scripts/
|   |-- __init__.py
|   |-- download_data.py              # CLI: fetch + cache market data
|   |-- train_model.py                # CLI: walk-forward model training
|   |-- run_backtest.py               # CLI: run backtest + generate dashboard
|   +-- live_trade.py                 # CLI: paper/live trading with SelfTrainer
|
|-- tests/
|   |-- __init__.py
|   |-- conftest.py                   # Shared pytest fixtures
|   |-- test_config/
|   |   +-- test_settings.py
|   |-- test_data/
|   |   |-- test_provider.py
|   |   |-- test_storage.py
|   |   +-- test_dataset.py
|   |-- test_features/
|   |   |-- test_technical.py
|   |   |-- test_time_features.py
|   |   |-- test_changepoint.py
|   |   +-- test_pipeline.py
|   |-- test_models/
|   |-- test_strategies/
|   |-- test_portfolio/
|   |-- test_backtest/
|   |-- test_execution/
|   |-- test_core/
|   +-- test_utils/
|
|-- data/
|   +-- cache/                        # Parquet cache files (gitignored)
|
|-- docs/
|   |-- diagrams/
|   |-- ARCHITECTURE.md               # This document
|   +-- ALGORITHMS.md                 # Mathematical algorithms reference
|
|-- logs/                             # Rotating log files (gitignored)
|-- checkpoints/                      # Model checkpoints (gitignored)
+-- state/                            # System state files (gitignored)
```

---

## 10. Implementation Order & Acceptance Criteria

### Phase 1: Foundation

| Step | Component | Acceptance Criteria |
|------|-----------|---------------------|
| 1 | Project setup | `pyproject.toml` valid; `pip install -e .` succeeds; `.gitignore` covers venv, cache, logs, checkpoints |
| 2 | Config layer | `SystemConfig()` instantiates with all defaults; `get_device()` returns valid device; all sub-configs accessible |
| 3 | Utils layer | `setup_logging("DEBUG")` creates console + file handler; log files appear in `logs/` |

### Phase 2: Data Pipeline

| Step | Component | Acceptance Criteria |
|------|-----------|---------------------|
| 4 | Data providers | `YFinanceProvider.fetch_ohlcv("AAPL", "1d", ...)` returns valid DataFrame; OHLCV columns present; DatetimeIndex UTC |
| 5 | Data storage | `DataStorage.get_or_fetch()` creates Parquet file on first call; returns cached data on second call; partial fetch fills gaps |
| 6 | Dataset + splitter | `TimeSeriesDataset` returns `(x, y)` tuples of correct shape; `WalkForwardSplitter` produces non-overlapping train/test ranges with gap |

### Phase 3: Feature Engineering

| Step | Component | Acceptance Criteria |
|------|-----------|---------------------|
| 7 | Technical indicators | `compute_technical_features(df)` adds ~41 columns; no duplicate column names; first 50 rows may have NaN |
| 8 | Time features | `compute_time_features(df)` adds 10 sin/cos columns; no NaN; values in [-1, 1] |
| 9 | BOCPD | `BayesianChangePointDetector.update(x)` returns `(cp_score, severity)` with `cp_score` in [0, 1]; state round-trips via `get_state`/`load_state` |
| 10 | Feature pipeline | `FeaturePipeline.fit_transform(df)` returns `(features, targets)` with no NaN; correlation filter removes redundant features; rolling z-score normalisation applied |

### Phase 4: Models

| Step | Component | Acceptance Criteria |
|------|-----------|---------------------|
| 11 | Base model | `save()` writes checkpoint; `load()` restores identical weights |
| 12 | DecoderTransformer | `forward(x)` with `x.shape = [B, 60, 50]` returns `[B, 5]`; causal mask prevents future leakage |
| 13 | ITransformer | `forward(x)` returns `[B, 5]`; transpose inverts feature/time dimensions |
| 14 | AttentionLSTM | `forward(x)` returns `[B, 5]`; attention applied over LSTM hidden states |
| 15 | MomentumTransformer | `forward(x)` returns `[B, 1]` with values in [-1, +1] |
| 16 | PPOAgent | `select_action()` returns valid portfolio weights summing to 1; `evaluate_actions()` returns log_probs and entropy |
| 17 | Losses | `DifferentiableSharpeRatio` is differentiable; gradient flows through positions |
| 18 | Trainer | `fit()` trains with early stopping; `continue_training()` fine-tunes with reduced LR |

### Phase 5: Strategy & Portfolio

| Step | Component | Acceptance Criteria |
|------|-----------|---------------------|
| 19 | Signal + BaseStrategy | `Signal` validates direction in {-1, 0, 1}; confidence clamped to [0, 1] |
| 20 | MLSignalStrategy | Predictions -> signals with cooldown, confirmation, confidence gating |
| 21 | EnsembleStrategy | Weighted voting produces consensus; `update_weights()` shifts allocation toward better-performing models |
| 22 | PortfolioOptimizer | All four methods return weights summing to 1.0, all >= 0 |
| 23 | PositionSizer | Kelly, vol-adjusted, and fixed-fraction all respect `max_position_pct` cap |
| 24 | RiskManager | Correctly detects drawdown, VaR, daily loss, and concentration breaches |

### Phase 6: Backtest & Execution

| Step | Component | Acceptance Criteria |
|------|-----------|---------------------|
| 25 | BacktestEngine | `run()` processes all bars; equity curve has correct length; trades logged with entry/exit details |
| 26 | compute_metrics | All 17 metrics computed; Sharpe matches manual calculation on known data |
| 27 | PaperExecutor | Orders fill with slippage; positions tracked; cash updated; close_all works |
| 28 | LiveExecutor | Connects to Alpaca paper API (with valid keys); submit_order returns order_id |

### Phase 7: Core & Scripts

| Step | Component | Acceptance Criteria |
|------|-----------|---------------------|
| 29 | SystemStateManager | Atomic save creates file; load recovers from backup when primary is corrupt |
| 30 | SelfTrainer | `process_bar()` runs full pipeline; retrain triggers fire correctly; state round-trips |
| 31 | download_data.py | CLI downloads and caches data for multiple symbols |
| 32 | train_model.py | CLI trains model(s) and saves checkpoints |
| 33 | run_backtest.py | CLI runs backtest and generates HTML dashboard |
| 34 | live_trade.py | CLI runs paper simulation using SelfTrainer loop |

---

## 11. End-to-End Verification

### E2E Test 1: Data Download -> Training -> Backtest

```bash
# 1. Download data
python scripts/download_data.py --assets AAPL --interval 1d --days 365

# 2. Train all models
python scripts/train_model.py --model all --assets AAPL --interval 1d --epochs 20

# 3. Run backtest
python scripts/run_backtest.py --strategy ensemble --assets AAPL \
    --start 2025-06-01 --end 2025-12-31 --capital 100000

# Verify:
# - Checkpoints exist in checkpoints/
# - Dashboard HTML generated in reports/
# - Sharpe ratio, max drawdown, trade count are non-zero
```

### E2E Test 2: Paper Trading with State Recovery

```bash
# 1. Start paper trading
python scripts/live_trade.py --strategy ensemble --broker paper \
    --assets AAPL --interval 5m --speed 0

# 2. After processing ~200 bars, press Ctrl+C
#    Verify: state/system_state.pt exists

# 3. Restart
python scripts/live_trade.py --strategy ensemble --broker paper \
    --assets AAPL --interval 5m --speed 0

# Verify:
# - "Resumed from checkpoint at bar_counter=..." appears in logs
# - bar_counter continues from where it left off
# - Ensemble weights reflect saved state
```

### E2E Test 3: Retrain Trigger

```bash
# Configure aggressive retrain settings:
#   retrain_interval = 100
#   sharpe_threshold = 2.0  (will trigger quickly)
#   retrain_min_samples = 50

# Run paper simulation
python scripts/live_trade.py --strategy ensemble --broker paper \
    --assets AAPL --interval 5m --speed 0

# Verify in logs:
# - "Retrain triggered at bar N | reasons: sharpe (...)" appears
# - "Retrain 'transformer' ACCEPTED/REJECTED" appears
# - Ensemble weights update after retrain
```

### E2E Test 4: Risk Management

```bash
# Set tight risk limits for testing:
#   max_drawdown = 0.01  (1%)
#   daily_loss_limit = 0.005  (0.5%)

# Run backtest with volatile asset
python scripts/run_backtest.py --strategy ml --assets AAPL \
    --start 2025-01-01 --end 2025-03-01

# Verify in logs:
# - "Max drawdown breached" or "Daily loss limit breached" appears
# - "Risk breached; skipping signal" appears
# - Final equity reflects risk-limited trading
```

### Verification Checklist

- [ ] All layers import without errors: `python -c "import quant"`
- [ ] Unit tests pass: `pytest tests/ -v`
- [ ] Data download works for stock, crypto, and forex symbols
- [ ] Feature pipeline produces consistent output shapes
- [ ] All four model architectures produce correct output shapes
- [ ] Walk-forward training completes without NaN losses
- [ ] Backtest produces non-trivial equity curve
- [ ] Risk manager correctly halts trading on limit breaches
- [ ] State persistence survives process restart
- [ ] HTML dashboard renders all six panels
- [ ] Live trading loop processes bars without errors

---

*Document generated for Quant 2.0 v2.0.0*
