# Quant 2.0

**Self-learning Multi-Asset Quantitative Trading System**

Quant 2.0 is a modular, end-to-end quantitative trading platform that combines deep learning models with adaptive portfolio management and risk controls. It supports stocks, crypto, and forex — from data ingestion through backtesting to live paper/real trading via the Alpaca API.

## Key Features

- **Multiple Model Architectures** — Decoder Transformer, ITransformer (ICLR 2024), Attention-LSTM, Momentum Transformer, Schrodinger Transformer, Topological Attention Network, Adversarial Regime Model, Entropic Portfolio Diffusion, Causal Discovery Transformer, Hamiltonian Neural ODE, and a PPO reinforcement learning agent
- **Self-Learning Loop** — Online retraining triggered by performance degradation, schedule, or Bayesian change-point detection (BOCPD)
- **Trading Strategies** — Mean Reversion (RSI + Bollinger Bands), Trend Following (MACD + ADX), Volatility Targeting, Regime Adaptive (changepoint-aware), ML Signal, and Sharpe-weighted Ensemble voting
- **Feature Engineering** — 41 technical indicators, 10 calendar encodings, BOCPD regime detection, 7 information geometry features (Fisher information, Renyi entropy spectrum, KL divergence rate), correlation filtering, and rolling z-score normalization
- **Portfolio Management** — Mean-variance, min-variance, risk-parity, and equal-weight optimization with Kelly/volatility-adjusted position sizing
- **Risk Management** — VaR, max drawdown, daily loss limits, concentration checks, and ATR-based stop-loss/take-profit
- **Backtesting Engine** — Walk-forward validation with 17 performance metrics and Plotly HTML dashboards
- **State Persistence** — Atomic checkpoints with 3-level backup rotation and graceful shutdown recovery
- **Multi-Asset Support** — Stocks (YFinance), crypto (ccxt), and forex data providers with Parquet caching

## Architecture

The system is organized into 11 layers with strict top-down dependencies:

```
Scripts → Core → Execution / Backtest / Strategies → Portfolio
                                                        ↓
                    Models → Features → Data → Config → Utils
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full diagrams, interface contracts, and data flow documentation.

## Tech Stack

- **Python 3.11+**
- **PyTorch** — model training and inference
- **pandas / NumPy / SciPy / scikit-learn** — data processing and feature engineering
- **yfinance / ccxt** — market data providers
- **Alpaca API** — paper and live trade execution
- **Plotly** — performance visualization dashboards
- **Parquet (PyArrow)** — local data caching
- **loguru** — structured logging

## Getting Started

```bash
# Clone and install
git clone https://github.com/Photonensauger/Quant-2.0.git
cd Quant-2.0
pip install -e .

# Copy environment config
cp .env.example .env
# Edit .env with your API keys (Alpaca, crypto exchange)

# Download market data
python scripts/download_data.py --assets AAPL --interval 1d --days 365

# Train models
python scripts/train_model.py --model all --assets AAPL --interval 1d --epochs 20

# Run backtest
python scripts/run_backtest.py --strategy ensemble --assets AAPL \
    --start 2025-06-01 --end 2025-12-31 --capital 100000

# Paper trading
python scripts/live_trade.py --strategy ensemble --broker paper \
    --assets AAPL --interval 5m
```

## Project Structure

```
quant/
├── config/       # SystemConfig and all sub-configs
├── data/         # Data providers, Parquet storage, PyTorch datasets
├── features/     # Technical indicators, time features, BOCPD, information geometry, pipeline
├── models/       # Transformer, iTransformer, LSTM, Momentum, Schrodinger, Topological, Adversarial, Entropic Diffusion, Causal Discovery, Hamiltonian ODE, PPO, losses, trainer
├── strategies/   # Mean reversion, trend following, volatility targeting, regime adaptive, ML signal, ensemble
├── portfolio/    # Portfolio optimization, position sizing, risk management
├── backtest/     # Backtest engine and performance metrics
├── execution/    # Paper and live (Alpaca) trade executors
├── core/         # SelfTrainer loop and state persistence
└── utils/        # Logging and Plotly visualization

scripts/          # CLI entry points (download, train, backtest, live)
tests/            # Unit tests mirroring package structure
docs/             # Architecture and algorithm documentation
```

## License

This project is licensed under the [MIT License](LICENSE).
