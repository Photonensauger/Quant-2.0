# Quant 2.0 Dashboard

Dash-based trading analytics dashboard for the Quant 2.0 platform.

## Pages

| Page | Path | Description |
|------|------|-------------|
| Overview | `/` | Market watchlist, portfolio KPIs, equity chart |
| Positions | `/positions` | Trade log (sortable/filterable), P&L charts |
| Backtest | `/backtest` | Run backtests, view results with equity/drawdown charts |
| Models | `/models` | Train ML models, view training metrics |
| Risk | `/risk` | Sharpe, VaR/CVaR, rolling volatility, threshold alerts |
| System | `/system` | System health, cache stats, data status |

## Local Development

```bash
# From project root
pip install -e ".[dashboard]"

# Run with hot reload
DASH_DEBUG=true python -m dashboard.app

# Open http://localhost:8050
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QUANT_DATA_DIR` | `data/cache` | Parquet market data directory |
| `QUANT_BACKTEST_DIR` | `data/backtest` | Backtest result JSONs |
| `QUANT_MODELS_DIR` | `data/models` | Trained model artifacts |
| `DASH_HOST` | `0.0.0.0` | Server bind address |
| `DASH_PORT` | `8050` | Server port |
| `DASH_DEBUG` | `false` | Enable Dash debug mode |
| `DASH_REFRESH_MS` | `60000` | Auto-refresh interval (ms) |

## Docker Build

```bash
docker build -t quant-dashboard -f Dockerfile .
docker run -p 8050:8050 \
  -v /path/to/data:/app/data \
  quant-dashboard
```

## Pi5 K3s Deployment

```bash
# Build ARM64 image
docker buildx build --platform linux/arm64 -t registry.lan/quant-dashboard:latest --push .

# Apply K3s manifests
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/dashboard-ingress.yaml

# Verify
kubectl get pods -l app=quant-dashboard
curl -k https://quant.lan
```

## Tests

```bash
pytest tests/test_dashboard/ -v
```

Tests cover the data layer (cache TTL/eviction, loader parquet/JSON reads, caching behavior). No browser-based callback tests.

## Architecture

```
dashboard/
├── app.py              # Dash app entrypoint
├── config.py           # Settings, colors, thresholds
├── assets/style.css    # Once UI dark theme
├── components/         # Reusable UI components
│   ├── kpi_card.py
│   ├── trade_table.py  # Dash DataTable with sort/filter
│   ├── equity_chart.py
│   ├── drawdown_chart.py
│   ├── heatmap.py
│   └── navbar.py
├── data/
│   ├── cache.py        # TTLCache with max_size eviction
│   └── loader.py       # Parquet + JSON data loader
└── pages/              # Multi-page Dash pages
    ├── overview.py
    ├── positions.py
    ├── backtest.py
    ├── models.py
    ├── risk.py
    └── system.py
```
