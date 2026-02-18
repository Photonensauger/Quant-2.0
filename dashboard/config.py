import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("QUANT_DATA_DIR", PROJECT_ROOT / "data" / "cache"))
BACKTEST_DIR = Path(os.environ.get("QUANT_BACKTEST_DIR", PROJECT_ROOT / "data" / "backtest"))
MODELS_DIR = Path(os.environ.get("QUANT_MODELS_DIR", PROJECT_ROOT / "data" / "models"))

# Dashboard settings
APP_TITLE = "Quant 2.0"
DEBUG = os.environ.get("DASH_DEBUG", "false").lower() == "true"
HOST = os.environ.get("DASH_HOST", "0.0.0.0")
PORT = int(os.environ.get("DASH_PORT", "8050"))
REFRESH_INTERVAL_MS = int(os.environ.get("DASH_REFRESH_MS", "60000"))

# Once UI color tokens
COLORS = {
    "bg_page": "#09090B",
    "bg_surface": "#111113",
    "bg_elevated": "#18181B",
    "bg_overlay": "#222225",
    "border": "rgba(255, 255, 255, 0.06)",
    "border_accent": "rgba(59, 130, 246, 0.35)",
    "text_strong": "#F4F4F5",
    "text": "#A1A1AA",
    "text_weak": "#63636B",
    "text_disabled": "#3F3F46",
    "accent": "#3B82F6",
    "accent_soft": "rgba(59, 130, 246, 0.10)",
    "profit": "#22C55E",
    "loss": "#EF4444",
    "warning": "#F59E0B",
    "neutral": "#52525B",
}

# Popular assets for dropdowns (shared across pages)
ASSET_OPTIONS = [
    # US Large Cap
    {"label": "AAPL — Apple", "value": "AAPL"},
    {"label": "MSFT — Microsoft", "value": "MSFT"},
    {"label": "GOOGL — Alphabet", "value": "GOOGL"},
    {"label": "AMZN — Amazon", "value": "AMZN"},
    {"label": "NVDA — NVIDIA", "value": "NVDA"},
    {"label": "META — Meta", "value": "META"},
    {"label": "TSLA — Tesla", "value": "TSLA"},
    {"label": "BRK-B — Berkshire", "value": "BRK-B"},
    {"label": "JPM — JPMorgan", "value": "JPM"},
    {"label": "V — Visa", "value": "V"},
    {"label": "JNJ — Johnson & Johnson", "value": "JNJ"},
    {"label": "UNH — UnitedHealth", "value": "UNH"},
    {"label": "XOM — Exxon Mobil", "value": "XOM"},
    {"label": "PG — Procter & Gamble", "value": "PG"},
    {"label": "MA — Mastercard", "value": "MA"},
    {"label": "HD — Home Depot", "value": "HD"},
    {"label": "DIS — Walt Disney", "value": "DIS"},
    {"label": "NFLX — Netflix", "value": "NFLX"},
    {"label": "AMD — AMD", "value": "AMD"},
    {"label": "CRM — Salesforce", "value": "CRM"},
    # ETFs
    {"label": "SPY — S&P 500 ETF", "value": "SPY"},
    {"label": "QQQ — Nasdaq 100 ETF", "value": "QQQ"},
    {"label": "IWM — Russell 2000 ETF", "value": "IWM"},
    {"label": "DIA — Dow Jones ETF", "value": "DIA"},
    {"label": "GLD — Gold ETF", "value": "GLD"},
    {"label": "TLT — 20+ Year Treasury ETF", "value": "TLT"},
    {"label": "VTI — Total Stock Market ETF", "value": "VTI"},
    # Crypto-related
    {"label": "COIN — Coinbase", "value": "COIN"},
    {"label": "MSTR — MicroStrategy", "value": "MSTR"},
    # International
    {"label": "BABA — Alibaba", "value": "BABA"},
    {"label": "TSM — TSMC", "value": "TSM"},
]

# Plotly chart base config (reused across all charts)
# NOTE: Do NOT include xaxis/yaxis/showlegend here — charts override these
# and Plotly raises "multiple values" if they appear in both **kwargs and keyword args.
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg_surface"],
    plot_bgcolor=COLORS["bg_surface"],
    font=dict(family="Plus Jakarta Sans, -apple-system, sans-serif", color=COLORS["text"]),
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
)

# Default axis styles — apply via fig.update_xaxes() / fig.update_yaxes()
PLOTLY_XAXIS = dict(showgrid=False, zeroline=False)
PLOTLY_YAXIS = dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False)

# Risk threshold alerts
RISK_THRESHOLDS = {
    "sharpe_min": 0.5,
    "max_drawdown_warn": -0.15,
    "max_drawdown_critical": -0.25,
    "volatility_warn": 0.30,
    "var_95_warn": -0.03,
}
