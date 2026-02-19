"""Page 3: Backtest Explorer — Once UI style."""

import datetime
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.components.kpi_card import create_kpi_card
from dashboard.data.loader import get_shared_loader
from dashboard import config

dash.register_page(__name__, path="/backtest", name="Backtest", order=2)

loader = get_shared_loader()
C = config.COLORS

ASSET_OPTIONS = config.ASSET_OPTIONS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"

_ALL_MODEL_KEYS = [
    "transformer", "itransformer", "lstm", "momentum",
    "causal", "schrodinger", "topological", "hamiltonian", "diffusion", "adversarial",
]
_MODEL_LABELS = {
    "transformer": "Transformer",
    "itransformer": "ITransformer",
    "lstm": "LSTM",
    "momentum": "Momentum",
    "causal": "Causal",
    "schrodinger": "Schrödinger",
    "topological": "Topological",
    "hamiltonian": "Hamiltonian",
    "diffusion": "Diffusion",
    "adversarial": "Adversarial",
}


def _get_trained_model_keys():
    """Return list of model keys that have a checkpoint file."""
    trained = []
    for key in _ALL_MODEL_KEYS:
        for suffix in ["_latest.pt", "_model.pt"]:
            if (_CHECKPOINT_DIR / f"{key}{suffix}").exists():
                trained.append(key)
                break
    return trained


def _get_model_options():
    """Build dropdown options; untrained models are disabled."""
    trained = set(_get_trained_model_keys())
    options = []
    for key in _ALL_MODEL_KEYS:
        is_trained = key in trained
        label = f"{_MODEL_LABELS[key]}  [trained]" if is_trained else f"{_MODEL_LABELS[key]}  [not trained]"
        options.append({"label": label, "value": key, "disabled": not is_trained})
    return options

# Metric display config: (key, label, format, higher_is_better)
METRICS_TABLE = [
    ("total_return", "Total Return", ".2%", True),
    ("annualized_return", "Annualized Return", ".2%", True),
    ("sharpe_ratio", "Sharpe Ratio", ".2f", True),
    ("sortino_ratio", "Sortino Ratio", ".2f", True),
    ("calmar_ratio", "Calmar Ratio", ".2f", True),
    ("max_drawdown", "Max Drawdown", ".2%", False),
    ("max_drawdown_duration", "DD Duration (bars)", ".0f", False),
    ("volatility", "Volatility", ".2%", False),
    ("total_trades", "Total Trades", ".0f", None),
    ("win_rate", "Win Rate", ".1%", True),
    ("avg_win", "Avg Win", ",.2f", True),
    ("avg_loss", "Avg Loss", ",.2f", None),
    ("profit_factor", "Profit Factor", ".2f", True),
    ("expectancy", "Expectancy", ",.2f", True),
    ("initial_capital", "Initial Capital", ",.0f", None),
    ("final_equity", "Final Equity", ",.0f", None),
]

# File-based state for cross-worker backtest tracking
_BT_DIR = Path(tempfile.gettempdir()) / "quant_bt_run"
_BT_DIR.mkdir(exist_ok=True)
_BT_PID_FILE = _BT_DIR / "pid"
_BT_LOG_FILE = _BT_DIR / "output.log"
_BT_RC_FILE = _BT_DIR / "rc"


def _metrics_table(metrics):
    """Build a styled metrics table from backtest metrics dict."""
    rows = []
    for key, label, fmt, higher_better in METRICS_TABLE:
        val = metrics.get(key)
        if val is None:
            continue
        formatted = f"{val:{fmt}}"
        if key in ("avg_win", "avg_loss", "initial_capital", "final_equity", "expectancy"):
            formatted = f"${formatted}"

        style = {"color": "var(--text-default)"}
        if higher_better is True and val > 0:
            style["color"] = "var(--status-profit)"
        elif higher_better is True and val < 0:
            style["color"] = "var(--status-loss)"
        elif higher_better is False and key == "max_drawdown":
            style["color"] = "var(--status-loss)"

        rows.append(html.Tr([
            html.Td(label, style={"color": "var(--text-weak)", "padding": "0.45rem 1rem",
                                   "fontSize": "0.75rem", "borderBottom": "1px solid var(--border-default)"}),
            html.Td(formatted, style={**style, "fontFamily": "var(--font-mono)", "padding": "0.45rem 1rem",
                                       "fontSize": "0.75rem", "textAlign": "right",
                                       "borderBottom": "1px solid var(--border-default)"}),
        ]))

    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


def _equity_with_trades(bt_data):
    """Equity curve with trade entry/exit markers."""
    equity = bt_data.get("equity_curve", [])
    timestamps = bt_data.get("timestamps", [])
    trades = bt_data.get("trades", bt_data.get("trade_log", []))

    ts = None
    if timestamps:
        try:
            ts = pd.to_datetime(timestamps[:len(equity)]).tolist()
        except Exception:
            ts = None
    x_axis = ts or list(range(len(equity)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, y=equity, mode="lines", name="Equity",
        line=dict(color=C["accent"], width=2),
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))

    if trades and ts:
        for t in trades:
            entry_time = t.get("entry_time")
            exit_time = t.get("exit_time")
            pnl = t.get("pnl", 0)

            if entry_time:
                try:
                    entry_dt = pd.Timestamp(entry_time)
                    ts_series = pd.DatetimeIndex(ts)
                    entry_idx = ts_series.get_indexer([entry_dt], method="nearest")[0]
                    if 0 <= entry_idx < len(equity):
                        fig.add_trace(go.Scatter(
                            x=[ts[entry_idx]], y=[equity[entry_idx]],
                            mode="markers", name="Entry",
                            marker=dict(symbol="triangle-up", size=10, color=C["accent"]),
                            hovertemplate=f"Entry: {t.get('symbol', '')} {t.get('side', '')}<br>"
                                          f"Price: ${t.get('entry_price', 0):,.2f}<extra></extra>",
                            showlegend=False,
                        ))
                except Exception:
                    pass

            if exit_time:
                try:
                    exit_dt = pd.Timestamp(exit_time)
                    ts_series = pd.DatetimeIndex(ts)
                    exit_idx = ts_series.get_indexer([exit_dt], method="nearest")[0]
                    if 0 <= exit_idx < len(equity):
                        color = C["profit"] if pnl >= 0 else C["loss"]
                        fig.add_trace(go.Scatter(
                            x=[ts[exit_idx]], y=[equity[exit_idx]],
                            mode="markers", name="Exit",
                            marker=dict(symbol="triangle-down", size=10, color=color),
                            hovertemplate=f"Exit: {t.get('symbol', '')} {t.get('reason', '')}<br>"
                                          f"P&L: ${pnl:+,.2f}<extra></extra>",
                            showlegend=False,
                        ))
                except Exception:
                    pass

    fig.update_layout(**config.PLOTLY_LAYOUT, showlegend=False)
    fig.update_xaxes(**config.PLOTLY_XAXIS)
    fig.update_yaxes(**config.PLOTLY_YAXIS, title="Equity ($)")
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "380px"})


def _comparison_chart(all_results, selected_keys):
    """Overlay equity curves from multiple backtests."""
    colors = [C["accent"], C["profit"], C["loss"], C["warning"], "#A78BFA", "#F472B6"]
    fig = go.Figure()

    for i, key in enumerate(selected_keys):
        bt = all_results.get(key, {})
        equity = bt.get("equity_curve", [])
        timestamps = bt.get("timestamps", [])

        ts = None
        if timestamps:
            try:
                ts = pd.to_datetime(timestamps[:len(equity)]).tolist()
            except Exception:
                ts = None
        x_axis = ts or list(range(len(equity)))

        if equity and equity[0] > 0:
            normalized = [(e / equity[0] - 1) * 100 for e in equity]
        else:
            normalized = equity

        fig.add_trace(go.Scatter(
            x=x_axis, y=normalized, mode="lines",
            name=key.replace("_", " ").title(),
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate="%{x}<br>%{y:+.2f}%<extra>" + key.replace("_", " ").title() + "</extra>",
        ))

    fig.update_layout(
        **config.PLOTLY_LAYOUT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11, color=C["text_weak"])),
    )
    fig.update_xaxes(**config.PLOTLY_XAXIS)
    fig.update_yaxes(**config.PLOTLY_YAXIS, title="Return (%)")
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "350px"})


def _export_btn(id_str, label):
    """Create a badge-pill export button."""
    return html.Button(label, id=id_str, n_clicks=0, className="bt-export-btn")


def _form_input(label, component, hint=None, flex=1):
    """Wrap a form input with a label and optional hint."""
    children = [
        html.Label(label, className="bt-form-label"),
        component,
    ]
    if hint:
        children.append(html.Span(hint, className="bt-form-hint"))
    return html.Div(children, style={"flex": str(flex), "minWidth": "140px"})


# Date defaults
_today = datetime.date.today().isoformat()
_one_year_ago = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()

# Static layout
layout = html.Div([
    dcc.Interval(id="backtest-refresh", interval=config.REFRESH_INTERVAL_MS, n_intervals=0),

    # Download components
    dcc.Download(id="backtest-download-metrics-json"),
    dcc.Download(id="backtest-download-trades-csv"),
    dcc.Download(id="backtest-download-trades-json"),

    # Header with selector
    html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Backtest Explorer", className="section-title"),
        ], style={"display": "flex", "alignItems": "center", "gap": "0.6rem"}),
        html.Div([
            html.Span("Backtest", style={
                "fontFamily": "var(--font-sans)", "fontSize": "0.75rem",
                "color": "var(--text-weak)", "fontWeight": "500",
            }),
            dcc.Dropdown(
                id="backtest-bt-select",
                placeholder="Select backtest...",
                clearable=False,
                style={"width": "260px"},
                className="once-dropdown",
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "0.6rem"}),
    ], className="section-header", style={"justifyContent": "space-between"}),

    # Detail view
    html.Div(id="backtest-detail"),

    # Comparison section
    dcc.ConfirmDialog(
        id="bt-reset-confirm",
        message="Alle Backtest-Ergebnisse unwiderruflich löschen?",
    ),
    html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Compare Backtests", className="section-title"),
            html.Div([
                html.Span(id="bt-reset-status"),
                html.Button(
                    [html.I(className="bi bi-trash3", style={"fontSize": "0.7rem"}), "Reset All"],
                    id="bt-reset-btn",
                    n_clicks=0,
                    className="bt-export-btn",
                    style={"color": "var(--status-loss)", "borderColor": "rgba(239,68,68,0.4)"},
                ),
            ], style={"marginLeft": "auto", "display": "flex", "alignItems": "center", "gap": "0.6rem"}),
        ], className="section-header", style={"display": "flex", "alignItems": "center"}),
        html.Div([
            dcc.Checklist(
                id="backtest-compare-select",
                inline=True,
                className="once-checklist",
                labelStyle={"display": "inline-flex", "alignItems": "center", "gap": "0.3rem",
                            "marginRight": "1rem", "fontSize": "0.75rem", "fontFamily": "var(--font-mono)",
                            "color": "var(--text-default)", "cursor": "pointer"},
                inputStyle={"accentColor": C["accent"]},
            ),
        ], style={"padding": "0 0 0.5rem"}),
        html.Div(id="backtest-comparison-chart"),
    ], className="section"),

    # Run New Backtest form
    html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Run New Backtest", className="section-title"),
        ], className="section-header"),
        html.Div([
            # Row 1: Strategy, Assets, Interval
            html.Div([
                _form_input("Strategy", dcc.Dropdown(
                    id="bt-run-strategy",
                    options=[
                        {"label": "Ensemble", "value": "ensemble"},
                        {"label": "ML (single model)", "value": "ml"},
                        {"label": "Mean Reversion", "value": "mean_reversion"},
                        {"label": "Trend Following", "value": "trend_following"},
                        {"label": "Volatility Targeting", "value": "volatility_targeting"},
                        {"label": "Regime Adaptive", "value": "regime_adaptive"},
                    ],
                    value="ensemble",
                    clearable=False,
                    className="once-dropdown",
                    style={"width": "100%"},
                )),
                _form_input("Assets", dcc.Dropdown(
                    id="bt-run-assets",
                    options=ASSET_OPTIONS,
                    value=["AAPL"],
                    multi=True,
                    placeholder="Select assets...",
                    searchable=True,
                    className="once-dropdown",
                    style={"width": "100%"},
                ), hint="Search or select multiple", flex=2),
                _form_input("Interval", dcc.Dropdown(
                    id="bt-run-interval",
                    options=[
                        {"label": "1 min", "value": "1m"},
                        {"label": "5 min", "value": "5m"},
                        {"label": "15 min", "value": "15m"},
                        {"label": "1 hour", "value": "1h"},
                        {"label": "1 day", "value": "1d"},
                    ],
                    value="1d",
                    clearable=False,
                    className="once-dropdown",
                    style={"width": "100%"},
                )),
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"}),

            # Row 1b: Model selection (visible only for ensemble/ml)
            html.Div([
                _form_input("Models", dcc.Dropdown(
                    id="bt-run-models",
                    options=_get_model_options(),
                    value=_get_trained_model_keys(),
                    multi=True,
                    placeholder="Select models...",
                    searchable=True,
                    className="once-dropdown",
                    style={"width": "100%"},
                ), hint="Only trained models can be selected", flex=1),
            ], id="bt-run-models-row", style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"}),

            # Divider
            html.Div(className="bt-form-divider"),

            # Row 2: Start, End, Capital
            html.Div([
                _form_input("Start Date", dcc.DatePickerSingle(
                    id="bt-run-start",
                    date=_one_year_ago,
                    max_date_allowed=_today,
                    display_format="YYYY-MM-DD",
                    className="bt-form-input",
                )),
                _form_input("End Date", dcc.DatePickerSingle(
                    id="bt-run-end",
                    date=_today,
                    max_date_allowed=_today,
                    display_format="YYYY-MM-DD",
                    className="bt-form-input",
                )),
                _form_input("Capital ($)", dcc.Input(
                    id="bt-run-capital",
                    type="number",
                    value=100000,
                    min=1000,
                    step=1000,
                    className="bt-form-input",
                )),
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"}),

            # Divider before button
            html.Div(className="bt-form-divider"),

            # Run button + status — right-aligned
            html.Div([
                html.Div(id="bt-run-status"),
                html.Button(
                    [
                        html.I(className="bi bi-play-fill", style={"fontSize": "0.8rem"}),
                        "Run Backtest",
                    ],
                    id="bt-run-btn",
                    n_clicks=0,
                    style={
                        "fontFamily": "var(--font-sans)", "fontSize": "0.78rem", "fontWeight": "700",
                        "padding": "0.6rem 1.6rem", "minHeight": "38px",
                        "borderRadius": "var(--radius-md)",
                        "border": "1px solid rgba(59, 130, 246, 0.4)",
                        "background": "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)",
                        "color": "#fff", "cursor": "pointer",
                        "display": "inline-flex", "alignItems": "center", "gap": "0.4rem",
                        "letterSpacing": "0.01em",
                    },
                ),
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end", "gap": "1rem"}),

            # Output log (hidden by default, animated via CSS class toggle)
            html.Pre(id="bt-run-log", style={
                "fontFamily": "var(--font-mono)", "fontSize": "0.68rem",
                "color": C["text"], "padding": "0 0.8rem",
                "background": C["bg_overlay"], "borderRadius": "var(--radius-sm)",
                "border": f"1px solid {C['border']}",
                "whiteSpace": "pre-wrap",
            }),

            # Interval to poll for backtest completion
            dcc.Interval(id="bt-run-poll", interval=2000, disabled=True),
        ], className="bt-run-card"),
    ], className="section"),
])


@callback(
    Output("backtest-bt-select", "options"),
    Output("backtest-bt-select", "value"),
    Output("backtest-compare-select", "options"),
    Input("backtest-refresh", "n_intervals"),
)
def update_bt_options(_n):
    bt_files = loader.list_backtest_results()
    if not bt_files:
        return [], None, []
    options = [{"label": p.stem.replace("_", " ").title(), "value": p.stem} for p in bt_files]
    return options, bt_files[-1].stem, options


@callback(
    Output("backtest-detail", "children"),
    Input("backtest-bt-select", "value"),
)
def update_detail(selected_bt):
    if not selected_bt:
        return html.Div(className="empty-state", children=[
            html.I(className="bi bi-clock-history",
                   style={"fontSize": "2.5rem", "color": "var(--text-disabled)"}),
            html.P("No backtest data available.", className="empty-state-text"),
        ])

    bt_data = loader.load_backtest_result(selected_bt)
    if not bt_data:
        return html.Div()

    metrics = bt_data.get("metrics", {})
    trades = bt_data.get("trades", bt_data.get("trade_log", []))

    total_return = metrics.get("total_return", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    max_dd = metrics.get("max_drawdown", 0)
    win_rate = metrics.get("win_rate", 0)
    total_trades = int(metrics.get("total_trades", 0))
    profit_factor = metrics.get("profit_factor", 0)

    kpis = html.Div([
        create_kpi_card("Total Return", f"{total_return:.2%}",
                        "profit" if total_return >= 0 else "loss"),
        create_kpi_card("Sharpe Ratio", f"{sharpe:.2f}",
                        "profit" if sharpe >= 1 else ("loss" if sharpe < 0 else "")),
        create_kpi_card("Max Drawdown", f"{max_dd:.2%}", "loss"),
        create_kpi_card("Win Rate", f"{win_rate:.1%}",
                        "profit" if win_rate >= 0.5 else "loss"),
        create_kpi_card("Trades", f"{total_trades}"),
        create_kpi_card("Profit Factor", f"{profit_factor:.2f}",
                        "profit" if profit_factor >= 1 else "loss"),
    ], className="kpi-grid")

    equity_section = html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Equity Curve", className="section-title"),
        ], className="section-header"),
        html.Div(_equity_with_trades(bt_data), className="chart-card"),
    ], className="section")

    # Full metrics table with export button
    metrics_section = html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Full Metrics", className="section-title"),
            html.Div([
                _export_btn("backtest-export-metrics-json", "Export JSON"),
            ], style={"marginLeft": "auto", "display": "flex", "gap": "0.4rem"}),
        ], className="section-header", style={"display": "flex", "alignItems": "center"}),
        html.Div(_metrics_table(metrics), className="card-static",
                 style={"padding": "0.5rem 0", "overflow": "auto"}),
    ], className="section")

    # Trade summary with export buttons
    trade_section = html.Div()
    if trades:
        pnls = [t.get("pnl", 0) for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        best = max(pnls) if pnls else 0
        worst = min(pnls) if pnls else 0

        trade_section = html.Div([
            html.Div([
                html.Div(className="section-indicator"),
                html.Div([
                    html.Span("Trade Summary", className="section-title"),
                    html.Span(f"{len(trades)} trades", className="section-count"),
                ], style={"display": "flex", "flexDirection": "column", "gap": "0.1rem"}),
                html.Div([
                    _export_btn("backtest-export-trades-csv", "CSV"),
                    _export_btn("backtest-export-trades-json", "JSON"),
                ], style={"marginLeft": "auto", "display": "flex", "gap": "0.4rem"}),
            ], className="section-header", style={"display": "flex", "alignItems": "center"}),
            html.Div([
                create_kpi_card("Wins", f"{wins}", "profit"),
                create_kpi_card("Losses", f"{losses}", "loss"),
                create_kpi_card("Best Trade", f"${best:+,.2f}",
                                "profit" if best > 0 else ""),
                create_kpi_card("Worst Trade", f"${worst:+,.2f}", "loss"),
            ], className="kpi-grid"),
        ], className="section")

    return html.Div([kpis, equity_section, html.Div([metrics_section, trade_section], className="grid-2")])


@callback(
    Output("backtest-comparison-chart", "children"),
    Input("backtest-compare-select", "value"),
)
def update_comparison(selected_keys):
    if not selected_keys or len(selected_keys) < 2:
        return html.P("Select 2 or more backtests to compare.",
                       style={"color": "var(--text-weak)", "fontSize": "0.75rem",
                              "textAlign": "center", "padding": "2rem"})

    all_results = loader.load_all_backtest_results()
    return html.Div(_comparison_chart(all_results, selected_keys), className="chart-card")


# ---------------------------------------------------------------------------
# Export callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("backtest-download-metrics-json", "data"),
    Input("backtest-export-metrics-json", "n_clicks"),
    State("backtest-bt-select", "value"),
    prevent_initial_call=True,
)
def export_metrics_json(n_clicks, selected_bt):
    if not n_clicks or not selected_bt:
        return no_update
    bt_data = loader.load_backtest_result(selected_bt)
    if not bt_data:
        return no_update
    metrics = bt_data.get("metrics", {})
    return dict(content=json.dumps(metrics, indent=2, default=str),
                filename=f"{selected_bt}_metrics.json")


@callback(
    Output("backtest-download-trades-csv", "data"),
    Input("backtest-export-trades-csv", "n_clicks"),
    State("backtest-bt-select", "value"),
    prevent_initial_call=True,
)
def export_trades_csv(n_clicks, selected_bt):
    if not n_clicks or not selected_bt:
        return no_update
    bt_data = loader.load_backtest_result(selected_bt)
    if not bt_data:
        return no_update
    trades = bt_data.get("trades", bt_data.get("trade_log", []))
    if not trades:
        return no_update
    df = pd.DataFrame(trades)
    return dict(content=df.to_csv(index=False), filename=f"{selected_bt}_trades.csv")


@callback(
    Output("backtest-download-trades-json", "data"),
    Input("backtest-export-trades-json", "n_clicks"),
    State("backtest-bt-select", "value"),
    prevent_initial_call=True,
)
def export_trades_json(n_clicks, selected_bt):
    if not n_clicks or not selected_bt:
        return no_update
    bt_data = loader.load_backtest_result(selected_bt)
    if not bt_data:
        return no_update
    trades = bt_data.get("trades", bt_data.get("trade_log", []))
    if not trades:
        return no_update
    return dict(content=json.dumps(trades, indent=2, default=str),
                filename=f"{selected_bt}_trades.json")


# ---------------------------------------------------------------------------
# Reset All Backtests
# ---------------------------------------------------------------------------

@callback(
    Output("bt-reset-confirm", "displayed"),
    Input("bt-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def confirm_reset(n_clicks):
    if not n_clicks:
        return False
    return True


@callback(
    Output("bt-reset-status", "children"),
    Output("backtest-bt-select", "options", allow_duplicate=True),
    Output("backtest-bt-select", "value", allow_duplicate=True),
    Output("backtest-compare-select", "options", allow_duplicate=True),
    Output("backtest-compare-select", "value"),
    Output("backtest-comparison-chart", "children", allow_duplicate=True),
    Input("bt-reset-confirm", "submit_n_clicks"),
    prevent_initial_call=True,
)
def reset_all_backtests(submit_n_clicks):
    if not submit_n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update

    bt_files = loader.list_backtest_results()
    count = 0
    for f in bt_files:
        try:
            f.unlink()
            count += 1
        except OSError:
            pass

    loader.clear_cache()

    status = html.Span(
        f"{count} Backtests gelöscht.",
        style={"fontSize": "0.72rem", "color": "var(--status-profit)", "fontFamily": "var(--font-mono)"},
    )
    return status, [], None, [], [], html.Div()


# ---------------------------------------------------------------------------
# Run Backtest helpers (file-based, works across gunicorn workers)
# ---------------------------------------------------------------------------

def _bt_is_running():
    """Check if a backtest process is currently running via PID file."""
    if not _BT_PID_FILE.exists():
        return False
    try:
        pid = int(_BT_PID_FILE.read_text().strip())
        os.kill(pid, 0)  # signal 0 = check if alive
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def _bt_read_log():
    """Read the current backtest log from file."""
    if _BT_LOG_FILE.exists():
        try:
            return _BT_LOG_FILE.read_text(errors="replace")
        except OSError:
            return ""
    return ""


def _bt_read_rc():
    """Read the return code file. Returns None if not finished."""
    if _BT_RC_FILE.exists():
        try:
            return int(_BT_RC_FILE.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


# ---------------------------------------------------------------------------
# Run Backtest callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("bt-run-models-row", "style"),
    Input("bt-run-strategy", "value"),
)
def toggle_model_selector(strategy):
    if strategy in ("ensemble", "ml"):
        return {"display": "flex", "gap": "1rem", "flexWrap": "wrap"}
    return {"display": "none"}


@callback(
    Output("bt-run-poll", "disabled"),
    Output("bt-run-status", "children"),
    Output("bt-run-log", "className"),
    Output("bt-run-log", "children"),
    Output("bt-run-btn", "disabled"),
    Input("bt-run-btn", "n_clicks"),
    State("bt-run-strategy", "value"),
    State("bt-run-assets", "value"),
    State("bt-run-interval", "value"),
    State("bt-run-start", "date"),
    State("bt-run-end", "date"),
    State("bt-run-capital", "value"),
    State("bt-run-models", "value"),
    prevent_initial_call=True,
)
def start_backtest(n_clicks, strategy, assets, interval, start, end, capital, models):
    if not n_clicks or not all([strategy, assets, interval, start, end, capital]):
        return no_update, no_update, no_update, no_update, no_update

    if _bt_is_running():
        status = html.Span([
            html.Span(className="status-dot", style={"background": "var(--status-warning)"}),
            "A backtest is already running.",
        ], className="bt-status-badge running")
        return no_update, status, no_update, no_update, no_update

    # Build assets string from multi-select dropdown
    assets_str = ",".join(assets) if isinstance(assets, list) else assets

    import sys
    project_root = Path(__file__).resolve().parent.parent.parent
    script = project_root / "scripts" / "run_backtest.py"

    # Clean up previous run files
    for f in (_BT_PID_FILE, _BT_LOG_FILE, _BT_RC_FILE):
        f.unlink(missing_ok=True)

    # Build command
    cmd = [
        sys.executable, str(script),
        "--strategy", strategy,
        "--assets", assets_str,
        "--interval", interval,
        "--start", start,
        "--end", end,
        "--capital", str(int(capital)),
    ]
    # Pass --models for ML-based strategies when user selected specific models
    if strategy in ("ensemble", "ml") and models:
        models_str = ",".join(models) if isinstance(models, list) else models
        cmd.extend(["--models", models_str])

    # Launch subprocess with stdout/stderr going to log file
    log_fh = open(_BT_LOG_FILE, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(project_root),
        start_new_session=True,  # detach from gunicorn worker
    )

    # Write PID so any worker can check on it
    _BT_PID_FILE.write_text(str(proc.pid))

    # Background waiter writes rc file when process exits
    import threading

    def _wait():
        rc = proc.wait()
        log_fh.close()
        _BT_RC_FILE.write_text(str(rc))

    threading.Thread(target=_wait, daemon=True).start()

    status = html.Span([
        html.Span(className="status-dot", style={
            "background": "var(--status-warning)",
            "boxShadow": "0 0 6px var(--status-warning)",
        }),
        "Running...",
    ], className="bt-status-badge running")

    return False, status, "visible", "Starting backtest...\n", True


@callback(
    Output("bt-run-status", "children", allow_duplicate=True),
    Output("bt-run-log", "children", allow_duplicate=True),
    Output("bt-run-poll", "disabled", allow_duplicate=True),
    Output("bt-run-btn", "disabled", allow_duplicate=True),
    Input("bt-run-poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_backtest(_n):
    log_text = _bt_read_log()

    # Check if still running
    if _bt_is_running():
        status = html.Span([
            html.Span(className="status-dot", style={
                "background": "var(--status-warning)",
                "boxShadow": "0 0 6px var(--status-warning)",
            }),
            "Running...",
        ], className="bt-status-badge running")
        return status, log_text or "Starting backtest...\n", no_update, True

    # Process finished — check return code
    rc = _bt_read_rc()

    if rc is None and not _BT_PID_FILE.exists():
        # No backtest was ever started in this session
        return no_update, no_update, True, False

    if rc is None:
        # PID file exists but process gone, rc not yet written — brief race, keep polling
        return no_update, log_text, no_update, True

    # Clear cache so new results show up
    loader.clear_cache()

    if rc == 0:
        status = html.Span([
            html.I(className="bi bi-check-circle-fill"),
            "Complete",
        ], className="bt-status-badge complete")
    else:
        status = html.Span([
            html.I(className="bi bi-exclamation-triangle-fill"),
            f"Failed (exit {rc})",
        ], className="bt-status-badge failed")

    return status, log_text, True, False
