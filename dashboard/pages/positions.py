"""Page 2: Positions & Trades — Once UI style."""

import dash
from dash import html, dcc, callback, Output, Input, State, no_update
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json

from dashboard.components.trade_table import create_trade_table
from dashboard.data.loader import DashboardDataLoader
from dashboard import config

dash.register_page(__name__, path="/positions", name="Positions", order=1)

loader = DashboardDataLoader()
C = config.COLORS

def _export_btn(id_str, label):
    return html.Button(label, id=id_str, n_clicks=0, className="bt-export-btn")


def _pnl_bar(trades):
    pnls = [t.get("pnl", 0) for t in trades]
    colors = [C["profit"] if p >= 0 else C["loss"] for p in pnls]

    fig = go.Figure(go.Bar(
        x=[f"#{i+1}" for i in range(len(pnls))], y=pnls,
        marker_color=colors,
        hovertemplate="Trade %{x}<br>P&L: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(**config.PLOTLY_LAYOUT)
    fig.update_yaxes(title="P&L ($)")
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "300px"})


def _cum_pnl(trades):
    pnls = [t.get("pnl", 0) for t in trades]
    cum = np.cumsum(pnls).tolist()

    fig = go.Figure(go.Scatter(
        x=list(range(1, len(cum) + 1)), y=cum,
        mode="lines+markers", line=dict(color=C["accent"], width=2),
        marker=dict(size=4), fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.08)",
        hovertemplate="Trade %{x}<br>Cum. P&L: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(**config.PLOTLY_LAYOUT)
    fig.update_yaxes(title="Cumulative P&L ($)")
    fig.update_xaxes(showgrid=False, title="Trade #")
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "300px"})


# Static layout — dropdown exists from the start
layout = html.Div([
    dcc.Interval(id="positions-refresh", interval=config.REFRESH_INTERVAL_MS, n_intervals=0),

    # Download components for exports
    dcc.Download(id="positions-download-trades-csv"),
    dcc.Download(id="positions-download-trades-json"),

    html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Positions & Trades", className="section-title"),
        ], style={"display": "flex", "alignItems": "center", "gap": "0.6rem"}),
        html.Div([
            html.Span("Backtest", style={
                "fontFamily": "var(--font-sans)", "fontSize": "0.75rem",
                "color": "var(--text-weak)", "fontWeight": "500",
            }),
            dcc.Dropdown(
                id="positions-bt-select",
                placeholder="Select backtest...",
                clearable=False,
                style={"width": "260px"},
                className="once-dropdown",
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "0.6rem"}),
    ], className="section-header", style={"justifyContent": "space-between"}),
    html.Div(id="positions-content"),
])


@callback(
    Output("positions-bt-select", "options"),
    Output("positions-bt-select", "value"),
    Input("positions-refresh", "n_intervals"),
)
def update_bt_options(_n):
    bt_files = loader.list_backtest_results()
    if not bt_files:
        return [], None
    options = [{"label": p.stem.replace("_", " ").title(), "value": p.stem} for p in bt_files]
    return options, bt_files[-1].stem


@callback(
    Output("positions-content", "children"),
    Input("positions-bt-select", "value"),
)
def update_positions(selected_bt):
    bt_files = loader.list_backtest_results()

    bt_name = selected_bt or (bt_files[-1].stem if bt_files else None)
    bt_data = loader.load_backtest_result(bt_name) if bt_name else None
    trades = bt_data.get("trades", bt_data.get("trade_log", [])) if bt_data else []

    # Active positions
    active = html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Active Positions", className="section-title"),
        ], className="section-header"),
        html.Div([
            html.P("No active positions. Backtest mode shows completed trades only.",
                   style={"color": "var(--text-weak)", "textAlign": "center", "padding": "2rem",
                          "fontSize": "0.78rem"}),
        ], className="card-static"),
    ], className="section")

    if not trades:
        empty = html.Div([
            html.Div([
                html.Div(className="section-indicator"),
                html.Span("Trade History", className="section-title"),
            ], className="section-header"),
            html.Div(className="empty-state", children=[
                html.I(className="bi bi-wallet2", style={"fontSize": "2.5rem", "color": "var(--text-disabled)"}),
                html.P("No trades available. Run a backtest to generate trade data.",
                       className="empty-state-text"),
            ]),
        ], className="section")
        return html.Div([active, empty])

    # Trade history with export buttons
    trade_section = html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Div([
                html.Span("Trade History", className="section-title"),
                html.Span(f"{len(trades)} trades", className="section-count"),
            ], style={"display": "flex", "flexDirection": "column", "gap": "0.1rem"}),
            html.Div([
                _export_btn("positions-export-trades-csv", "CSV"),
                _export_btn("positions-export-trades-json", "JSON"),
            ], style={"marginLeft": "auto", "display": "flex", "gap": "0.4rem"}),
        ], className="section-header", style={"display": "flex", "alignItems": "center"}),
        html.Div(create_trade_table(trades), className="card-static",
                 style={"padding": "0", "overflow": "hidden"}),
    ], className="section")

    # Charts
    charts = html.Div([
        html.Div([
            html.Div([
                html.Div(className="section-indicator"),
                html.Span("P&L by Trade", className="section-title"),
            ], className="section-header"),
            html.Div(_pnl_bar(trades), className="chart-card"),
        ]),
        html.Div([
            html.Div([
                html.Div(className="section-indicator"),
                html.Span("Cumulative P&L", className="section-title"),
            ], className="section-header"),
            html.Div(_cum_pnl(trades), className="chart-card"),
        ]),
    ], className="grid-2", style={"marginBottom": "2rem"})

    return html.Div([active, trade_section, charts])


# ---------------------------------------------------------------------------
# Export callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("positions-download-trades-csv", "data"),
    Input("positions-export-trades-csv", "n_clicks"),
    State("positions-bt-select", "value"),
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
    Output("positions-download-trades-json", "data"),
    Input("positions-export-trades-json", "n_clicks"),
    State("positions-bt-select", "value"),
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
