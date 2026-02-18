"""Page 1: Portfolio Overview — Once UI style."""

import dash
from dash import html, dcc, callback, Output, Input
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from dashboard.components.kpi_card import create_kpi_card
from dashboard.components.equity_chart import create_equity_chart
from dashboard.components.drawdown_chart import create_drawdown_chart
from dashboard.components.heatmap import create_monthly_returns_heatmap
from dashboard.data.loader import DashboardDataLoader
from dashboard import config

dash.register_page(__name__, path="/", name="Overview", order=0)

loader = DashboardDataLoader()
C = config.COLORS


def _empty_state():
    return html.Div([
        html.Div(className="empty-state-icon", children=html.I(className="bi bi-inbox")),
        html.P("No backtest data available. Run a backtest first.", className="empty-state-text"),
        html.Code(
            "python scripts/run_backtest.py --strategy ensemble --assets AAPL --interval 1d",
            className="empty-state-hint",
        ),
    ], className="empty-state")


def _return_histogram(returns):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=50, name="Daily Returns",
        marker_color=C["accent"], opacity=0.7,
    ))

    if len(returns) > 1:
        mu, sigma = np.mean(returns), np.std(returns)
        if sigma > 0:
            x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
            bin_width = (max(returns) - min(returns)) / 50 if len(returns) > 0 else 1
            fig.add_trace(go.Scatter(
                x=x_range, y=pdf * len(returns) * bin_width,
                mode="lines", name="Normal Dist.",
                line=dict(color=C["loss"], width=2, dash="dash"),
            ))

    fig.update_layout(
        **config.PLOTLY_LAYOUT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11, color=C["text_weak"])),
    )
    fig.update_xaxes(title="Daily Return", showgrid=False)
    fig.update_yaxes(title="Frequency", showgrid=True, gridcolor="rgba(255,255,255,0.04)")

    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "300px"})


# Static layout — dropdown exists from the start
layout = html.Div([
    dcc.Interval(id="overview-refresh", interval=config.REFRESH_INTERVAL_MS, n_intervals=0),
    html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Portfolio Overview", className="section-title"),
        ], style={"display": "flex", "alignItems": "center", "gap": "0.6rem"}),
        html.Div([
            html.Span("Backtest", style={
                "fontFamily": "var(--font-sans)", "fontSize": "0.75rem",
                "color": "var(--text-weak)", "fontWeight": "500",
            }),
            dcc.Dropdown(
                id="overview-bt-select",
                placeholder="Select backtest...",
                clearable=False,
                style={"width": "260px"},
                className="once-dropdown",
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "0.6rem"}),
    ], className="section-header", style={"justifyContent": "space-between"}),
    html.Div(id="overview-content"),
])


@callback(
    Output("overview-bt-select", "options"),
    Output("overview-bt-select", "value"),
    Input("overview-refresh", "n_intervals"),
)
def update_bt_options(_n):
    bt_files = loader.list_backtest_results()
    if not bt_files:
        return [], None
    options = [{"label": p.stem.replace("_", " ").title(), "value": p.stem} for p in bt_files]
    return options, bt_files[-1].stem


@callback(
    Output("overview-content", "children"),
    Input("overview-bt-select", "value"),
)
def update_overview(selected_bt):
    bt_files = loader.list_backtest_results()

    if not bt_files or not selected_bt:
        return _empty_state()

    bt_data = loader.load_backtest_result(selected_bt)

    if not bt_data or not bt_data.get("equity_curve"):
        return _empty_state()

    metrics = bt_data.get("metrics", {})
    equity_curve = bt_data.get("equity_curve", [])
    returns = bt_data.get("returns", [])
    timestamps = bt_data.get("timestamps")

    ts_list = None
    if timestamps:
        try:
            ts_list = pd.to_datetime(timestamps[:len(equity_curve)]).tolist()
        except Exception:
            ts_list = None

    # KPI values
    total_return = metrics.get("total_return", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    max_dd = metrics.get("max_drawdown", 0)
    win_rate = metrics.get("win_rate", 0)
    final_equity = metrics.get("final_equity", 0)
    daily_pnl = returns[-1] * 100 if returns else 0

    # KPI section
    kpi_section = html.Div([
        html.Div([
            create_kpi_card("Total Return", f"{total_return:.2%}",
                            "profit" if total_return >= 0 else "loss"),
            create_kpi_card("Sharpe Ratio", f"{sharpe:.2f}",
                            "profit" if sharpe >= 1 else ("loss" if sharpe < 0 else "")),
            create_kpi_card("Max Drawdown", f"{max_dd:.2%}", "loss"),
            create_kpi_card("Win Rate", f"{win_rate:.1%}",
                            "profit" if win_rate >= 0.5 else "loss"),
            create_kpi_card("Portfolio Value", f"${final_equity:,.0f}", "accent"),
            create_kpi_card("Daily P&L", f"{daily_pnl:+.2f}%",
                            "profit" if daily_pnl >= 0 else "loss"),
        ], className="kpi-grid"),
    ], className="section")

    # Equity curve
    equity_section = html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Equity Curve", className="section-title"),
        ], className="section-header"),
        html.Div([
            create_equity_chart(equity_curve, timestamps=ts_list),
        ], className="chart-card"),
    ], className="section")

    # Drawdown + Heatmap
    returns_series = pd.Series(dtype=float)
    if returns:
        if ts_list and len(ts_list) >= len(returns):
            returns_series = pd.Series(returns, index=pd.DatetimeIndex(ts_list[:len(returns)]))
        else:
            returns_series = pd.Series(returns)

    charts_section = html.Div([
        html.Div([
            html.Div([
                html.Div(className="section-indicator"),
                html.Span("Drawdown", className="section-title"),
            ], className="section-header"),
            html.Div(create_drawdown_chart(equity_curve, timestamps=ts_list), className="chart-card"),
        ]),
        html.Div([
            html.Div([
                html.Div(className="section-indicator"),
                html.Span("Monthly Returns", className="section-title"),
            ], className="section-header"),
            html.Div(create_monthly_returns_heatmap(returns_series), className="chart-card"),
        ]),
    ], className="grid-2", style={"marginBottom": "2rem"})

    # Return distribution
    dist_section = html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Return Distribution", className="section-title"),
        ], className="section-header"),
        html.Div(_return_histogram(returns), className="chart-card"),
    ], className="section")

    return html.Div([kpi_section, equity_section, charts_section, dist_section])
