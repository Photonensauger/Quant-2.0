"""Page 5: Risk Management — Once UI style."""

import dash
from dash import html, dcc, callback, Output, Input
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from dashboard.components.kpi_card import create_kpi_card
from dashboard.data.loader import DashboardDataLoader
from dashboard import config

dash.register_page(__name__, path="/risk", name="Risk", order=4)

loader = DashboardDataLoader()
C = config.COLORS
THRESHOLDS = config.RISK_THRESHOLDS


def _rolling_chart(returns, timestamps=None, window=21):
    """Rolling volatility and Sharpe ratio chart."""
    if len(returns) < window:
        return html.P("Insufficient data for rolling metrics.",
                       style={"color": "var(--text-weak)", "fontSize": "0.75rem",
                              "textAlign": "center", "padding": "2rem"})

    ret = pd.Series(returns)
    roll_vol = ret.rolling(window).std() * np.sqrt(252)
    roll_mean = ret.rolling(window).mean() * 252
    roll_sharpe = roll_mean / (roll_vol + 1e-10)

    ts = None
    if timestamps:
        try:
            ts = pd.to_datetime(timestamps[:len(returns)]).tolist()
        except Exception:
            ts = None
    x = ts or list(range(len(returns)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=roll_vol.tolist(), mode="lines", name=f"Vol ({window}d)",
        line=dict(color=C["warning"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=roll_sharpe.tolist(), mode="lines", name=f"Sharpe ({window}d)",
        line=dict(color=C["accent"], width=2), yaxis="y2",
    ))
    fig.update_layout(
        **config.PLOTLY_LAYOUT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11, color=C["text_weak"])),
        yaxis=dict(title=dict(text="Annualized Vol", font=dict(color=C["warning"])),
                   showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        yaxis2=dict(title=dict(text="Sharpe", font=dict(color=C["accent"])),
                    overlaying="y", side="right", showgrid=False,
                    zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
    )
    fig.update_xaxes(**config.PLOTLY_XAXIS)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "320px"})


def _drawdown_duration_chart(equity, timestamps=None):
    """Drawdown periods visualization."""
    if not equity or len(equity) < 2:
        return html.P("No equity data.", style={"color": "var(--text-weak)", "fontSize": "0.75rem",
                                                 "textAlign": "center", "padding": "2rem"})

    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak

    ts = None
    if timestamps:
        try:
            ts = pd.to_datetime(timestamps[:len(equity)]).tolist()
        except Exception:
            ts = None
    x = ts or list(range(len(equity)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=(dd * 100).tolist(), mode="lines", name="Drawdown",
        fill="tozeroy", line=dict(color=C["loss"], width=1.5),
        fillcolor="rgba(239, 68, 68, 0.15)",
        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**config.PLOTLY_LAYOUT, showlegend=False)
    fig.update_xaxes(**config.PLOTLY_XAXIS)
    fig.update_yaxes(**config.PLOTLY_YAXIS, title="Drawdown (%)")
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "280px"})


def _var_chart(returns):
    """Return distribution with VaR/CVaR lines."""
    if len(returns) < 10:
        return html.P("Insufficient data for VaR.", style={"color": "var(--text-weak)",
                       "fontSize": "0.75rem", "textAlign": "center", "padding": "2rem"})

    ret = np.array(returns)
    var_95 = np.percentile(ret, 5)
    cvar_95 = ret[ret <= var_95].mean() if np.any(ret <= var_95) else var_95

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ret.tolist(), nbinsx=40, name="Returns",
        marker_color=C["accent"], opacity=0.6,
    ))
    fig.add_vline(x=var_95, line=dict(color=C["warning"], width=2, dash="dash"),
                  annotation_text=f"VaR 95%: {var_95:.4f}", annotation_position="top left",
                  annotation_font=dict(color=C["warning"], size=11))
    fig.add_vline(x=cvar_95, line=dict(color=C["loss"], width=2, dash="dash"),
                  annotation_text=f"CVaR 95%: {cvar_95:.4f}", annotation_position="top left",
                  annotation_font=dict(color=C["loss"], size=11))

    fig.update_layout(**config.PLOTLY_LAYOUT, showlegend=False)
    fig.update_xaxes(title="Daily Return", showgrid=False)
    fig.update_yaxes(**config.PLOTLY_YAXIS, title="Frequency")
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "300px"})


def _check_thresholds(sharpe, max_dd, volatility, var_95):
    """Compare metrics against RISK_THRESHOLDS, return list of alert dicts."""
    alerts = []

    if sharpe < THRESHOLDS["sharpe_min"]:
        alerts.append({
            "level": "warning",
            "msg": f"Sharpe {sharpe:.2f} below minimum {THRESHOLDS['sharpe_min']}",
        })

    if max_dd < THRESHOLDS["max_drawdown_critical"]:
        alerts.append({
            "level": "critical",
            "msg": f"Max Drawdown {max_dd:.2%} breaches critical {THRESHOLDS['max_drawdown_critical']:.0%}",
        })
    elif max_dd < THRESHOLDS["max_drawdown_warn"]:
        alerts.append({
            "level": "warning",
            "msg": f"Max Drawdown {max_dd:.2%} exceeds warning {THRESHOLDS['max_drawdown_warn']:.0%}",
        })

    if volatility > THRESHOLDS["volatility_warn"]:
        alerts.append({
            "level": "warning",
            "msg": f"Volatility {volatility:.2%} exceeds {THRESHOLDS['volatility_warn']:.0%}",
        })

    if var_95 < THRESHOLDS["var_95_warn"]:
        alerts.append({
            "level": "warning",
            "msg": f"VaR (95%) {var_95:.4f} exceeds warning {THRESHOLDS['var_95_warn']}",
        })

    return alerts


def _alert_banner(alerts):
    """Render colored alert banner from threshold breaches."""
    if not alerts:
        return html.Div()

    has_critical = any(a["level"] == "critical" for a in alerts)
    bg = "var(--loss-soft)" if has_critical else "var(--warning-soft)"
    border_color = "rgba(239, 68, 68, 0.25)" if has_critical else "rgba(245, 158, 11, 0.25)"
    icon_color = "var(--status-loss)" if has_critical else "var(--status-warning)"
    label = "CRITICAL" if has_critical else "WARNING"

    items = [html.Li(a["msg"], style={"fontSize": "0.75rem"}) for a in alerts]

    return html.Div([
        html.Div([
            html.I(className="bi bi-exclamation-triangle-fill",
                   style={"fontSize": "1rem", "color": icon_color}),
            html.Span(label, style={
                "fontFamily": "var(--font-mono)", "fontSize": "0.7rem",
                "fontWeight": "700", "color": icon_color,
            }),
        ], style={"display": "flex", "alignItems": "center", "gap": "0.5rem", "marginBottom": "0.4rem"}),
        html.Ul(items, style={
            "margin": "0", "paddingLeft": "1.5rem", "color": icon_color,
        }),
    ], style={
        "background": bg,
        "border": f"1px solid {border_color}",
        "borderRadius": "var(--radius-md)",
        "padding": "0.85rem 1rem",
        "marginBottom": "1rem",
    })


def _ratio_cls(v, metric=None):
    """Classify ratio value for KPI card color. Considers thresholds."""
    if metric == "sharpe" and v < THRESHOLDS["sharpe_min"]:
        return "warning"
    if metric == "max_drawdown" and v < THRESHOLDS["max_drawdown_critical"]:
        return "loss"
    if metric == "max_drawdown" and v < THRESHOLDS["max_drawdown_warn"]:
        return "warning"
    if metric == "volatility" and v > THRESHOLDS["volatility_warn"]:
        return "warning"
    if metric == "var_95" and v < THRESHOLDS["var_95_warn"]:
        return "warning"
    # Default ratio coloring
    if v >= 1:
        return "profit"
    if v < 0:
        return "loss"
    return ""


# Static layout — dropdown in layout, not callback
layout = html.Div([
    dcc.Interval(id="risk-refresh", interval=config.REFRESH_INTERVAL_MS, n_intervals=0),

    html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Risk Management", className="section-title"),
        ], style={"display": "flex", "alignItems": "center", "gap": "0.6rem"}),
        html.Div([
            html.Span("Backtest", style={
                "fontFamily": "var(--font-sans)", "fontSize": "0.75rem",
                "color": "var(--text-weak)", "fontWeight": "500",
            }),
            dcc.Dropdown(
                id="risk-bt-select",
                placeholder="Select backtest...",
                clearable=False,
                style={"width": "260px"},
                className="once-dropdown",
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "0.6rem"}),
    ], className="section-header", style={"justifyContent": "space-between"}),

    html.Div(id="risk-content"),
])


@callback(
    Output("risk-bt-select", "options"),
    Output("risk-bt-select", "value"),
    Input("risk-refresh", "n_intervals"),
)
def update_bt_options(_n):
    bt_files = loader.list_backtest_results()
    if not bt_files:
        return [], None
    options = [{"label": p.stem.replace("_", " ").title(), "value": p.stem} for p in bt_files]
    return options, bt_files[-1].stem


@callback(
    Output("risk-content", "children"),
    Input("risk-bt-select", "value"),
)
def update_risk(selected_bt):
    if not selected_bt:
        return html.Div(className="empty-state", children=[
            html.I(className="bi bi-shield-check",
                   style={"fontSize": "2.5rem", "color": "var(--text-disabled)"}),
            html.P("No backtest data available.", className="empty-state-text"),
        ])

    bt_data = loader.load_backtest_result(selected_bt)
    if not bt_data or not bt_data.get("metrics"):
        return html.Div()

    m = bt_data["metrics"]
    returns = bt_data.get("returns", [])
    equity = bt_data.get("equity_curve", [])
    timestamps = bt_data.get("timestamps")

    # Compute VaR/CVaR
    ret_arr = np.array(returns) if returns else np.array([0])
    var_95 = float(np.percentile(ret_arr, 5)) if len(ret_arr) > 1 else 0
    cvar_95 = float(ret_arr[ret_arr <= var_95].mean()) if np.any(ret_arr <= var_95) else var_95

    sharpe = m.get("sharpe_ratio", 0)
    sortino = m.get("sortino_ratio", 0)
    calmar = m.get("calmar_ratio", 0)
    max_dd = m.get("max_drawdown", 0)
    volatility = m.get("volatility", 0)
    dd_dur = m.get("max_drawdown_duration", 0)

    # Threshold alerts
    alerts = _check_thresholds(sharpe, max_dd, volatility, var_95)
    alert_banner = _alert_banner(alerts)

    # KPIs
    kpis = html.Div([
        create_kpi_card("Sharpe Ratio", f"{sharpe:.2f}", _ratio_cls(sharpe, "sharpe")),
        create_kpi_card("Sortino Ratio", f"{sortino:.2f}", _ratio_cls(sortino)),
        create_kpi_card("Calmar Ratio", f"{calmar:.2f}", _ratio_cls(calmar)),
        create_kpi_card("VaR (95%)", f"{var_95:.4f}", _ratio_cls(var_95, "var_95")),
        create_kpi_card("CVaR (95%)", f"{cvar_95:.4f}", "loss" if cvar_95 < 0 else ""),
        create_kpi_card("Volatility", f"{volatility:.2%}", _ratio_cls(volatility, "volatility")),
        create_kpi_card("Max Drawdown", f"{max_dd:.2%}", _ratio_cls(max_dd, "max_drawdown")),
        create_kpi_card("DD Duration", f"{dd_dur:.0f} bars"),
    ], className="kpi-grid")

    # Rolling metrics
    rolling_section = html.Div([
        html.Div([
            html.Div(className="section-indicator"),
            html.Span("Rolling Metrics (21-day)", className="section-title"),
        ], className="section-header"),
        html.Div(_rolling_chart(returns, timestamps, window=21), className="chart-card"),
    ], className="section")

    # Drawdown + VaR charts side by side
    charts = html.Div([
        html.Div([
            html.Div([
                html.Div(className="section-indicator"),
                html.Span("Drawdown Periods", className="section-title"),
            ], className="section-header"),
            html.Div(_drawdown_duration_chart(equity, timestamps), className="chart-card"),
        ]),
        html.Div([
            html.Div([
                html.Div(className="section-indicator"),
                html.Span("Value at Risk", className="section-title"),
            ], className="section-header"),
            html.Div(_var_chart(returns), className="chart-card"),
        ]),
    ], className="grid-2", style={"marginBottom": "2rem"})

    return html.Div([alert_banner, kpis, rolling_section, charts])
