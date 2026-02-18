"""Once UI styled drawdown underwater chart."""

import numpy as np
import plotly.graph_objects as go
from dash import dcc

from dashboard.config import COLORS, PLOTLY_LAYOUT, PLOTLY_XAXIS, PLOTLY_YAXIS


def create_drawdown_chart(equity_curve, timestamps=None):
    equity = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / np.where(running_max == 0, 1, running_max) * 100

    x_axis = timestamps if timestamps is not None else list(range(len(equity_curve)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, y=drawdown,
        mode="lines", fill="tozeroy", name="Drawdown",
        line=dict(color=COLORS["loss"], width=1),
        fillcolor="rgba(239, 68, 68, 0.15)",
    ))

    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
    fig.update_xaxes(**PLOTLY_XAXIS)
    fig.update_yaxes(**PLOTLY_YAXIS, title="Drawdown (%)", ticksuffix="%")

    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "280px"})
