"""Once UI styled equity curve chart."""

import plotly.graph_objects as go
from dash import dcc

from dashboard.config import COLORS, PLOTLY_LAYOUT, PLOTLY_XAXIS, PLOTLY_YAXIS


def create_equity_chart(equity_data, benchmark_data=None, timestamps=None):
    x_axis = timestamps if timestamps is not None else list(range(len(equity_data)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, y=equity_data,
        mode="lines", name="Portfolio",
        line=dict(color=COLORS["accent"], width=2),
    ))

    if benchmark_data is not None:
        fig.add_trace(go.Scatter(
            x=x_axis, y=benchmark_data,
            mode="lines", name="Benchmark",
            line=dict(color=COLORS["text_weak"], width=1.5, dash="dot"),
        ))
        fig.update_layout(showlegend=True, legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color=COLORS["text_weak"]),
        ))

    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
    fig.update_xaxes(**PLOTLY_XAXIS)
    fig.update_yaxes(**PLOTLY_YAXIS, title="Equity")

    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "350px"})
