"""Once UI styled monthly returns heatmap."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc

from dashboard.config import COLORS, PLOTLY_LAYOUT, PLOTLY_XAXIS, PLOTLY_YAXIS

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def create_monthly_returns_heatmap(returns_series):
    if returns_series.empty:
        fig = go.Figure()
        fig.update_layout(
            **PLOTLY_LAYOUT,
            annotations=[dict(
                text="No return data available",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(color=COLORS["text_weak"], size=13),
            )],
        )
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    monthly = returns_series.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = pivot.pivot(index="year", columns="month", values="return")
    pivot = pivot.reindex(columns=range(1, 13))

    years = [str(y) for y in pivot.index]
    z_values = np.where(pivot.isna(), None, np.round(pivot.values, 2))

    text_matrix = [
        [f"{val:.2f}%" if val is not None else "" for val in row]
        for row in z_values
    ]

    # Scale row height based on number of years (min 80px per row)
    n_years = len(years)
    row_h = max(80, 60)
    chart_h = max(280, n_years * row_h + 80)

    fig = go.Figure(data=go.Heatmap(
        z=z_values, x=MONTH_LABELS, y=years,
        colorscale=[[0, COLORS["loss"]], [0.5, COLORS["bg_overlay"]], [1, COLORS["profit"]]],
        zmid=0,
        text=text_matrix, texttemplate="%{text}",
        textfont=dict(size=10, family="JetBrains Mono"),
        hovertemplate="%{y} %{x}: %{text}<extra></extra>",
        colorbar=dict(
            title=dict(text="Return", font=dict(size=11)),
            ticksuffix="%", len=0.6, thickness=12,
            x=1.02, xpad=5,
        ),
        xgap=2, ygap=2,
    ))

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(margin=dict(l=0, r=60, t=30, b=0))
    fig.update_xaxes(side="top", showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=11))

    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": f"{chart_h}px"})
