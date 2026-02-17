"""Performance visualization dashboard using Plotly.

Generates a self-contained HTML file with six interactive charts that
summarise backtest performance.  No running server required -- just open
the HTML in any modern browser.

Example
-------
>>> from quant.utils.viz import PerformanceVisualizer
>>> viz = PerformanceVisualizer()
>>> path = viz.generate_dashboard(backtest_result, "reports/dashboard.html")
>>> print(f"Dashboard written to {path}")
"""

from __future__ import annotations

import calendar
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger


# ---------------------------------------------------------------------------
# Dark-theme colour palette
# ---------------------------------------------------------------------------

_BG_COLOR = "#0e1117"
_PAPER_COLOR = "#0e1117"
_GRID_COLOR = "#1e2530"
_TEXT_COLOR = "#e0e0e0"
_ACCENT_GREEN = "#00d97e"
_ACCENT_RED = "#e63946"
_ACCENT_BLUE = "#4cc9f0"
_ACCENT_PURPLE = "#b794f6"
_ACCENT_ORANGE = "#ff8c42"
_ACCENT_YELLOW = "#ffd166"
_DRAWDOWN_FILL = "rgba(230, 57, 70, 0.25)"
_EQUITY_FILL = "rgba(0, 217, 126, 0.08)"

_DARK_LAYOUT = dict(
    paper_bgcolor=_PAPER_COLOR,
    plot_bgcolor=_BG_COLOR,
    font=dict(family="Inter, Segoe UI, Roboto, sans-serif", color=_TEXT_COLOR, size=12),
    title_font=dict(size=16, color=_TEXT_COLOR),
    xaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
    yaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_COLOR)),
    margin=dict(l=60, r=30, t=60, b=40),
)


# ---------------------------------------------------------------------------
# Helper: extract common series from a backtest result dict
# ---------------------------------------------------------------------------

def _extract_series(result: dict[str, Any]) -> dict[str, pd.Series]:
    """Normalise a backtest result dict into canonical pandas Series.

    The backtest engine is expected to produce a dict with at least some of:

    * ``equity_curve``  -- pd.Series or list of portfolio values indexed by
      datetime.
    * ``returns``       -- pd.Series of period returns.
    * ``trades``        -- pd.DataFrame with columns ``pnl``,
      ``entry_time``, ``exit_time`` (or ``duration``).

    Missing keys are derived when possible (e.g. returns from the equity
    curve).
    """
    out: dict[str, Any] = {}

    # Equity curve -----------------------------------------------------------
    equity = result.get("equity_curve")
    if equity is not None:
        if not isinstance(equity, pd.Series):
            equity = pd.Series(equity)
        if not isinstance(equity.index, pd.DatetimeIndex):
            equity.index = pd.to_datetime(equity.index)
        out["equity"] = equity

    # Returns ----------------------------------------------------------------
    returns = result.get("returns")
    if returns is not None:
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        if not isinstance(returns.index, pd.DatetimeIndex) and "equity" in out:
            returns.index = out["equity"].index[: len(returns)]
        out["returns"] = returns
    elif "equity" in out:
        out["returns"] = out["equity"].pct_change().dropna()

    # Drawdown series --------------------------------------------------------
    if "equity" in out:
        eq = out["equity"]
        running_max = eq.cummax()
        dd = (eq - running_max) / running_max
        out["drawdown"] = dd

    # Trades -----------------------------------------------------------------
    trades = result.get("trades")
    if trades is not None:
        if isinstance(trades, list):
            trades = pd.DataFrame(trades)
        out["trades"] = trades

    # Metrics ----------------------------------------------------------------
    metrics = result.get("metrics")
    if metrics is not None:
        out["metrics"] = metrics

    return out


# ---------------------------------------------------------------------------
# PerformanceVisualizer
# ---------------------------------------------------------------------------

class PerformanceVisualizer:
    """Create interactive HTML performance dashboards from backtest results.

    All charts use a consistent professional dark theme and are fully
    self-contained (no external JS/CSS dependencies).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dashboard(
        self,
        backtest_result: dict[str, Any],
        output_path: str | Path = "reports/dashboard.html",
    ) -> str:
        """Build a six-panel HTML dashboard and write it to *output_path*.

        Charts
        ------
        1. Equity curve with drawdown shading
        2. Returns distribution histogram with normal overlay
        3. Monthly returns heatmap
        4. Drawdown area chart
        5. Trade analysis -- PnL vs duration scatter
        6. Rolling Sharpe ratio

        Parameters
        ----------
        backtest_result : dict
            Output of the backtesting engine.  Expected keys:
            ``equity_curve``, ``returns``, ``trades``, ``metrics``.
        output_path : str | Path
            Destination file path.  Parent directories are created
            automatically.

        Returns
        -------
        str
            Absolute path of the written HTML file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = _extract_series(backtest_result)
        logger.info("Generating performance dashboard ...")

        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Equity Curve",
                "Returns Distribution",
                "Monthly Returns Heatmap",
                "Drawdown",
                "Trade Analysis (PnL vs Duration)",
                "Rolling Sharpe Ratio",
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.07,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
            ],
        )

        # 1. Equity curve
        self._add_equity_curve(fig, data, row=1, col=1)

        # 2. Returns distribution
        self._add_returns_distribution(fig, data, row=1, col=2)

        # 3. Monthly returns heatmap
        self._add_monthly_heatmap(fig, data, row=2, col=1)

        # 4. Drawdown chart
        self._add_drawdown_chart(fig, data, row=2, col=2)

        # 5. Trade analysis
        self._add_trade_analysis(fig, data, row=3, col=1)

        # 6. Rolling Sharpe
        self._add_rolling_sharpe(fig, data, row=3, col=2)

        # ---- Global layout ------------------------------------------------
        fig.update_layout(
            height=1200,
            width=1600,
            showlegend=False,
            title_text="Backtest Performance Dashboard",
            title_x=0.5,
            title_font=dict(size=22, color=_ACCENT_BLUE),
            paper_bgcolor=_PAPER_COLOR,
            plot_bgcolor=_BG_COLOR,
            font=dict(
                family="Inter, Segoe UI, Roboto, sans-serif",
                color=_TEXT_COLOR,
                size=11,
            ),
            margin=dict(l=60, r=40, t=90, b=50),
        )

        # Apply dark grid to every axis
        for ax_name in fig.to_dict()["layout"]:
            if ax_name.startswith("xaxis") or ax_name.startswith("yaxis"):
                fig.update_layout(
                    **{
                        ax_name: dict(
                            gridcolor=_GRID_COLOR,
                            zerolinecolor=_GRID_COLOR,
                        )
                    }
                )

        # ---- Metrics summary banner ----------------------------------------
        if "metrics" in data:
            annotation_text = self._build_banner_text(data["metrics"])
            fig.add_annotation(
                text=annotation_text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.02,
                showarrow=False,
                font=dict(size=11, color=_TEXT_COLOR, family="monospace"),
                align="center",
                bgcolor="rgba(30,37,48,0.85)",
                bordercolor=_GRID_COLOR,
                borderwidth=1,
                borderpad=6,
            )

        # ---- Write standalone HTML -----------------------------------------
        fig.write_html(
            str(output_path),
            include_plotlyjs=True,
            full_html=True,
            config={
                "displayModeBar": True,
                "scrollZoom": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "width": 1920,
                    "height": 1080,
                },
            },
        )

        abs_path = str(output_path.resolve())
        logger.success("Dashboard saved to {}", abs_path)
        return abs_path

    def generate_summary_table(self, metrics: dict[str, Any]) -> str:
        """Render a key-metric summary as a self-contained HTML table.

        Parameters
        ----------
        metrics : dict
            Flat dictionary of metric names to values (int, float, str, ...).

        Returns
        -------
        str
            HTML string containing a ``<table>`` element styled with the
            dashboard dark theme.
        """
        rows_html = ""
        for key, value in metrics.items():
            display_key = key.replace("_", " ").title()
            formatted = self._format_metric(value)
            rows_html += (
                f"  <tr>\n"
                f"    <td style='padding:8px 16px; border-bottom:1px solid {_GRID_COLOR}; "
                f"color:{_ACCENT_BLUE}; font-weight:600;'>{display_key}</td>\n"
                f"    <td style='padding:8px 16px; border-bottom:1px solid {_GRID_COLOR}; "
                f"text-align:right; font-family:monospace;'>{formatted}</td>\n"
                f"  </tr>\n"
            )

        html = (
            f"<table style='border-collapse:collapse; background:{_BG_COLOR}; "
            f"color:{_TEXT_COLOR}; font-family:Inter,Segoe UI,Roboto,sans-serif; "
            f"font-size:13px; border-radius:8px; overflow:hidden; "
            f"box-shadow:0 2px 12px rgba(0,0,0,0.4);'>\n"
            f"  <thead>\n"
            f"    <tr style='background:{_GRID_COLOR};'>\n"
            f"      <th style='padding:10px 16px; text-align:left; "
            f"color:{_ACCENT_BLUE}; font-size:14px;' colspan='2'>"
            f"Performance Summary</th>\n"
            f"    </tr>\n"
            f"  </thead>\n"
            f"  <tbody>\n"
            f"{rows_html}"
            f"  </tbody>\n"
            f"</table>"
        )
        return html

    # ------------------------------------------------------------------
    # Private: individual chart builders
    # ------------------------------------------------------------------

    def _add_equity_curve(
        self,
        fig: go.Figure,
        data: dict[str, Any],
        row: int,
        col: int,
    ) -> None:
        """Panel 1 -- Equity curve with drawdown shading."""
        equity = data.get("equity")
        if equity is None:
            self._add_placeholder(fig, "No equity data", row, col)
            return

        # Equity line
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                line=dict(color=_ACCENT_GREEN, width=1.8),
                fill="tozeroy",
                fillcolor=_EQUITY_FILL,
                name="Equity",
                hovertemplate="Date: %{x}<br>Equity: $%{y:,.0f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Drawdown shading as secondary y on same panel
        dd = data.get("drawdown")
        if dd is not None:
            running_max = equity.cummax()
            fig.add_trace(
                go.Scatter(
                    x=running_max.index,
                    y=running_max.values,
                    mode="lines",
                    line=dict(color=_ACCENT_BLUE, width=0.8, dash="dot"),
                    name="High-water",
                    hovertemplate="HWM: $%{y:,.0f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    def _add_returns_distribution(
        self,
        fig: go.Figure,
        data: dict[str, Any],
        row: int,
        col: int,
    ) -> None:
        """Panel 2 -- Returns histogram with normal overlay."""
        returns = data.get("returns")
        if returns is None:
            self._add_placeholder(fig, "No return data", row, col)
            return

        ret = returns.dropna()

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=ret.values,
                nbinsx=80,
                marker_color=_ACCENT_PURPLE,
                opacity=0.75,
                name="Returns",
                histnorm="probability density",
                hovertemplate="Return: %{x:.4f}<br>Density: %{y:.2f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Normal overlay
        mu, sigma = float(ret.mean()), float(ret.std())
        if sigma > 0:
            x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            normal_pdf = (
                1.0
                / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
            )
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=normal_pdf,
                    mode="lines",
                    line=dict(color=_ACCENT_ORANGE, width=2),
                    name="Normal",
                ),
                row=row,
                col=col,
            )

        # Vertical mean line
        fig.add_vline(
            x=mu,
            line_dash="dash",
            line_color=_ACCENT_YELLOW,
            line_width=1,
            row=row,
            col=col,
        )

    def _add_monthly_heatmap(
        self,
        fig: go.Figure,
        data: dict[str, Any],
        row: int,
        col: int,
    ) -> None:
        """Panel 3 -- Monthly returns heatmap."""
        returns = data.get("returns")
        if returns is None or not isinstance(returns.index, pd.DatetimeIndex):
            self._add_placeholder(fig, "No monthly data", row, col)
            return

        # Aggregate to monthly returns
        monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        if monthly.empty:
            self._add_placeholder(fig, "Insufficient data for monthly heatmap", row, col)
            return

        monthly_df = monthly.to_frame("return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month

        pivot = monthly_df.pivot_table(
            values="return", index="year", columns="month", aggfunc="sum"
        )

        # Ensure all 12 months
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = np.nan
        pivot = pivot[sorted(pivot.columns)]

        month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
        year_labels = [str(y) for y in pivot.index]

        # Colour scale: red for losses, green for gains
        z_vals = pivot.values * 100  # convert to percentage
        text_vals = [
            [
                f"{v:.1f}%" if not np.isnan(v) else ""
                for v in row_vals
            ]
            for row_vals in z_vals
        ]

        fig.add_trace(
            go.Heatmap(
                z=z_vals,
                x=month_labels,
                y=year_labels,
                text=text_vals,
                texttemplate="%{text}",
                textfont=dict(size=10, color=_TEXT_COLOR),
                colorscale=[
                    [0.0, _ACCENT_RED],
                    [0.5, _BG_COLOR],
                    [1.0, _ACCENT_GREEN],
                ],
                zmid=0,
                showscale=True,
                colorbar=dict(
                    title=dict(text="%", font=dict(color=_TEXT_COLOR)),
                    tickfont=dict(color=_TEXT_COLOR),
                    len=0.3,
                    y=0.5,
                ),
                hovertemplate=(
                    "Year: %{y}<br>Month: %{x}<br>"
                    "Return: %{text}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    def _add_drawdown_chart(
        self,
        fig: go.Figure,
        data: dict[str, Any],
        row: int,
        col: int,
    ) -> None:
        """Panel 4 -- Drawdown area chart."""
        dd = data.get("drawdown")
        if dd is None:
            self._add_placeholder(fig, "No drawdown data", row, col)
            return

        dd_pct = dd * 100

        fig.add_trace(
            go.Scatter(
                x=dd_pct.index,
                y=dd_pct.values,
                mode="lines",
                line=dict(color=_ACCENT_RED, width=1.2),
                fill="tozeroy",
                fillcolor=_DRAWDOWN_FILL,
                name="Drawdown",
                hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Annotate max drawdown
        if len(dd_pct) > 0:
            worst_idx = dd_pct.idxmin()
            worst_val = float(dd_pct.loc[worst_idx])
            fig.add_annotation(
                x=worst_idx,
                y=worst_val,
                text=f"Max DD: {worst_val:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=_ACCENT_RED,
                font=dict(color=_ACCENT_RED, size=10),
                bgcolor="rgba(14,17,23,0.8)",
                bordercolor=_ACCENT_RED,
                borderwidth=1,
                xref=f"x{self._axis_idx(row, col)}",
                yref=f"y{self._axis_idx(row, col)}",
            )

    def _add_trade_analysis(
        self,
        fig: go.Figure,
        data: dict[str, Any],
        row: int,
        col: int,
    ) -> None:
        """Panel 5 -- PnL vs trade duration scatter."""
        trades = data.get("trades")
        if trades is None or (isinstance(trades, pd.DataFrame) and trades.empty):
            self._add_placeholder(fig, "No trade data", row, col)
            return

        df = trades.copy()

        # Compute duration if not present
        if "duration" not in df.columns:
            if "entry_time" in df.columns and "exit_time" in df.columns:
                df["entry_time"] = pd.to_datetime(df["entry_time"])
                df["exit_time"] = pd.to_datetime(df["exit_time"])
                df["duration"] = (
                    df["exit_time"] - df["entry_time"]
                ).dt.total_seconds() / 3600.0  # hours
            else:
                df["duration"] = range(len(df))

        if "pnl" not in df.columns:
            self._add_placeholder(fig, "No PnL column in trades", row, col)
            return

        is_win = df["pnl"] >= 0
        colors = [_ACCENT_GREEN if w else _ACCENT_RED for w in is_win]
        sizes = np.clip(np.abs(df["pnl"].values) / (np.abs(df["pnl"]).max() + 1e-9) * 15 + 4, 4, 20)

        fig.add_trace(
            go.Scatter(
                x=df["duration"],
                y=df["pnl"],
                mode="markers",
                marker=dict(
                    color=colors,
                    size=sizes,
                    opacity=0.7,
                    line=dict(width=0.5, color=_TEXT_COLOR),
                ),
                name="Trades",
                hovertemplate=(
                    "Duration: %{x:.1f}h<br>"
                    "PnL: $%{y:,.2f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # Zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=_TEXT_COLOR,
            line_width=0.8,
            row=row,
            col=col,
        )

        # Axis labels
        ax_idx = self._axis_idx(row, col)
        fig.update_xaxes(title_text="Duration (hours)", row=row, col=col)
        fig.update_yaxes(title_text="PnL ($)", row=row, col=col)

    def _add_rolling_sharpe(
        self,
        fig: go.Figure,
        data: dict[str, Any],
        row: int,
        col: int,
        window: int = 63,
    ) -> None:
        """Panel 6 -- Rolling Sharpe ratio (annualised, 63-bar window)."""
        returns = data.get("returns")
        if returns is None or len(returns) < window:
            self._add_placeholder(fig, "Insufficient data for rolling Sharpe", row, col)
            return

        roll_mean = returns.rolling(window=window).mean()
        roll_std = returns.rolling(window=window).std()
        # Annualise assuming 252 trading days
        rolling_sharpe = (roll_mean / (roll_std + 1e-10)) * np.sqrt(252)
        rolling_sharpe = rolling_sharpe.dropna()

        # Color based on value: green above 1, yellow 0-1, red below 0
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode="lines",
                line=dict(color=_ACCENT_BLUE, width=1.5),
                name="Rolling Sharpe",
                hovertemplate="Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Reference lines
        for level, color, label in [
            (0.0, _ACCENT_RED, None),
            (1.0, _ACCENT_YELLOW, "Sharpe = 1"),
            (2.0, _ACCENT_GREEN, "Sharpe = 2"),
        ]:
            fig.add_hline(
                y=level,
                line_dash="dot",
                line_color=color,
                line_width=0.8,
                row=row,
                col=col,
            )

        # Fill between zero and Sharpe curve
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=[0] * len(rolling_sharpe),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _axis_idx(row: int, col: int, n_cols: int = 2) -> str:
        """Plotly axis number string for a given subplot position.

        Plotly uses ``x``, ``x2``, ``x3``, ... for the first, second,
        third axes.  Subplot (1,1) is ``x`` (no suffix), (1,2) is ``x2``,
        etc.
        """
        idx = (row - 1) * n_cols + col
        return "" if idx == 1 else str(idx)

    @staticmethod
    def _add_placeholder(
        fig: go.Figure, message: str, row: int, col: int
    ) -> None:
        """Add a text annotation when data is unavailable for a panel."""
        fig.add_annotation(
            text=f"<i>{message}</i>",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color=_TEXT_COLOR),
            row=row,
            col=col,
        )

    @staticmethod
    def _format_metric(value: Any) -> str:
        """Format a single metric value for display."""
        if isinstance(value, float):
            if abs(value) < 1:
                return f"{value:+.4f}"
            if abs(value) < 1000:
                return f"{value:+.2f}"
            return f"{value:+,.0f}"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M")
        return str(value)

    @staticmethod
    def _build_banner_text(metrics: dict[str, Any]) -> str:
        """Build a one-line summary string for the dashboard banner."""
        parts: list[str] = []

        key_map = {
            "total_return": ("Return", lambda v: f"{v * 100:+.2f}%"),
            "sharpe_ratio": ("Sharpe", lambda v: f"{v:.2f}"),
            "max_drawdown": ("Max DD", lambda v: f"{v * 100:.1f}%"),
            "win_rate": ("Win Rate", lambda v: f"{v * 100:.1f}%"),
            "total_trades": ("Trades", lambda v: f"{v:,}"),
            "profit_factor": ("PF", lambda v: f"{v:.2f}"),
            "calmar_ratio": ("Calmar", lambda v: f"{v:.2f}"),
            "annual_return": ("Ann. Return", lambda v: f"{v * 100:+.1f}%"),
            "annual_volatility": ("Ann. Vol", lambda v: f"{v * 100:.1f}%"),
            "sortino_ratio": ("Sortino", lambda v: f"{v:.2f}"),
        }

        for key, (label, fmt) in key_map.items():
            if key in metrics:
                try:
                    parts.append(f"{label}: {fmt(metrics[key])}")
                except (ValueError, TypeError):
                    parts.append(f"{label}: {metrics[key]}")

        return "   |   ".join(parts) if parts else ""
