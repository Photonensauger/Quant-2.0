"""Tests for dashboard/components/ — pure unit tests, no Dash app needed."""

import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table

from dashboard.components.navbar import create_header, update_active_nav, NAV_ITEMS
from dashboard.components.kpi_card import create_kpi_card
from dashboard.components.equity_chart import create_equity_chart
from dashboard.components.drawdown_chart import create_drawdown_chart
from dashboard.components.heatmap import create_monthly_returns_heatmap
from dashboard.components.trade_table import create_trade_table


# ── navbar ────────────────────────────────────────────────────────────────

class TestCreateHeader:
    def test_returns_header(self):
        result = create_header()
        assert isinstance(result, html.Header)

    def test_contains_title(self):
        header = create_header()
        # header-left is the first child, which contains H1
        header_left = header.children[0]
        h1 = header_left.children[0]
        assert isinstance(h1, html.H1)
        assert h1.children == "Quant 2.0"

    def test_nav_pill_count(self):
        header = create_header()
        header_right = header.children[1]
        nav_pills_container = header_right.children[0]
        assert len(nav_pills_container.children) == len(NAV_ITEMS)

    def test_nav_pill_hrefs(self):
        header = create_header()
        header_right = header.children[1]
        nav_pills_container = header_right.children[0]
        hrefs = [link.href for link in nav_pills_container.children]
        expected = [item["href"] for item in NAV_ITEMS]
        assert hrefs == expected


class TestUpdateActiveNav:
    def test_root_path(self):
        classes = update_active_nav("/")
        assert classes[0] == "nav-pill active"
        assert all(c == "nav-pill" for c in classes[1:])

    def test_positions_path(self):
        classes = update_active_nav("/positions")
        assert classes[1] == "nav-pill active"
        assert classes[0] == "nav-pill"

    def test_prefix_match(self):
        classes = update_active_nav("/backtest/detail")
        assert classes[2] == "nav-pill active"

    def test_none_pathname(self):
        classes = update_active_nav(None)
        assert all(c == "nav-pill" for c in classes)


# ── kpi_card ──────────────────────────────────────────────────────────────

class TestKpiCard:
    def test_returns_div(self):
        card = create_kpi_card("Return", "15%")
        assert isinstance(card, html.Div)
        assert "kpi-card" in card.className

    def test_label_and_value(self):
        card = create_kpi_card("Sharpe", "1.50")
        label_div = card.children[0]
        value_div = card.children[1]
        assert label_div.children == "Sharpe"
        assert value_div.children == "1.50"

    def test_color_class(self):
        card = create_kpi_card("PnL", "+$50", color_class="profit")
        value_div = card.children[1]
        assert "profit" in value_div.className

    def test_subtitle(self):
        card = create_kpi_card("Vol", "18%", subtitle="annualized")
        assert len(card.children) == 3
        assert card.children[2].children == "annualized"

    def test_no_subtitle(self):
        card = create_kpi_card("Vol", "18%")
        assert len(card.children) == 2


# ── equity_chart ──────────────────────────────────────────────────────────

class TestEquityChart:
    def test_returns_graph(self, sample_equity_curve):
        result = create_equity_chart(sample_equity_curve)
        assert isinstance(result, dcc.Graph)

    def test_single_trace_without_benchmark(self, sample_equity_curve):
        graph = create_equity_chart(sample_equity_curve)
        fig = graph.figure
        assert len(fig.data) == 1
        assert fig.data[0].name == "Portfolio"

    def test_two_traces_with_benchmark(self, sample_equity_curve):
        benchmark = [v * 0.95 for v in sample_equity_curve]
        graph = create_equity_chart(sample_equity_curve, benchmark_data=benchmark)
        fig = graph.figure
        assert len(fig.data) == 2
        assert fig.data[1].name == "Benchmark"

    def test_timestamps_used_as_x(self, sample_equity_curve):
        timestamps = list(range(len(sample_equity_curve)))
        graph = create_equity_chart(sample_equity_curve, timestamps=timestamps)
        fig = graph.figure
        assert list(fig.data[0].x) == timestamps


# ── drawdown_chart ────────────────────────────────────────────────────────

class TestDrawdownChart:
    def test_returns_graph(self, sample_equity_curve):
        result = create_drawdown_chart(sample_equity_curve)
        assert isinstance(result, dcc.Graph)

    def test_values_non_positive(self, sample_equity_curve):
        graph = create_drawdown_chart(sample_equity_curve)
        y_values = graph.figure.data[0].y
        assert all(v <= 0 for v in y_values)

    def test_zero_at_initial_peak(self):
        equity = [100, 110, 105, 115, 110]
        graph = create_drawdown_chart(equity)
        y = graph.figure.data[0].y
        assert y[0] == 0.0
        assert y[1] == 0.0  # new peak
        assert y[3] == 0.0  # new peak at 115


# ── heatmap ───────────────────────────────────────────────────────────────

class TestHeatmap:
    def test_returns_graph(self, sample_returns_series):
        result = create_monthly_returns_heatmap(sample_returns_series)
        assert isinstance(result, dcc.Graph)

    def test_heatmap_trace_type(self, sample_returns_series):
        graph = create_monthly_returns_heatmap(sample_returns_series)
        trace = graph.figure.data[0]
        assert isinstance(trace, go.Heatmap)

    def test_empty_series_has_annotation(self):
        empty = pd.Series(dtype=float)
        graph = create_monthly_returns_heatmap(empty)
        annotations = graph.figure.layout.annotations
        assert len(annotations) == 1
        assert "No return data" in annotations[0].text

    def test_x_axis_has_twelve_months(self, sample_returns_series):
        graph = create_monthly_returns_heatmap(sample_returns_series)
        trace = graph.figure.data[0]
        assert list(trace.x) == [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]


# ── trade_table ───────────────────────────────────────────────────────────

class TestTradeTable:
    def test_returns_datatable(self, sample_trade_log):
        result = create_trade_table(sample_trade_log)
        assert isinstance(result, dash_table.DataTable)

    def test_empty_produces_columns_no_data(self):
        table = create_trade_table([])
        assert len(table.columns) == 9
        assert table.data == []

    def test_trades_populated(self, sample_trade_log):
        table = create_trade_table(sample_trade_log)
        assert len(table.data) == 2
        assert table.data[0]["Asset"] == "AAPL"

    def test_side_uppercased(self, sample_trade_log):
        table = create_trade_table(sample_trade_log)
        assert table.data[0]["Side"] == "LONG"
        assert table.data[1]["Side"] == "SHORT"

    def test_symbol_key_fallback(self):
        trades = [{"symbol": "GOOG", "side": "long", "pnl": 10}]
        table = create_trade_table(trades)
        assert table.data[0]["Asset"] == "GOOG"
