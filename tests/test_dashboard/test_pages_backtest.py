"""Tests for dashboard/pages/backtest.py — pure unit tests."""

from unittest.mock import MagicMock

from dash import html, dcc, no_update

from dashboard.pages import backtest as bt_mod


# ── _metrics_table ────────────────────────────────────────────────────────

class TestMetricsTable:
    def test_empty_metrics(self):
        table = bt_mod._metrics_table({})
        assert isinstance(table, html.Table)
        assert table.children == []

    def test_known_keys_produce_rows(self):
        metrics = {"sharpe_ratio": 1.5, "total_return": 0.15}
        table = bt_mod._metrics_table(metrics)
        rows = table.children
        assert len(rows) == 2

    def test_unknown_keys_skipped(self):
        metrics = {"made_up_metric": 99}
        table = bt_mod._metrics_table(metrics)
        assert table.children == []


# ── _comparison_chart ─────────────────────────────────────────────────────

class TestComparisonChart:
    def test_returns_graph(self, sample_bt_data):
        all_results = {"run_a": sample_bt_data, "run_b": sample_bt_data}
        result = bt_mod._comparison_chart(all_results, ["run_a", "run_b"])
        assert isinstance(result, dcc.Graph)

    def test_one_trace_per_key(self, sample_bt_data):
        all_results = {"a": sample_bt_data, "b": sample_bt_data, "c": sample_bt_data}
        graph = bt_mod._comparison_chart(all_results, ["a", "c"])
        assert len(graph.figure.data) == 2


# ── update_bt_options ─────────────────────────────────────────────────────

class TestBtOptions:
    def test_empty_returns_triple(self, monkeypatch):
        mock = MagicMock()
        mock.list_backtest_results.return_value = []
        monkeypatch.setattr(bt_mod, "loader", mock)
        opts, val, compare = bt_mod.update_bt_options(0)
        assert opts == []
        assert val is None
        assert compare == []


# ── update_detail ─────────────────────────────────────────────────────────

class TestUpdateDetail:
    def test_none_returns_empty_state(self, monkeypatch, mock_loader):
        monkeypatch.setattr(bt_mod, "loader", mock_loader)
        result = bt_mod.update_detail(None)
        assert isinstance(result, html.Div)
        assert "empty-state" in result.className

    def test_valid_bt_returns_populated(self, monkeypatch, mock_loader):
        monkeypatch.setattr(bt_mod, "loader", mock_loader)
        result = bt_mod.update_detail("test_run")
        assert isinstance(result, html.Div)
        # KPIs + equity + grid-2 (metrics + trades)
        assert len(result.children) == 3


# ── update_comparison ─────────────────────────────────────────────────────

class TestUpdateComparison:
    def test_none_returns_hint(self, monkeypatch, mock_loader):
        monkeypatch.setattr(bt_mod, "loader", mock_loader)
        result = bt_mod.update_comparison(None)
        assert isinstance(result, html.P)

    def test_single_key_returns_hint(self, monkeypatch, mock_loader):
        monkeypatch.setattr(bt_mod, "loader", mock_loader)
        result = bt_mod.update_comparison(["one"])
        assert isinstance(result, html.P)

    def test_two_keys_returns_chart(self, monkeypatch, mock_loader):
        monkeypatch.setattr(bt_mod, "loader", mock_loader)
        result = bt_mod.update_comparison(["test_run", "second_run"])
        assert isinstance(result, html.Div)


# ── export callbacks ──────────────────────────────────────────────────────

class TestExportMetricsJson:
    def test_no_clicks(self, monkeypatch, mock_loader):
        monkeypatch.setattr(bt_mod, "loader", mock_loader)
        assert bt_mod.export_metrics_json(0, "test_run") is no_update

    def test_valid(self, monkeypatch, mock_loader):
        monkeypatch.setattr(bt_mod, "loader", mock_loader)
        result = bt_mod.export_metrics_json(1, "test_run")
        assert isinstance(result, dict)
        assert result["filename"].endswith("_metrics.json")


class TestExportTradesCsv:
    def test_no_clicks(self, monkeypatch, mock_loader):
        monkeypatch.setattr(bt_mod, "loader", mock_loader)
        assert bt_mod.export_trades_csv(0, "test_run") is no_update

    def test_valid(self, monkeypatch, mock_loader):
        monkeypatch.setattr(bt_mod, "loader", mock_loader)
        result = bt_mod.export_trades_csv(1, "test_run")
        assert isinstance(result, dict)
        assert result["filename"].endswith("_trades.csv")


class TestExportTradesJson:
    def test_no_bt(self, monkeypatch, mock_loader):
        monkeypatch.setattr(bt_mod, "loader", mock_loader)
        assert bt_mod.export_trades_json(0, None) is no_update

    def test_valid(self, monkeypatch, mock_loader):
        monkeypatch.setattr(bt_mod, "loader", mock_loader)
        result = bt_mod.export_trades_json(1, "test_run")
        assert isinstance(result, dict)
        assert result["filename"].endswith("_trades.json")
