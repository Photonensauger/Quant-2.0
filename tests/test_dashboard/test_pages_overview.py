"""Tests for dashboard/pages/overview.py — pure unit tests."""

from dash import html, dcc

from dashboard.pages import overview as overview_mod


class TestEmptyState:
    def test_returns_div(self):
        result = overview_mod._empty_state()
        assert isinstance(result, html.Div)

    def test_has_empty_state_class(self):
        result = overview_mod._empty_state()
        assert "empty-state" in result.className


class TestReturnHistogram:
    def test_returns_graph(self, sample_returns):
        result = overview_mod._return_histogram(sample_returns)
        assert isinstance(result, dcc.Graph)

    def test_single_element(self):
        result = overview_mod._return_histogram([0.01])
        assert isinstance(result, dcc.Graph)
        # With a single element there should be 1 trace (histogram only, no normal)
        assert len(result.figure.data) == 1


class TestUpdateBtOptions:
    def test_empty_returns_empty(self, monkeypatch):
        mock = type("M", (), {"list_backtest_results": lambda self: []})()
        monkeypatch.setattr(overview_mod, "loader", mock)
        options, value = overview_mod.update_bt_options(0)
        assert options == []
        assert value is None

    def test_with_files_returns_options(self, monkeypatch, mock_loader):
        monkeypatch.setattr(overview_mod, "loader", mock_loader)
        options, value = overview_mod.update_bt_options(0)
        assert len(options) == 2
        assert value == "second_run"


class TestUpdateOverview:
    def test_none_returns_empty_state(self, monkeypatch, mock_loader):
        monkeypatch.setattr(overview_mod, "loader", mock_loader)
        result = overview_mod.update_overview(None)
        assert isinstance(result, html.Div)
        assert "empty-state" in result.className

    def test_valid_bt_returns_populated_div(self, monkeypatch, mock_loader):
        monkeypatch.setattr(overview_mod, "loader", mock_loader)
        result = overview_mod.update_overview("test_run")
        assert isinstance(result, html.Div)
        # Should have KPI section, equity section, charts section, dist section
        assert len(result.children) == 4
