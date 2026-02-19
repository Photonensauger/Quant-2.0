"""Tests for dashboard/pages/positions.py — pure unit tests."""

from unittest.mock import MagicMock

from dash import html, no_update

from dashboard.pages import positions as pos_mod


class TestUpdateBtOptions:
    def test_empty_files(self, monkeypatch):
        mock = MagicMock()
        mock.list_backtest_results.return_value = []
        monkeypatch.setattr(pos_mod, "loader", mock)
        options, value = pos_mod.update_bt_options(0)
        assert options == []
        assert value is None


class TestUpdatePositions:
    def test_none_bt_returns_div(self, monkeypatch, mock_loader):
        mock_loader.list_backtest_results.return_value = []
        mock_loader.load_backtest_result.return_value = None
        monkeypatch.setattr(pos_mod, "loader", mock_loader)
        result = pos_mod.update_positions(None)
        assert isinstance(result, html.Div)

    def test_no_trades(self, monkeypatch, mock_loader):
        bt_data = {"trades": [], "metrics": {}}
        mock_loader.load_backtest_result.return_value = bt_data
        monkeypatch.setattr(pos_mod, "loader", mock_loader)
        result = pos_mod.update_positions("test_run")
        assert isinstance(result, html.Div)

    def test_with_trades(self, monkeypatch, mock_loader):
        monkeypatch.setattr(pos_mod, "loader", mock_loader)
        result = pos_mod.update_positions("test_run")
        assert isinstance(result, html.Div)
        # Should contain active + trade_section + charts
        assert len(result.children) == 3


class TestExportTradesCsv:
    def test_zero_clicks_returns_no_update(self, monkeypatch, mock_loader):
        monkeypatch.setattr(pos_mod, "loader", mock_loader)
        result = pos_mod.export_trades_csv(0, "test_run")
        assert result is no_update

    def test_valid_returns_csv(self, monkeypatch, mock_loader):
        monkeypatch.setattr(pos_mod, "loader", mock_loader)
        result = pos_mod.export_trades_csv(1, "test_run")
        assert isinstance(result, dict)
        assert "content" in result
        assert result["filename"].endswith(".csv")


class TestExportTradesJson:
    def test_no_bt_returns_no_update(self, monkeypatch, mock_loader):
        monkeypatch.setattr(pos_mod, "loader", mock_loader)
        result = pos_mod.export_trades_json(1, None)
        assert result is no_update

    def test_valid_returns_json(self, monkeypatch, mock_loader):
        monkeypatch.setattr(pos_mod, "loader", mock_loader)
        result = pos_mod.export_trades_json(1, "test_run")
        assert isinstance(result, dict)
        assert result["filename"].endswith(".json")
