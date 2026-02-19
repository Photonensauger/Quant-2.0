"""Tests for dashboard/pages/models.py — pure unit tests."""

from pathlib import Path

from dash import html, dcc

from dashboard.pages import models as models_mod


# ── _get_checkpoint_info ──────────────────────────────────────────────────

class TestGetCheckpointInfo:
    def test_nonexistent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(models_mod, "_CHECKPOINT_DIR", tmp_path)
        info = models_mod._get_checkpoint_info("transformer")
        assert info["exists"] is False

    def test_with_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(models_mod, "_CHECKPOINT_DIR", tmp_path)
        ckpt = tmp_path / "transformer_latest.pt"
        ckpt.write_bytes(b"\x00" * 1024)
        info = models_mod._get_checkpoint_info("transformer")
        assert info["exists"] is True
        assert "size_mb" in info
        assert "modified" in info


# ── _trade_by_model_chart ─────────────────────────────────────────────────

class TestTradeByModelChart:
    def test_empty_returns_paragraph(self):
        result = models_mod._trade_by_model_chart({})
        assert isinstance(result, html.P)

    def test_with_data_returns_graph(self, sample_bt_data):
        result = models_mod._trade_by_model_chart({"run_a": sample_bt_data})
        assert isinstance(result, dcc.Graph)


# ── _metrics_comparison_table ─────────────────────────────────────────────

class TestMetricsComparisonTable:
    def test_empty_returns_div(self):
        result = models_mod._metrics_comparison_table({})
        assert isinstance(result, html.Div)

    def test_with_data_returns_table(self, sample_bt_data):
        result = models_mod._metrics_comparison_table({"run_a": sample_bt_data})
        assert isinstance(result, html.Table)


# ── update_models callback ───────────────────────────────────────────────

class TestUpdateModels:
    def test_no_backtests(self, monkeypatch, mock_loader, tmp_path):
        mock_loader.load_all_backtest_results.return_value = {}
        monkeypatch.setattr(models_mod, "loader", mock_loader)
        monkeypatch.setattr(models_mod, "_CHECKPOINT_DIR", tmp_path)
        result = models_mod.update_models(0)
        assert isinstance(result, html.Div)

    def test_with_backtests(self, monkeypatch, mock_loader, tmp_path):
        monkeypatch.setattr(models_mod, "loader", mock_loader)
        monkeypatch.setattr(models_mod, "_CHECKPOINT_DIR", tmp_path)
        result = models_mod.update_models(0)
        assert isinstance(result, html.Div)
        # kpis + registry + comparison
        assert len(result.children) == 3
