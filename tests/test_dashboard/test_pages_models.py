"""Tests for dashboard/pages/models.py — pure unit tests."""

from pathlib import Path
from unittest.mock import patch

import torch
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


# ── Device KPI card ─────────────────────────────────────────────────────

class TestDeviceKpi:
    """The Models page must display a Device KPI card."""

    def _get_kpi_grid(self, monkeypatch, mock_loader, tmp_path):
        monkeypatch.setattr(models_mod, "loader", mock_loader)
        monkeypatch.setattr(models_mod, "_CHECKPOINT_DIR", tmp_path)
        result = models_mod.update_models(0)
        # First child is the KPI grid div
        return result.children[0]

    def test_device_kpi_present(self, monkeypatch, mock_loader, tmp_path):
        kpi_grid = self._get_kpi_grid(monkeypatch, mock_loader, tmp_path)
        # KPIs: Models, Trained, Backtests, Device
        assert len(kpi_grid.children) == 4

    def test_device_kpi_shows_cpu(self, monkeypatch, mock_loader, tmp_path):
        with patch("quant.config.settings.get_device",
                   return_value=torch.device("cpu")):
            kpi_grid = self._get_kpi_grid(monkeypatch, mock_loader, tmp_path)
            device_card = kpi_grid.children[3]
            # The card should contain the text "CPU"
            assert "CPU" in _extract_text(device_card)

    def test_device_kpi_shows_mps(self, monkeypatch, mock_loader, tmp_path):
        with patch("quant.config.settings.get_device",
                   return_value=torch.device("mps")):
            kpi_grid = self._get_kpi_grid(monkeypatch, mock_loader, tmp_path)
            device_card = kpi_grid.children[3]
            assert "MPS" in _extract_text(device_card)


def _extract_text(component) -> str:
    """Recursively extract text content from a Dash component tree."""
    parts = []
    if isinstance(component, str):
        return component
    children = getattr(component, "children", None)
    if children is None:
        return ""
    if isinstance(children, str):
        return children
    if isinstance(children, list):
        for child in children:
            parts.append(_extract_text(child))
    return " ".join(parts)
