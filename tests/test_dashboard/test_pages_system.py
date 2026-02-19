"""Tests for dashboard/pages/system.py — pure unit tests."""

from datetime import datetime, timedelta
from pathlib import Path

from dash import html

from dashboard.pages import system as sys_mod


# ── _format_size ──────────────────────────────────────────────────────────

class TestFormatSize:
    def test_bytes(self):
        assert sys_mod._format_size(512) == "512 B"

    def test_kilobytes(self):
        result = sys_mod._format_size(2048)
        assert "KB" in result

    def test_megabytes(self):
        result = sys_mod._format_size(5 * 1024 * 1024)
        assert "MB" in result


# ── _freshness_cls ────────────────────────────────────────────────────────

class TestFreshnessCls:
    def test_recent(self):
        recent = datetime.now() - timedelta(hours=2)
        assert sys_mod._freshness_cls(recent) == "profit"

    def test_few_days(self):
        days_ago = datetime.now() - timedelta(days=3)
        assert sys_mod._freshness_cls(days_ago) == "warning"

    def test_old(self):
        old = datetime.now() - timedelta(days=30)
        assert sys_mod._freshness_cls(old) == "loss"


# ── update_system callback ───────────────────────────────────────────────

class TestUpdateSystem:
    def test_returns_div(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "cache"
        data_dir.mkdir()
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Write a dummy parquet-named file so the glob finds something
        (data_dir / "TEST_1d.parquet").write_bytes(b"\x00" * 100)

        monkeypatch.setattr(sys_mod.config, "DATA_DIR", data_dir)
        monkeypatch.setattr(sys_mod.config, "MODELS_DIR", models_dir)

        result = sys_mod.update_system(0)
        assert isinstance(result, html.Div)
        # data_section + model_section + sys_section + footer
        assert len(result.children) == 4

    def test_empty_dirs(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "cache"
        data_dir.mkdir()
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        monkeypatch.setattr(sys_mod.config, "DATA_DIR", data_dir)
        monkeypatch.setattr(sys_mod.config, "MODELS_DIR", models_dir)

        result = sys_mod.update_system(0)
        assert isinstance(result, html.Div)
