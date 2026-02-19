"""Tests for dashboard/pages/risk.py — pure unit tests."""

from dash import html, dcc

from dashboard.pages import risk as risk_mod
from dashboard import config

THRESHOLDS = config.RISK_THRESHOLDS


# ── _check_thresholds ─────────────────────────────────────────────────────

class TestCheckThresholds:
    def test_no_breaches(self):
        alerts = risk_mod._check_thresholds(
            sharpe=1.5, max_dd=-0.05, volatility=0.10, var_95=-0.01,
        )
        assert alerts == []

    def test_low_sharpe(self):
        alerts = risk_mod._check_thresholds(
            sharpe=0.3, max_dd=-0.05, volatility=0.10, var_95=-0.01,
        )
        assert len(alerts) == 1
        assert alerts[0]["level"] == "warning"
        assert "Sharpe" in alerts[0]["msg"]

    def test_critical_drawdown(self):
        alerts = risk_mod._check_thresholds(
            sharpe=1.5, max_dd=-0.30, volatility=0.10, var_95=-0.01,
        )
        assert any(a["level"] == "critical" for a in alerts)

    def test_warning_drawdown(self):
        alerts = risk_mod._check_thresholds(
            sharpe=1.5, max_dd=-0.20, volatility=0.10, var_95=-0.01,
        )
        assert any(a["level"] == "warning" and "Drawdown" in a["msg"] for a in alerts)

    def test_high_volatility(self):
        alerts = risk_mod._check_thresholds(
            sharpe=1.5, max_dd=-0.05, volatility=0.40, var_95=-0.01,
        )
        assert any("Volatility" in a["msg"] for a in alerts)

    def test_var_breach(self):
        alerts = risk_mod._check_thresholds(
            sharpe=1.5, max_dd=-0.05, volatility=0.10, var_95=-0.05,
        )
        assert any("VaR" in a["msg"] for a in alerts)

    def test_multiple_breaches(self):
        alerts = risk_mod._check_thresholds(
            sharpe=0.2, max_dd=-0.30, volatility=0.50, var_95=-0.05,
        )
        assert len(alerts) >= 3


# ── _ratio_cls ────────────────────────────────────────────────────────────

class TestRatioCls:
    def test_profit(self):
        assert risk_mod._ratio_cls(1.5) == "profit"

    def test_loss(self):
        assert risk_mod._ratio_cls(-0.5) == "loss"

    def test_empty(self):
        assert risk_mod._ratio_cls(0.5) == ""

    def test_sharpe_warning(self):
        assert risk_mod._ratio_cls(0.3, metric="sharpe") == "warning"

    def test_max_dd_critical(self):
        assert risk_mod._ratio_cls(-0.30, metric="max_drawdown") == "loss"

    def test_max_dd_warning(self):
        assert risk_mod._ratio_cls(-0.20, metric="max_drawdown") == "warning"

    def test_volatility_warning(self):
        assert risk_mod._ratio_cls(0.40, metric="volatility") == "warning"


# ── _alert_banner ─────────────────────────────────────────────────────────

class TestAlertBanner:
    def test_no_alerts_empty_div(self):
        result = risk_mod._alert_banner([])
        assert isinstance(result, html.Div)
        assert result.children is None  # empty Div()

    def test_warning_alerts(self):
        alerts = [{"level": "warning", "msg": "Low Sharpe"}]
        result = risk_mod._alert_banner(alerts)
        # Find the label span
        header_div = result.children[0]
        label_span = header_div.children[1]
        assert label_span.children == "WARNING"

    def test_critical_alerts(self):
        alerts = [{"level": "critical", "msg": "Severe drawdown"}]
        result = risk_mod._alert_banner(alerts)
        header_div = result.children[0]
        label_span = header_div.children[1]
        assert label_span.children == "CRITICAL"


# ── _var_chart ────────────────────────────────────────────────────────────

class TestVarChart:
    def test_sufficient_data(self, sample_returns):
        result = risk_mod._var_chart(sample_returns)
        assert isinstance(result, dcc.Graph)

    def test_insufficient_data(self):
        result = risk_mod._var_chart([0.01, 0.02])
        assert isinstance(result, html.P)


# ── update_risk callback ─────────────────────────────────────────────────

class TestUpdateRisk:
    def test_none_returns_empty_state(self, monkeypatch, mock_loader):
        monkeypatch.setattr(risk_mod, "loader", mock_loader)
        result = risk_mod.update_risk(None)
        assert isinstance(result, html.Div)
        assert "empty-state" in result.className

    def test_valid_bt_returns_populated(self, monkeypatch, mock_loader):
        monkeypatch.setattr(risk_mod, "loader", mock_loader)
        result = risk_mod.update_risk("test_run")
        assert isinstance(result, html.Div)
        # alert_banner + kpis + rolling_section + charts
        assert len(result.children) == 4
