"""Tests for the shared DashboardDataLoader singleton and cross-page cache.

These tests guard against the bug where each dashboard page created its own
DashboardDataLoader instance, causing cache invalidation on one page to have
no effect on other pages.
"""

from unittest.mock import patch

from dashboard.data import loader as loader_mod
from dashboard.data.loader import DashboardDataLoader, get_shared_loader


class TestGetSharedLoaderSingleton:
    """get_shared_loader() must return the same instance every time."""

    def test_returns_dashboard_data_loader(self):
        with patch.object(loader_mod, "_shared_loader", None):
            instance = get_shared_loader()
            assert isinstance(instance, DashboardDataLoader)

    def test_returns_same_instance(self):
        with patch.object(loader_mod, "_shared_loader", None):
            a = get_shared_loader()
            b = get_shared_loader()
            assert a is b

    def test_does_not_recreate_after_first_call(self):
        with patch.object(loader_mod, "_shared_loader", None):
            first = get_shared_loader()
            # Mutate internal state to prove identity
            first._test_marker = True
            second = get_shared_loader()
            assert getattr(second, "_test_marker", False) is True


class TestAllPagesShareLoader:
    """Every dashboard page module must reference the same shared loader.

    This prevents regression to the bug where each page had its own cache.
    """

    def test_overview_uses_shared_loader(self):
        from dashboard.pages import overview as mod
        assert mod.loader is get_shared_loader()

    def test_positions_uses_shared_loader(self):
        from dashboard.pages import positions as mod
        assert mod.loader is get_shared_loader()

    def test_backtest_uses_shared_loader(self):
        from dashboard.pages import backtest as mod
        assert mod.loader is get_shared_loader()

    def test_models_uses_shared_loader(self):
        from dashboard.pages import models as mod
        assert mod.loader is get_shared_loader()

    def test_risk_uses_shared_loader(self):
        from dashboard.pages import risk as mod
        assert mod.loader is get_shared_loader()


class TestCrossPageCacheInvalidation:
    """Clearing the cache on the shared loader must affect all pages."""

    def test_clear_cache_propagates(self, tmp_data_dir, tmp_backtest_dir):
        with patch.object(loader_mod, "_shared_loader", None):
            # Force a fresh shared loader for this test
            loader_mod._shared_loader = DashboardDataLoader(
                data_dir=tmp_data_dir, backtest_dir=tmp_backtest_dir,
            )
            shared = get_shared_loader()

            # Load data to populate cache
            df1 = shared.load_market_data("AAPL", "1d")
            assert df1 is shared.load_market_data("AAPL", "1d")  # cached

            # Simulate what poll_training / poll_backtest does
            shared.clear_cache()

            # After clearing, the next load returns a new object (re-read)
            df2 = shared.load_market_data("AAPL", "1d")
            assert df1 is not df2
