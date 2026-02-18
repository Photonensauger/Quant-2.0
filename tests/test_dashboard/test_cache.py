"""Tests for dashboard.data.cache.TTLCache."""

import time

from dashboard.data.cache import TTLCache


def test_set_and_get():
    cache = TTLCache(ttl_seconds=10)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_get_missing_key():
    cache = TTLCache(ttl_seconds=10)
    assert cache.get("nonexistent") is None


def test_ttl_expiry():
    cache = TTLCache(ttl_seconds=0.1)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    time.sleep(0.15)
    assert cache.get("key1") is None


def test_clear():
    cache = TTLCache(ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


def test_max_size_eviction():
    cache = TTLCache(ttl_seconds=60, max_size=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert len(cache) == 3

    # Adding a 4th entry should evict the oldest
    cache.set("d", 4)
    assert len(cache) == 3
    assert cache.get("a") is None  # oldest evicted
    assert cache.get("d") == 4


def test_max_size_update_existing():
    """Updating an existing key should not trigger eviction."""
    cache = TTLCache(ttl_seconds=60, max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("a", 10)  # update, not new
    assert len(cache) == 2
    assert cache.get("a") == 10
    assert cache.get("b") == 2


def test_len():
    cache = TTLCache(ttl_seconds=60)
    assert len(cache) == 0
    cache.set("x", 1)
    assert len(cache) == 1


def test_stats():
    cache = TTLCache(ttl_seconds=0.1, max_size=10)
    cache.set("a", 1)
    cache.set("b", 2)
    time.sleep(0.15)
    cache.set("c", 3)

    s = cache.stats()
    assert s["total_entries"] == 3
    assert s["alive_entries"] == 1
    assert s["expired_entries"] == 2
    assert s["max_size"] == 10
    assert s["ttl_seconds"] == 0.1
