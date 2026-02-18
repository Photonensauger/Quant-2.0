"""Simple TTL cache for dashboard data to avoid repeated disk reads."""

from __future__ import annotations

import time
from typing import Any


class TTLCache:
    """In-memory key-value cache with per-entry time-to-live expiration.

    Parameters
    ----------
    ttl_seconds : int
        Number of seconds before a cached entry expires.  Default 60.
    max_size : int
        Maximum number of entries.  Oldest entries are evicted when exceeded.
        Default 50.  Set to 0 for unlimited.
    """

    def __init__(self, ttl_seconds: int = 60, max_size: int = 50) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        """Return cached value if present and not expired, else ``None``."""
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.time() - stored_at > self.ttl_seconds:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* with the current timestamp."""
        # Evict oldest entry if at capacity (and key is new)
        if self.max_size and key not in self._store and len(self._store) >= self.max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]
        self._store[key] = (time.time(), value)

    def clear(self) -> None:
        """Invalidate all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        """Return number of entries (including possibly expired ones)."""
        return len(self._store)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics for debugging."""
        now = time.time()
        alive = sum(1 for ts, _ in self._store.values() if now - ts <= self.ttl_seconds)
        return {
            "total_entries": len(self._store),
            "alive_entries": alive,
            "expired_entries": len(self._store) - alive,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }
