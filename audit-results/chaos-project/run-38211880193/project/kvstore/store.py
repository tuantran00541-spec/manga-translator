"""A tiny in-memory key/value store with TTL and JSON file persistence.

Values must be JSON-serializable.  Expiry is computed against a
``clock`` callable (``time.time`` by default) so tests can run with a
fake clock and never have to sleep.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Optional


class Store:
    """A small key/value store with optional per-key TTLs."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._data: dict[str, tuple[Any, Optional[float]]] = {}

    # -- helpers ---------------------------------------------------------

    def _expired(self, expires_at: Optional[float]) -> bool:
        return expires_at is not None and self.clock() >= expires_at

    # -- core API --------------------------------------------------------

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store ``value`` under ``key``.

        ``ttl`` is in seconds, measured against ``self.clock``: once
        ``clock()`` reaches the expiry the key behaves as deleted.
        """
        json.dumps(value)  # fail early on non-serializable values
        expires_at = self.clock() + ttl if ttl is not None else None
        self._data[key] = (value, expires_at)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` or ``default`` if absent/expired."""
        entry = self._data.get(key)
        if entry is None:
            return default
        value, expires_at = entry
        if self._expired(expires_at):
            del self._data[key]
            return default
        return value

    def delete(self, key: str) -> bool:
        """Delete ``key``; return True if a live entry was removed."""
        entry = self._data.get(key)
        if entry is None:
            return False
        if self._expired(entry[1]):
            del self._data[key]
            return False
        del self._data[key]
        return True

    def keys(self) -> list:
        """Sorted list of live keys (expired keys are dropped)."""
        self._purge_expired()
        return sorted(self._data)

    def _purge_expired(self) -> None:
        self._data = {
            key: entry
            for key, entry in self._data.items()
            if not self._expired(entry[1])
        }

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        """Write the store to ``path`` as JSON, including expiries."""
        self._purge_expired()
        payload = {
            "keys": {
                key: {"value": value, "expires_at": expires_at}
                for key, (value, expires_at) in self._data.items()
            }
        }
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.write("\n")

    @classmethod
    def load(cls, path: str, clock: Callable[[], float] = time.time) -> "Store":
        """Read a store back from ``path``.

        Keys that have already expired by ``clock`` are dropped.
        """
        store = cls(clock=clock)
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        for key, entry in payload["keys"].items():
            expires_at = entry.get("expires_at")
            if expires_at is not None and clock() >= expires_at:
                continue
            store._data[key] = (entry["value"], expires_at)
        return store
