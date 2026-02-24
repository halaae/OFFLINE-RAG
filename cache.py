"""
cache.py — Two-level query result cache.

L1 : In-process LRU cache (functools.lru_cache pattern, thread-safe via OrderedDict)
L2 : Disk cache (shelve — persistent across restarts)

Cache key  : SHA-256 of the normalised query string
Cache value : the full result dict returned by RAGPipeline.query()

Why two levels?
  • L1 is microsecond retrieval, lives in RAM.
  • L2 survives process restarts — useful during long research sessions.

Both layers use the same serialisation format (pickle via shelve).
"""

import hashlib
import logging
import shelve
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

log = logging.getLogger("cache")

L1_MAX_SIZE = 128   # maximum number of entries in the in-process LRU cache


class HybridCache:
    """
    Thread-safe two-level cache (RAM LRU + disk shelve).

    Parameters
    ----------
    cache_dir : directory where the disk cache is stored
    lru_size  : maximum number of L1 RAM entries
    """

    def __init__(self, cache_dir: str = "cache/disk", lru_size: int = L1_MAX_SIZE):
        self._dir      = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path  = str(self._dir / "query_cache")
        self._lru_size = lru_size
        self._l1: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

        # Stats
        self._l1_hits  = 0
        self._l2_hits  = 0
        self._misses   = 0

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(query: str) -> str:
        """Normalise query and return its SHA-256 hex digest as cache key."""
        normalised = " ".join(query.lower().split())
        return hashlib.sha256(normalised.encode()).hexdigest()

    # ------------------------------------------------------------------
    # L1 (RAM) operations
    # ------------------------------------------------------------------

    def _l1_get(self, key: str) -> Optional[dict]:
        with self._lock:
            if key in self._l1:
                # Move to end (most-recently-used)
                self._l1.move_to_end(key)
                return self._l1[key]
        return None

    def _l1_put(self, key: str, value: dict):
        with self._lock:
            if key in self._l1:
                self._l1.move_to_end(key)
            else:
                if len(self._l1) >= self._lru_size:
                    self._l1.popitem(last=False)   # evict oldest
            self._l1[key] = value

    # ------------------------------------------------------------------
    # L2 (disk) operations
    # ------------------------------------------------------------------

    def _l2_get(self, key: str) -> Optional[dict]:
        try:
            with shelve.open(self._db_path, flag="r") as db:
                return db.get(key)
        except Exception:
            return None

    def _l2_put(self, key: str, value: dict):
        try:
            with shelve.open(self._db_path, flag="c") as db:
                db[key] = value
        except Exception as exc:
            log.warning("Disk cache write failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[dict]:
        """
        Look up *query* in cache.

        Returns the cached result dict, or None on a cache miss.
        """
        key = self._make_key(query)

        # L1 check
        result = self._l1_get(key)
        if result is not None:
            self._l1_hits += 1
            log.debug("L1 cache HIT for key %s…", key[:12])
            return result

        # L2 check
        result = self._l2_get(key)
        if result is not None:
            self._l2_hits += 1
            self._l1_put(key, result)   # promote to L1
            log.debug("L2 cache HIT for key %s…", key[:12])
            return result

        self._misses += 1
        return None

    def set(self, query: str, result: dict):
        """Store *result* for *query* in both cache layers."""
        key = self._make_key(query)

        # Strip heavy numpy arrays before pickling to keep disk cache small
        serialisable = {
            k: v for k, v in result.items()
            if k not in {"query_embedding"}
        }

        self._l1_put(key, serialisable)
        self._l2_put(key, serialisable)

    def stats(self) -> dict:
        total = self._l1_hits + self._l2_hits + self._misses
        return {
            "l1_hits":     self._l1_hits,
            "l2_hits":     self._l2_hits,
            "misses":      self._misses,
            "total_lookups": total,
            "hit_rate":    (self._l1_hits + self._l2_hits) / max(total, 1),
            "l1_size":     len(self._l1),
        }

    def clear(self):
        """Flush both cache layers."""
        with self._lock:
            self._l1.clear()
        try:
            import glob
            for f in glob.glob(self._db_path + "*"):
                Path(f).unlink(missing_ok=True)
        except Exception as exc:
            log.warning("Cache clear failed: %s", exc)

    def close(self):
        """No-op — shelve is opened/closed per operation for safety."""
        pass