"""
Financial RAG System — L1 Exact Query Cache

Hash-based exact-match cache with TTL expiration and LRU eviction.
Uses an in-memory OrderedDict for O(1) lookups with LRU ordering.
An optional Redis backend is attempted at startup; if unavailable the
cache degrades gracefully to in-memory only.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Final

from app.config import get_config
from app.models import CacheEntry, CacheLayer, QueryResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Redis backend
# ---------------------------------------------------------------------------
_redis_available: bool = False
try:
    import redis.asyncio as aioredis  # type: ignore[import-untyped]
    from redis.backoff import ExponentialBackoff
    from redis.retry import Retry

    _redis_available = True
except ImportError:
    aioredis = None  # type: ignore[assignment]


class ExactCache:
    """L1 exact-match query cache.

    Keys are SHA-256 hashes of the raw query string, giving
    deterministic O(1) lookups.  The in-memory store is an
    :class:`~collections.OrderedDict` so the least-recently-used entry
    can be evicted in O(1) when ``max_entries`` is reached.

    Attributes:
        hits:   Total number of cache hits since creation.
        misses: Total number of cache misses since creation.
    """

    _KEY_PREFIX: Final[str] = "frag:l1:"

    def __init__(self) -> None:
        cfg = get_config().cache
        self._ttl: float = float(cfg.l1_ttl_seconds)
        self._max_entries: int = cfg.l1_max_entries

        # In-memory LRU store: hash -> CacheEntry
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

        # Metrics
        self.hits: int = 0
        self.misses: int = 0
        self.redis_hits: int = 0
        self.redis_misses: int = 0
        self.redis_reconnects: int = 0

        # Optional Redis handle (lazily connected)
        self._redis: aioredis.Redis | None = None  # type: ignore[name-defined]
        self._redis_enabled: bool = _redis_available
        self._redis_connected: bool = False
        self._redis_last_failure: float = 0.0
        self._redis_reconnect_interval: float = 30.0  # seconds

        logger.info(
            "ExactCache initialised — ttl=%ss, max_entries=%d, redis=%s",
            self._ttl,
            self._max_entries,
            self._redis_enabled,
        )

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    async def _get_redis(self) -> aioredis.Redis | None:  # type: ignore[name-defined]
        """Return a lazily-created Redis connection, or *None*."""
        if not _redis_available:
            return None

        # If previously failed, only retry after reconnect interval
        if not self._redis_enabled and self._redis_last_failure > 0:
            if (time.time() - self._redis_last_failure) < self._redis_reconnect_interval:
                return None
            # Time to retry
            logger.info("ExactCache attempting Redis reconnect...")
            self._redis_enabled = True

        if self._redis is not None and self._redis_connected:
            return self._redis

        try:
            redis_cfg = get_config().redis
            self._redis = aioredis.Redis(
                host=redis_cfg.host,
                port=redis_cfg.port,
                db=redis_cfg.db,
                password=redis_cfg.password,
                decode_responses=redis_cfg.decode_responses,
                socket_timeout=redis_cfg.socket_timeout,
                socket_connect_timeout=redis_cfg.socket_timeout,
                retry=Retry(ExponentialBackoff(), 3),
                retry_on_timeout=True,
            )
            await self._redis.ping()
            if not self._redis_connected:
                self.redis_reconnects += 1
            self._redis_connected = True
            self._redis_enabled = True
            logger.info("ExactCache connected to Redis at %s:%d", redis_cfg.host, redis_cfg.port)
            return self._redis
        except Exception:
            logger.warning("ExactCache Redis unavailable — falling back to in-memory")
            self._redis_enabled = False
            self._redis_connected = False
            self._redis_last_failure = time.time()
            self._redis = None
            return None

    async def warm(self) -> None:
        """Resolve Redis availability during startup instead of first query."""
        await self._get_redis()

    async def health_check(self) -> dict:
        """Return Redis connection status and metrics."""
        redis_cfg = get_config().redis
        return {
            "redis_connected": self._redis_connected,
            "redis_host": f"{redis_cfg.host}:{redis_cfg.port}",
            "redis_hits": self.redis_hits,
            "redis_misses": self.redis_misses,
            "redis_reconnects": self.redis_reconnects,
            "redis_status": "connected" if self._redis_connected else "disconnected",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_query(query: str) -> str:
        """Produce a deterministic SHA-256 hex digest for *query*."""
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()

    async def get(self, query: str) -> QueryResponse | None:
        """Look up *query* in L1 cache.

        Returns the cached :class:`QueryResponse` on hit, or ``None``
        on miss.  Expired entries are pruned lazily on access.
        """
        key = self._hash_query(query)

        async with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                if entry.is_expired:
                    del self._store[key]
                    logger.debug("ExactCache expired key %s", key[:12])
                else:
                    # Move to end (most-recently-used)
                    self._store.move_to_end(key)
                    entry.access_count += 1
                    self.hits += 1
                    logger.debug("ExactCache HIT key %s (accesses=%d)", key[:12], entry.access_count)
                    return entry.response

        # Try Redis as secondary store
        redis = await self._get_redis()
        if redis is not None:
            try:
                raw = await redis.get(f"{self._KEY_PREFIX}{key}")
                if raw is not None:
                    entry = CacheEntry.model_validate_json(raw)
                    if not entry.is_expired:
                        # Promote into in-memory store
                        async with self._lock:
                            self._store[key] = entry
                            self._store.move_to_end(key)
                            self._evict_if_needed()
                        self.hits += 1
                        self.redis_hits += 1
                        logger.debug("ExactCache HIT (redis) key %s", key[:12])
                        return entry.response
                    self.redis_misses += 1
            except Exception as exc:
                logger.warning("ExactCache Redis GET failed: %s", exc)

        self.misses += 1
        self.redis_misses += 1
        return None

    async def set(
        self,
        query: str,
        response: QueryResponse,
        tickers: list[str] | None = None,
    ) -> None:
        """Store a response in L1 cache keyed by *query*.

        Parameters:
            query:    The original query string.
            response: The fully-formed response to cache.
            tickers:  Optional list of ticker symbols associated with
                      the entry (used for targeted invalidation).
        """
        key = self._hash_query(query)
        entry = CacheEntry(
            query=query,
            query_hash=key,
            response=response,
            created_at=time.time(),
            ttl_seconds=self._ttl,
            tickers=tickers or [],
        )

        async with self._lock:
            # Insert / update and mark as most-recently-used
            self._store[key] = entry
            self._store.move_to_end(key)
            self._evict_if_needed()

        # Write-through to Redis (fire-and-forget)
        redis = await self._get_redis()
        if redis is not None:
            try:
                await redis.setex(
                    f"{self._KEY_PREFIX}{key}",
                    int(self._ttl),
                    entry.model_dump_json(),
                )
            except Exception as exc:
                logger.warning("ExactCache Redis SET failed: %s", exc)

        logger.debug("ExactCache SET key %s (ttl=%ss)", key[:12], self._ttl)

    async def invalidate(self, query: str) -> bool:
        """Remove a specific query from the cache.

        Returns ``True`` if the entry existed and was removed.
        """
        key = self._hash_query(query)
        removed = False

        async with self._lock:
            if key in self._store:
                del self._store[key]
                removed = True

        redis = await self._get_redis()
        if redis is not None:
            try:
                await redis.delete(f"{self._KEY_PREFIX}{key}")
            except Exception as exc:
                logger.warning("ExactCache Redis DELETE failed: %s", exc)

        if removed:
            logger.debug("ExactCache INVALIDATED key %s", key[:12])
        return removed

    async def invalidate_by_tickers(self, tickers: list[str]) -> int:
        """Invalidate all entries associated with any of *tickers*.

        Returns the number of entries removed.
        """
        if not tickers:
            return 0

        ticker_set = {t.upper() for t in tickers}
        keys_to_remove: list[str] = []

        async with self._lock:
            for key, entry in self._store.items():
                if ticker_set.intersection(t.upper() for t in entry.tickers):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._store[key]

        # Best-effort Redis cleanup
        if keys_to_remove:
            redis = await self._get_redis()
            if redis is not None:
                try:
                    await redis.delete(*(f"{self._KEY_PREFIX}{k}" for k in keys_to_remove))
                except Exception as exc:
                    logger.warning("ExactCache Redis bulk DELETE failed: %s", exc)

        if keys_to_remove:
            logger.info(
                "ExactCache invalidated %d entries for tickers %s",
                len(keys_to_remove),
                tickers,
            )
        return len(keys_to_remove)

    async def clear(self) -> None:
        """Remove **all** entries from the cache."""
        async with self._lock:
            count = len(self._store)
            self._store.clear()

        redis = await self._get_redis()
        if redis is not None:
            try:
                cursor: int | str = 0
                while True:
                    cursor, keys = await redis.scan(cursor=int(cursor), match=f"{self._KEY_PREFIX}*", count=500)
                    if keys:
                        await redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as exc:
                logger.warning("ExactCache Redis CLEAR failed: %s", exc)

        logger.info("ExactCache CLEARED %d entries", count)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a float in [0.0, 1.0]."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Return the current number of entries in the in-memory store."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict the least-recently-used entry if over capacity.

        Must be called while holding ``self._lock``.
        """
        while len(self._store) > self._max_entries:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("ExactCache EVICTED key %s", evicted_key[:12])
