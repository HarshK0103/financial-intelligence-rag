"""
Financial RAG System — Multi-Layer Cache Manager

Orchestrates L1 (exact) → L2 (semantic) → L3 (hot ticker) cache
lookups, short-circuiting on the first hit.  On a complete miss the
response is populated into the upper layers so subsequent queries
benefit from caching.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.cache.exact_cache import ExactCache
from app.cache.hot_ticker_cache import HotTickerCache, RefreshCallback
from app.cache.semantic_cache import SemanticCache
from app.models import CacheLayer, QueryResponse

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Aggregate hit/miss statistics across all cache layers."""

    total_lookups: int = 0
    total_hits: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    misses: int = 0

    # Per-layer timing (cumulative, seconds)
    l1_time_total: float = 0.0
    l2_time_total: float = 0.0
    l3_time_total: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Overall cache hit rate in [0.0, 1.0]."""
        return self.total_hits / self.total_lookups if self.total_lookups else 0.0

    @property
    def l1_hit_rate(self) -> float:
        return self.l1_hits / self.total_lookups if self.total_lookups else 0.0

    @property
    def l2_hit_rate(self) -> float:
        return self.l2_hits / self.total_lookups if self.total_lookups else 0.0

    @property
    def l3_hit_rate(self) -> float:
        return self.l3_hits / self.total_lookups if self.total_lookups else 0.0


class CacheManager:
    """Multi-layer cache orchestrator.

    Lookup order:

    1. **L1 — Exact cache**: hash-based exact match on the raw query.
    2. **L2 — Semantic cache**: cosine similarity on query embeddings.
    3. **L3 — Hot ticker cache**: precomputed results for trending
       tickers.

    The first layer to return a result short-circuits the remaining
    lookups.  On a full miss, :meth:`set` should be called to
    populate the upper layers with the newly computed response.

    Usage::

        mgr = CacheManager()
        layer, response = await mgr.get(query, embedding)
        if response is None:
            response = await compute_response(query)
            await mgr.set(query, embedding, response, tickers=["AAPL"])
    """

    def __init__(
        self,
        *,
        exact_cache: ExactCache | None = None,
        semantic_cache: SemanticCache | None = None,
        hot_ticker_cache: HotTickerCache | None = None,
    ) -> None:
        self.l1: ExactCache = exact_cache or ExactCache()
        self.l2: SemanticCache = semantic_cache or SemanticCache()
        self.l3: HotTickerCache = hot_ticker_cache or HotTickerCache()

        self.metrics = CacheMetrics()

        logger.info("CacheManager initialised with L1+L2+L3 layers")

    # ------------------------------------------------------------------
    # Lifecycle helpers (delegate to L3's background refresh)
    # ------------------------------------------------------------------

    def register_hot_ticker_callback(self, callback: RefreshCallback) -> None:
        """Register the async callback used by L3 to refresh hot ticker data."""
        self.l3.register_refresh_callback(callback)

    async def start(self) -> None:
        """Start background tasks (L3 refresh loop)."""
        await self.l1.warm()
        await self.l3.start()

    async def stop(self) -> None:
        """Gracefully stop background tasks."""
        await self.l3.stop()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def get(
        self,
        query: str,
        embedding: list[float] | NDArray[np.float32] | None = None,
    ) -> tuple[CacheLayer, QueryResponse | None]:
        """Look up *query* across all cache layers.

        Parameters:
            query:     The raw query string.
            embedding: The query embedding (required for L2; L1 and L3
                       work without it).

        Returns:
            A ``(CacheLayer, response)`` tuple.  ``response`` is
            ``None`` when all layers miss.
        """
        self.metrics.total_lookups += 1

        # ── L1: Exact match ──────────────────────────────────────
        t0 = time.perf_counter()
        try:
            response = await self.l1.get(query)
        except Exception as exc:
            logger.error("CacheManager L1 lookup failed: %s", exc)
            response = None
        elapsed = time.perf_counter() - t0
        self.metrics.l1_time_total += elapsed

        if response is not None:
            self.metrics.total_hits += 1
            self.metrics.l1_hits += 1
            logger.debug("CacheManager HIT L1 (%.2fms)", elapsed * 1000)
            return CacheLayer.L1_EXACT, response

        # ── L2: Semantic similarity ──────────────────────────────
        if embedding is not None:
            t0 = time.perf_counter()
            try:
                response = await self.l2.get(query, embedding)
            except Exception as exc:
                logger.error("CacheManager L2 lookup failed: %s", exc)
                response = None
            elapsed = time.perf_counter() - t0
            self.metrics.l2_time_total += elapsed

            if response is not None:
                self.metrics.total_hits += 1
                self.metrics.l2_hits += 1
                logger.debug("CacheManager HIT L2 (%.2fms)", elapsed * 1000)
                return CacheLayer.L2_SEMANTIC, response

        # ── L3: Hot ticker ───────────────────────────────────────
        t0 = time.perf_counter()
        try:
            response = await self.l3.get(query)
        except Exception as exc:
            logger.error("CacheManager L3 lookup failed: %s", exc)
            response = None
        elapsed = time.perf_counter() - t0
        self.metrics.l3_time_total += elapsed

        if response is not None:
            self.metrics.total_hits += 1
            self.metrics.l3_hits += 1
            logger.debug("CacheManager HIT L3 (%.2fms)", elapsed * 1000)
            return CacheLayer.L3_HOT_TICKER, response

        # ── Miss ─────────────────────────────────────────────────
        self.metrics.misses += 1
        logger.debug("CacheManager MISS for query=%r", query[:60])
        return CacheLayer.MISS, None

    async def set(
        self,
        query: str,
        embedding: list[float] | NDArray[np.float32] | None,
        response: QueryResponse,
        tickers: list[str] | None = None,
    ) -> None:
        """Populate L1 and L2 caches with a newly computed response.

        L3 is populated separately through its background refresh
        mechanism and is not written to here.

        Parameters:
            query:     The original query string.
            embedding: The query embedding (required for L2).
            response:  The computed response to cache.
            tickers:   Optional tickers for invalidation targeting.
        """
        # L1 — always write
        try:
            await self.l1.set(query, response, tickers=tickers)
        except Exception as exc:
            logger.error("CacheManager L1 set failed: %s", exc)

        # L2 — write only when we have an embedding
        if embedding is not None:
            try:
                await self.l2.set(query, embedding, response, tickers=tickers)
            except Exception as exc:
                logger.error("CacheManager L2 set failed: %s", exc)

        logger.debug("CacheManager SET query=%r", query[:60])

    async def invalidate_by_tickers(self, tickers: list[str]) -> dict[str, int]:
        """Invalidate all cache entries across every layer that are
        associated with any of *tickers*.

        Returns:
            A dict mapping layer names to the count of invalidated
            entries, e.g. ``{"l1": 3, "l2": 1, "l3": 2}``.
        """
        if not tickers:
            return {"l1": 0, "l2": 0, "l3": 0}

        results: dict[str, int] = {}

        try:
            results["l1"] = await self.l1.invalidate_by_tickers(tickers)
        except Exception as exc:
            logger.error("CacheManager L1 invalidation failed: %s", exc)
            results["l1"] = 0

        try:
            results["l2"] = await self.l2.invalidate_by_tickers(tickers)
        except Exception as exc:
            logger.error("CacheManager L2 invalidation failed: %s", exc)
            results["l2"] = 0

        try:
            results["l3"] = await self.l3.invalidate(tickers)
        except Exception as exc:
            logger.error("CacheManager L3 invalidation failed: %s", exc)
            results["l3"] = 0

        total = sum(results.values())
        if total:
            logger.info(
                "CacheManager invalidated %d entries for tickers %s — %s",
                total,
                tickers,
                results,
            )
        return results

    async def clear(self) -> None:
        """Clear all cache layers."""
        await self.l1.clear()
        await self.l2.clear()
        await self.l3.invalidate(list(self.l3._tickers))
        logger.info("CacheManager all layers CLEARED")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> CacheMetrics:
        """Return the aggregate cache metrics."""
        return self.metrics

    @property
    def hit_rate(self) -> float:
        """Overall cache hit rate across all layers."""
        return self.metrics.hit_rate
