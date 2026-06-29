"""
Financial RAG System — L3 Hot Ticker Cache

Maintains precomputed embeddings and pre-retrieved document chunks for
a configurable list of trending / high-traffic tickers.  Queries that
mention one of these tickers can be served almost instantly.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np
from numpy.typing import NDArray

from app.config import get_config
from app.models import CacheLayer, QueryResponse, ScoredDocument

logger = logging.getLogger(__name__)


@dataclass
class HotTickerEntry:
    """Pre-computed data for a single hot ticker."""

    ticker: str
    embedding: NDArray[np.float32] | None = None
    documents: list[ScoredDocument] = field(default_factory=list)
    precomputed_response: QueryResponse | None = None
    last_refreshed: float = field(default_factory=time.time)


# Type alias for the refresh callback
RefreshCallback = Callable[[str], Awaitable[HotTickerEntry]]


class HotTickerCache:
    """L3 cache for high-traffic tickers.

    A background task periodically invokes a user-supplied refresh
    callback for each configured ticker, storing pre-retrieved
    documents and embeddings.  On :meth:`get`, the cache detects
    ticker mentions in the query and returns the precomputed result
    if available.

    Attributes:
        hits:   Number of cache hits since creation.
        misses: Number of cache misses since creation.
    """

    def __init__(self, refresh_callback: RefreshCallback | None = None) -> None:
        cfg = get_config().cache
        self._tickers: list[str] = [t.upper() for t in cfg.l3_tickers]
        self._refresh_interval: float = float(cfg.l3_refresh_interval_seconds)

        # Ticker -> precomputed data
        self._store: dict[str, HotTickerEntry] = {}
        self._lock = asyncio.Lock()

        # Optional refresh callback — set via `register_refresh_callback`
        # or passed at construction time.
        self._refresh_cb: RefreshCallback | None = refresh_callback

        # Background refresh task handle
        self._refresh_task: asyncio.Task[None] | None = None

        # Compiled regex: match any configured ticker as a whole word
        # (case-insensitive).  E.g.  \b(AAPL|NVDA|TSLA|...)\b
        if self._tickers:
            escaped = "|".join(re.escape(t) for t in self._tickers)
            self._ticker_re = re.compile(rf"\b({escaped})\b", re.IGNORECASE)
        else:
            self._ticker_re = None

        # Metrics
        self.hits: int = 0
        self.misses: int = 0

        logger.info(
            "HotTickerCache initialised — %d tickers, refresh_interval=%ss",
            len(self._tickers),
            self._refresh_interval,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register_refresh_callback(self, callback: RefreshCallback) -> None:
        """Register (or replace) the async callback used to refresh data
        for each ticker.

        The callback signature must be::

            async def refresh(ticker: str) -> HotTickerEntry: ...
        """
        self._refresh_cb = callback
        logger.info("HotTickerCache refresh callback registered")

    async def start(self) -> None:
        """Start the periodic background refresh loop."""
        if self._refresh_task is not None:
            logger.warning("HotTickerCache refresh loop already running")
            return
        self._refresh_task = asyncio.create_task(self._refresh_loop(), name="hot-ticker-refresh")
        logger.info("HotTickerCache background refresh started")

    async def stop(self) -> None:
        """Cancel the background refresh loop."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
            logger.info("HotTickerCache background refresh stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, query: str) -> QueryResponse | None:
        """Return a precomputed response if *query* mentions a hot ticker.

        The first detected hot ticker wins (queries mentioning multiple
        tickers fall through to the full retrieval pipeline).
        """
        detected = self._detect_tickers(query)
        if not detected:
            self.misses += 1
            return None

        # Only serve from cache when exactly one hot ticker is mentioned
        # — multi-ticker queries need the full pipeline.
        if len(detected) != 1:
            self.misses += 1
            logger.debug("HotTickerCache SKIP — multiple tickers detected: %s", detected)
            return None

        ticker = detected[0]

        async with self._lock:
            entry = self._store.get(ticker)
            if entry is None or entry.precomputed_response is None:
                self.misses += 1
                return None

            # Check staleness
            age = time.time() - entry.last_refreshed
            if age > self._refresh_interval * 3:
                # Entry too stale — treat as miss
                self.misses += 1
                logger.debug("HotTickerCache STALE — ticker=%s age=%.1fs", ticker, age)
                return None

            self.hits += 1
            logger.debug("HotTickerCache HIT — ticker=%s", ticker)
            return entry.precomputed_response

    async def get_documents(self, ticker: str) -> list[ScoredDocument]:
        """Return precomputed documents for *ticker*, or an empty list."""
        async with self._lock:
            entry = self._store.get(ticker.upper())
            if entry is not None:
                return list(entry.documents)
        return []

    async def refresh(self, tickers: list[str] | None = None) -> int:
        """Manually refresh one or more tickers.

        Parameters:
            tickers: Tickers to refresh.  Defaults to all configured tickers.

        Returns:
            Number of tickers successfully refreshed.
        """
        targets = [t.upper() for t in tickers] if tickers else list(self._tickers)
        refreshed = 0

        for ticker in targets:
            try:
                entry = await self._refresh_single(ticker)
                if entry is not None:
                    async with self._lock:
                        self._store[ticker] = entry
                    refreshed += 1
            except Exception as exc:
                logger.error("HotTickerCache refresh failed for %s: %s", ticker, exc)

        logger.info("HotTickerCache refreshed %d/%d tickers", refreshed, len(targets))
        return refreshed

    async def invalidate(self, tickers: list[str]) -> int:
        """Remove specific tickers from the hot cache.

        Returns the number of entries removed.
        """
        removed = 0
        async with self._lock:
            for ticker in tickers:
                key = ticker.upper()
                if key in self._store:
                    del self._store[key]
                    removed += 1
        if removed:
            logger.info("HotTickerCache invalidated %d tickers: %s", removed, tickers)
        return removed

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Cache hit rate in [0.0, 1.0]."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Number of tickers currently cached."""
        return len(self._store)

    def get_cached_tickers(self) -> list[str]:
        """Return the tickers with currently cached entries."""
        return list(self._store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_tickers(self, query: str) -> list[str]:
        """Return a deduplicated list of hot tickers found in *query*."""
        if self._ticker_re is None:
            return []
        matches = self._ticker_re.findall(query)
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for m in matches:
            upper = m.upper()
            if upper not in seen:
                seen.add(upper)
                result.append(upper)
        return result

    async def _refresh_single(self, ticker: str) -> HotTickerEntry | None:
        """Refresh data for a single ticker via the registered callback."""
        if self._refresh_cb is None:
            return None
        return await self._refresh_cb(ticker)

    async def _refresh_loop(self) -> None:
        """Background loop that periodically refreshes all hot tickers."""
        logger.info("HotTickerCache refresh loop started (interval=%ss)", self._refresh_interval)
        while True:
            try:
                await asyncio.sleep(self._refresh_interval)
                await self.refresh()
            except asyncio.CancelledError:
                logger.info("HotTickerCache refresh loop cancelled")
                raise
            except Exception as exc:
                logger.error("HotTickerCache refresh loop error: %s", exc)
                # Continue running — don't let one failure kill the loop
                await asyncio.sleep(1.0)
