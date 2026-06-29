"""
Financial RAG System — Write-Through Cache Invalidator

Listens for data-update events (e.g. new SEC filings, price feeds) and
proactively invalidates all affected cache entries across every layer
so that subsequent queries always hit fresh data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.cache.cache_manager import CacheManager
from app.models import IngestionEvent

logger = logging.getLogger(__name__)


@dataclass
class InvalidationRecord:
    """Immutable log entry for a single invalidation event."""

    timestamp: float
    tickers: list[str]
    source: str
    l1_invalidated: int = 0
    l2_invalidated: int = 0
    l3_invalidated: int = 0
    total_invalidated: int = 0
    trigger: str = "manual"  # "manual" | "ingestion"


class CacheInvalidator:
    """Write-through cache invalidation controller.

    Holds a reference to the :class:`~app.cache.cache_manager.CacheManager`
    and exposes two invalidation paths:

    * **Manual**: call :meth:`invalidate` with a list of tickers.
    * **Event-driven**: call :meth:`on_data_update` when new data
      arrives via the ingestion pipeline.

    Every invalidation is logged with per-layer counts and stored in a
    bounded history ring for observability.

    Usage::

        invalidator = CacheInvalidator(cache_manager)
        # On ingestion event:
        await invalidator.on_data_update(event)
        # Manual invalidation:
        await invalidator.invalidate(["AAPL", "NVDA"])
    """

    _MAX_HISTORY: int = 500  # Ring buffer size for invalidation records

    def __init__(self, cache_manager: CacheManager) -> None:
        self._cache_manager: CacheManager = cache_manager
        self._history: list[InvalidationRecord] = []
        self._total_invalidations: int = 0

        logger.info("CacheInvalidator initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def invalidate(
        self,
        tickers: list[str],
        *,
        source: str = "unknown",
    ) -> InvalidationRecord:
        """Invalidate all cache entries for the given *tickers*.

        Parameters:
            tickers: Ticker symbols whose cache entries should be purged.
            source:  A human-readable label for the event source
                     (e.g. ``"price_feed"``, ``"sec_filing"``).

        Returns:
            An :class:`InvalidationRecord` summarising what was purged.
        """
        if not tickers:
            logger.debug("CacheInvalidator.invalidate called with empty tickers list")
            return InvalidationRecord(
                timestamp=time.time(),
                tickers=[],
                source=source,
                trigger="manual",
            )

        t0 = time.perf_counter()
        results = await self._cache_manager.invalidate_by_tickers(tickers)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        record = InvalidationRecord(
            timestamp=time.time(),
            tickers=list(tickers),
            source=source,
            l1_invalidated=results.get("l1", 0),
            l2_invalidated=results.get("l2", 0),
            l3_invalidated=results.get("l3", 0),
            total_invalidated=sum(results.values()),
            trigger="manual",
        )

        self._record(record)

        logger.info(
            "CacheInvalidator invalidated %d entries for tickers=%s " "source=%s in %.2fms (L1=%d, L2=%d, L3=%d)",
            record.total_invalidated,
            tickers,
            source,
            elapsed_ms,
            record.l1_invalidated,
            record.l2_invalidated,
            record.l3_invalidated,
        )

        return record

    async def on_data_update(self, event: IngestionEvent) -> InvalidationRecord:
        """Handle an incoming data-update event from the ingestion pipeline.

        Extracts all affected tickers from the event's documents and
        invalidates corresponding cache entries.

        Parameters:
            event: The ingestion event containing new/updated documents.

        Returns:
            An :class:`InvalidationRecord` summarising invalidations.
        """
        # Collect unique tickers from the incoming documents
        tickers: list[str] = []
        seen: set[str] = set()
        for doc in event.documents:
            if doc.ticker and doc.ticker.upper() not in seen:
                seen.add(doc.ticker.upper())
                tickers.append(doc.ticker.upper())

        if not tickers:
            logger.debug(
                "CacheInvalidator.on_data_update — event %s has no ticker-bearing documents",
                event.event_id,
            )
            return InvalidationRecord(
                timestamp=time.time(),
                tickers=[],
                source=event.source,
                trigger="ingestion",
            )

        t0 = time.perf_counter()
        results = await self._cache_manager.invalidate_by_tickers(tickers)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        record = InvalidationRecord(
            timestamp=time.time(),
            tickers=tickers,
            source=event.source,
            l1_invalidated=results.get("l1", 0),
            l2_invalidated=results.get("l2", 0),
            l3_invalidated=results.get("l3", 0),
            total_invalidated=sum(results.values()),
            trigger="ingestion",
        )

        self._record(record)

        logger.info(
            "CacheInvalidator on_data_update — event=%s source=%s " "tickers=%s invalidated=%d entries in %.2fms",
            event.event_id,
            event.source,
            tickers,
            record.total_invalidated,
            elapsed_ms,
        )

        return record

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def total_invalidations(self) -> int:
        """Cumulative count of entries invalidated since creation."""
        return self._total_invalidations

    @property
    def history(self) -> list[InvalidationRecord]:
        """Read-only view of the invalidation history (most recent last)."""
        return list(self._history)

    def recent_history(self, n: int = 20) -> list[InvalidationRecord]:
        """Return the *n* most recent invalidation records."""
        return list(self._history[-n:])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(self, record: InvalidationRecord) -> None:
        """Append an invalidation record to the bounded ring buffer."""
        self._total_invalidations += record.total_invalidated
        self._history.append(record)
        # Trim to max history size
        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[-self._MAX_HISTORY :]
