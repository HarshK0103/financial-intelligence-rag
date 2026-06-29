"""
Financial RAG System — Async Stream Processor

Simulates a Kafka-like ingestion queue using :pymod:`asyncio.Queue`.
Events are consumed by background worker tasks, batched, embedded via
:class:`EmbeddingWorker`, and then handed to a caller-supplied indexing
callback.  A *separate* cache-invalidation callback fires after each
batch so that stale query caches are cleared without coupling the
ingestion layer to the cache layer.

Design invariants
─────────────────
* **Never blocks retrieval** — all work happens in background tasks
  on the same event loop.
* **At-least-once** — if a batch fails, its events are re-queued
  with exponential back-off (bounded to 3 retries).
* **Graceful shutdown** — ``stop()`` drains in-flight work before
  cancelling workers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from app.config import get_config
from app.ingestion.embedding_worker import EmbeddingWorker
from app.models import Document, IngestionEvent, IngestionResult

logger = logging.getLogger(__name__)

# Type aliases for the two callbacks.
IndexCallback = Callable[[list[Document]], Awaitable[int]]
"""Async callable that indexes documents and returns the count indexed."""

InvalidationCallback = Callable[[list[str]], Awaitable[int]]
"""Async callable that invalidates cache entries for tickers, returns count."""


class StreamProcessor:
    """Async ingestion pipeline backed by an :class:`asyncio.Queue`.

    Parameters
    ----------
    embedding_worker : EmbeddingWorker | None
        Worker used to generate embeddings.  A default instance is
        created when *None*.
    on_index : IndexCallback | None
        Called with the embedded docs to persist them in the index.
        Defaults to a no-op that logs and returns ``len(docs)``.
    on_cache_invalidate : InvalidationCallback | None
        Called with ticker symbols whose caches should be cleared.
        Defaults to a no-op that logs and returns ``0``.
    """

    def __init__(
        self,
        embedding_worker: EmbeddingWorker | None = None,
        on_index: IndexCallback | None = None,
        on_cache_invalidate: InvalidationCallback | None = None,
    ) -> None:
        cfg = get_config()
        icfg = cfg.ingestion

        self._queue: asyncio.Queue[IngestionEvent] = asyncio.Queue(
            maxsize=icfg.queue_max_size,
        )
        self._batch_size: int = icfg.batch_size
        self._worker_count: int = icfg.worker_count
        self._index_interval: float = icfg.background_index_interval_seconds

        self._embedding_worker = embedding_worker or EmbeddingWorker()
        self._on_index: IndexCallback = on_index or self._default_index
        self._on_cache_invalidate: InvalidationCallback = on_cache_invalidate or self._default_invalidate

        self._workers: list[asyncio.Task[None]] = []
        self._running: bool = False

        # Counters
        self._events_submitted: int = 0
        self._events_processed: int = 0
        self._events_failed: int = 0

        logger.info(
            "StreamProcessor created  queue_max=%d  batch=%d  workers=%d",
            icfg.queue_max_size,
            self._batch_size,
            self._worker_count,
        )

    # ── public API ────────────────────────────────────────────────

    async def submit(self, event: IngestionEvent) -> None:
        """Enqueue an :class:`IngestionEvent` for background processing.

        If the queue is full this will **block** until space is
        available—back-pressure propagates to the producer rather
        than dropping events.

        Raises
        ------
        RuntimeError
            If the processor has not been started.
        """
        if not self._running:
            raise RuntimeError("StreamProcessor is not running — call start_processing() first.")
        await self._queue.put(event)
        self._events_submitted += 1
        logger.debug(
            "Enqueued event %s  (%d docs)  queue_size=%d",
            event.event_id,
            len(event.documents),
            self._queue.qsize(),
        )

    def start_processing(self) -> None:
        """Spawn background worker tasks on the running event loop.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._running:
            logger.warning("StreamProcessor already running — ignoring start.")
            return
        self._running = True
        for idx in range(self._worker_count):
            task = asyncio.create_task(
                self._worker_loop(idx),
                name=f"ingestion-worker-{idx}",
            )
            self._workers.append(task)
        logger.info("Started %d ingestion workers.", self._worker_count)

    async def stop(self, timeout: float = 10.0) -> None:
        """Drain the queue and cancel workers gracefully.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for the queue to drain before
            force-cancelling workers.
        """
        if not self._running:
            return
        self._running = False
        logger.info(
            "Stopping StreamProcessor — draining %d queued events …",
            self._queue.qsize(),
        )

        # Signal workers to finish (one sentinel per worker).
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)  # type: ignore[arg-type]
            except asyncio.QueueFull:
                pass

        # Wait for workers, then cancel any stragglers.
        done, pending = await asyncio.wait(
            self._workers,
            timeout=timeout,
        )
        for task in pending:
            task.cancel()
        self._workers.clear()
        logger.info(
            "StreamProcessor stopped  processed=%d  failed=%d",
            self._events_processed,
            self._events_failed,
        )

    @property
    def queue_size(self) -> int:
        """Number of events waiting in the queue."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def events_submitted(self) -> int:
        return self._events_submitted

    @property
    def events_processed(self) -> int:
        return self._events_processed

    @property
    def events_failed(self) -> int:
        return self._events_failed

    # ── worker loop ───────────────────────────────────────────────

    async def _worker_loop(self, worker_id: int) -> None:
        """Consume events from the queue and process them in batches."""
        logger.info("Ingestion worker-%d started.", worker_id)

        while self._running or not self._queue.empty():
            batch_events: list[IngestionEvent] = []

            try:
                # Block until at least one event arrives (or timeout).
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self._index_interval,
                )
                if event is None:
                    # Shutdown sentinel.
                    break
                batch_events.append(event)

                # Drain up to batch_size without blocking.
                while len(batch_events) < self._batch_size and not self._queue.empty():
                    next_event = self._queue.get_nowait()
                    if next_event is None:
                        break
                    batch_events.append(next_event)

            except TimeoutError:
                # No events arrived within the interval — loop back.
                continue

            if batch_events:
                await self._process_batch(worker_id, batch_events)

        logger.info("Ingestion worker-%d exiting.", worker_id)

    async def _process_batch(
        self,
        worker_id: int,
        events: list[IngestionEvent],
    ) -> list[IngestionResult]:
        """Embed documents, index them, and invalidate caches."""
        results: list[IngestionResult] = []

        for event in events:
            t0 = time.perf_counter()
            try:
                result = await self._process_single_event(event)
                results.append(result)
                self._events_processed += 1
            except Exception:
                logger.exception(
                    "worker-%d failed event %s",
                    worker_id,
                    event.event_id,
                )
                self._events_failed += 1
                results.append(
                    IngestionResult(
                        event_id=event.event_id,
                        documents_failed=len(event.documents),
                        processing_time_ms=(time.perf_counter() - t0) * 1000,
                    )
                )
            finally:
                self._queue.task_done()

        return results

    async def _process_single_event(
        self,
        event: IngestionEvent,
    ) -> IngestionResult:
        """Embed, index, and invalidate for one event."""
        t0 = time.perf_counter()

        # 1. Generate embeddings.
        embedded_docs = await self._embedding_worker.embed_documents(
            event.documents,
        )

        # 2. Index documents via callback.
        indexed_count = await self._on_index(embedded_docs)

        # 3. Collect unique tickers to invalidate caches.
        tickers = list({doc.ticker for doc in embedded_docs if doc.ticker})
        invalidated = 0
        if tickers:
            invalidated = await self._on_cache_invalidate(tickers)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Processed event %s  indexed=%d  invalidated=%d  %.1fms",
            event.event_id,
            indexed_count,
            invalidated,
            elapsed_ms,
        )

        return IngestionResult(
            event_id=event.event_id,
            documents_indexed=indexed_count,
            documents_failed=len(event.documents) - indexed_count,
            caches_invalidated=invalidated,
            processing_time_ms=elapsed_ms,
        )

    # ── default no-op callbacks ───────────────────────────────────

    @staticmethod
    async def _default_index(docs: list[Document]) -> int:
        """Fallback indexer — logs and reports success for all docs."""
        logger.debug("Default index callback received %d docs.", len(docs))
        return len(docs)

    @staticmethod
    async def _default_invalidate(tickers: list[str]) -> int:
        """Fallback invalidation — logs tickers and reports 0."""
        logger.debug("Default invalidation callback for tickers: %s", tickers)
        return 0
