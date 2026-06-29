"""
Financial RAG System — Hot Store (Real-Time Data)

In-memory store for real-time financial documents: live prices, breaking
news, order-book snapshots, etc.  Entries auto-expire after a
configurable window (default 5 minutes) to guarantee freshness.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Sequence

from app.models import Document

logger = logging.getLogger(__name__)

# Default time-to-live for hot documents (seconds).
_DEFAULT_TTL_SECONDS: float = 300.0  # 5 minutes


class HotStore:
    """Thread-safe in-memory store for real-time financial documents.

    Expired entries are lazily evicted on every read/write operation and
    can also be purged explicitly via :meth:`evict_expired`.
    """

    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
        self._ttl: float = ttl_seconds

        # Primary store: doc_id → Document
        self._docs: dict[str, Document] = {}
        # Secondary index: ticker → set of doc_ids
        self._ticker_index: dict[str, set[str]] = {}
        # Expiry tracking: doc_id → expiry timestamp
        self._expiry: dict[str, float] = {}

        self._lock = asyncio.Lock()

    # ── Write ──────────────────────────────────────────────────────

    async def add_document(self, doc: Document) -> None:
        """Insert or replace a document in the hot store."""
        async with self._lock:
            self._evict_expired_unlocked()
            self._insert_unlocked(doc)
            logger.debug("HotStore: added doc_id=%s ticker=%s", doc.doc_id, doc.ticker)

    async def add_documents(self, docs: Sequence[Document]) -> int:
        """Batch-insert documents.  Returns the count added."""
        async with self._lock:
            self._evict_expired_unlocked()
            for doc in docs:
                self._insert_unlocked(doc)
            logger.info("HotStore: batch added %d documents", len(docs))
            return len(docs)

    # ── Read ───────────────────────────────────────────────────────

    async def get_document(self, doc_id: str) -> Document | None:
        """Return a single document by *doc_id*, or ``None`` if missing/expired."""
        async with self._lock:
            self._evict_expired_unlocked()
            return self._docs.get(doc_id)

    async def get_by_ticker(self, ticker: str) -> list[Document]:
        """Return all live documents for *ticker*."""
        async with self._lock:
            self._evict_expired_unlocked()
            ids = self._ticker_index.get(ticker.upper(), set())
            return [self._docs[did] for did in ids if did in self._docs]

    async def get_all(self) -> list[Document]:
        """Return all non-expired documents."""
        async with self._lock:
            self._evict_expired_unlocked()
            return list(self._docs.values())

    # ── Management ─────────────────────────────────────────────────

    async def clear(self) -> int:
        """Remove **all** documents.  Returns the count removed."""
        async with self._lock:
            count = len(self._docs)
            self._docs.clear()
            self._ticker_index.clear()
            self._expiry.clear()
            logger.info("HotStore: cleared %d documents", count)
            return count

    async def count(self) -> int:
        """Return the number of live (non-expired) documents."""
        async with self._lock:
            self._evict_expired_unlocked()
            return len(self._docs)

    async def evict_expired(self) -> int:
        """Explicitly evict all expired entries.  Returns count evicted."""
        async with self._lock:
            return self._evict_expired_unlocked()

    # ── Private helpers ────────────────────────────────────────────

    def _insert_unlocked(self, doc: Document) -> None:
        """Insert a single document (caller must hold ``_lock``)."""
        # Remove old entry if overwriting.
        if doc.doc_id in self._docs:
            self._remove_unlocked(doc.doc_id)

        self._docs[doc.doc_id] = doc
        self._expiry[doc.doc_id] = time.time() + self._ttl

        if doc.ticker:
            ticker_key = doc.ticker.upper()
            self._ticker_index.setdefault(ticker_key, set()).add(doc.doc_id)

    def _remove_unlocked(self, doc_id: str) -> None:
        """Remove a single document by id (caller must hold ``_lock``)."""
        doc = self._docs.pop(doc_id, None)
        self._expiry.pop(doc_id, None)
        if doc and doc.ticker:
            ticker_key = doc.ticker.upper()
            id_set = self._ticker_index.get(ticker_key)
            if id_set:
                id_set.discard(doc_id)
                if not id_set:
                    del self._ticker_index[ticker_key]

    def _evict_expired_unlocked(self) -> int:
        """Remove all expired documents.  Returns count evicted."""
        now = time.time()
        expired_ids = [did for did, exp in self._expiry.items() if exp <= now]
        for did in expired_ids:
            self._remove_unlocked(did)
        if expired_ids:
            logger.debug("HotStore: evicted %d expired documents", len(expired_ids))
        return len(expired_ids)
