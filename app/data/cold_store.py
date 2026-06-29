"""
Financial RAG System — Cold Store (Historical Data)

In-memory store for historical financial documents: SEC filings, analyst
reports, earnings transcripts, research notes, etc.  Documents in the
cold store do **not** expire — they persist for the lifetime of the
process.

This is a prototype implementation using plain dicts; the interface is
designed so that a persistent back-end (e.g. PostgreSQL, S3) can be
swapped in without changing callers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from app.models import Document

logger = logging.getLogger(__name__)


class ColdStore:
    """Persistent (in-memory prototype) store for historical documents.

    Thread-safe via an asyncio lock.  All public methods are async to
    allow a future migration to I/O-backed storage.
    """

    def __init__(self) -> None:
        # Primary store: doc_id → Document
        self._docs: dict[str, Document] = {}
        # Secondary index: ticker → set of doc_ids
        self._ticker_index: dict[str, set[str]] = {}

        self._lock = asyncio.Lock()

    # ── Write ──────────────────────────────────────────────────────

    async def add_document(self, doc: Document) -> None:
        """Insert or replace a single document."""
        async with self._lock:
            self._insert_unlocked(doc)
            logger.debug("ColdStore: added doc_id=%s ticker=%s", doc.doc_id, doc.ticker)

    async def add_documents(self, docs: Sequence[Document]) -> int:
        """Batch-insert documents.  Returns count added."""
        async with self._lock:
            for doc in docs:
                self._insert_unlocked(doc)
            logger.info("ColdStore: batch added %d documents", len(docs))
            return len(docs)

    # ── Read ───────────────────────────────────────────────────────

    async def get_document(self, doc_id: str) -> Document | None:
        """Return a document by *doc_id*, or ``None`` if not found."""
        async with self._lock:
            return self._docs.get(doc_id)

    async def get_by_ticker(self, ticker: str) -> list[Document]:
        """Return all documents associated with *ticker*."""
        async with self._lock:
            ids = self._ticker_index.get(ticker.upper(), set())
            return [self._docs[did] for did in ids if did in self._docs]

    async def search_by_content(
        self,
        query: str,
        *,
        ticker: str | None = None,
        max_results: int = 50,
    ) -> list[Document]:
        """Brute-force substring search over document content.

        This is a prototype search — production would delegate to a
        proper full-text index.

        Parameters
        ----------
        query:
            Case-insensitive substring to match against
            ``Document.content``.
        ticker:
            If given, restrict search to documents with this ticker.
        max_results:
            Maximum number of matches to return.

        Returns
        -------
        list[Document]
            Matching documents, ordered by timestamp descending
            (newest first).
        """
        async with self._lock:
            query_lower = query.lower()
            candidates: list[Document]

            if ticker:
                ids = self._ticker_index.get(ticker.upper(), set())
                candidates = [self._docs[did] for did in ids if did in self._docs]
            else:
                candidates = list(self._docs.values())

            matches = [doc for doc in candidates if query_lower in doc.content.lower()]

            # Sort by timestamp descending (newest first).
            matches.sort(key=lambda d: d.timestamp, reverse=True)
            return matches[:max_results]

    # ── Management ─────────────────────────────────────────────────

    async def remove_documents(self, doc_ids: Sequence[str]) -> int:
        """Remove documents by *doc_ids*.  Returns count removed."""
        async with self._lock:
            removed = 0
            for did in doc_ids:
                if self._remove_unlocked(did):
                    removed += 1
            if removed:
                logger.info("ColdStore: removed %d documents", removed)
            return removed

    async def count(self) -> int:
        """Return total number of documents in the cold store."""
        async with self._lock:
            return len(self._docs)

    async def clear(self) -> int:
        """Remove all documents.  Returns count cleared."""
        async with self._lock:
            n = len(self._docs)
            self._docs.clear()
            self._ticker_index.clear()
            logger.info("ColdStore: cleared %d documents", n)
            return n

    # ── Private helpers ────────────────────────────────────────────

    def _insert_unlocked(self, doc: Document) -> None:
        """Insert a single document (caller must hold ``_lock``)."""
        # Remove stale entry if overwriting.
        if doc.doc_id in self._docs:
            self._remove_unlocked(doc.doc_id)

        self._docs[doc.doc_id] = doc
        if doc.ticker:
            ticker_key = doc.ticker.upper()
            self._ticker_index.setdefault(ticker_key, set()).add(doc.doc_id)

    def _remove_unlocked(self, doc_id: str) -> bool:
        """Remove a single document.  Returns True if found."""
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return False
        if doc.ticker:
            ticker_key = doc.ticker.upper()
            id_set = self._ticker_index.get(ticker_key)
            if id_set:
                id_set.discard(doc_id)
                if not id_set:
                    del self._ticker_index[ticker_key]
        return True
