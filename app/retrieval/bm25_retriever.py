"""
Financial RAG System — BM25 Exact-Match Retriever

Uses the rank_bm25 library (BM25Okapi) to perform lexical retrieval over
a tokenized corpus of financial documents.  Optimized for ticker symbols,
financial numbers, and date expressions.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import re
import time
from typing import Sequence

from rank_bm25 import BM25Okapi

from app.config import get_config
from app.models import Document, ScoredDocument

logger = logging.getLogger(__name__)

# ── Financial-aware tokeniser ──────────────────────────────────────

# Matches tickers ($AAPL), percentages (3.5%), dollar amounts ($1.2B),
# dates (2024-01-15, Q3 2024), and ordinary words/numbers.
_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"\$[A-Z]{1,5}"           # $TICKER
    r"|[A-Z]{1,5}(?=\s)"     # bare tickers (all-caps ≤5 chars)
    r"|\d+[.,]?\d*[%BMKbmk]?" # numbers with optional suffix
    r"|Q[1-4]\s?\d{4}"       # quarter references
    r"|\d{4}[-/]\d{2}[-/]\d{2}"  # ISO dates
    r"|[A-Za-z0-9]+"         # ordinary tokens
    , re.ASCII,
)


def _tokenize(text: str) -> list[str]:
    """Tokenize *text* for BM25 with financial-domain awareness.

    Returns lower-cased tokens while preserving ticker symbols and
    numeric patterns that carry meaning in financial queries.
    """
    return [tok.upper() if tok.startswith("$") else tok.lower()
            for tok in _TOKEN_PATTERN.findall(text)]


class BM25Retriever:
    """Lexical retriever backed by BM25Okapi.

    Thread-safe for concurrent reads; writes are serialised through an
    asyncio lock so the index is never in an inconsistent state.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._default_top_k: int = cfg.retrieval.bm25_top_k

        # Parallel arrays — position *i* in every list corresponds to
        # the same document.
        self._documents: list[Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._doc_id_to_idx: dict[str, int] = {}

        # BM25 index (rebuilt after corpus mutations)
        self._index: BM25Okapi | None = None
        self._lock = asyncio.Lock()

    # ── Corpus mutations ───────────────────────────────────────────

    async def add_documents(self, docs: Sequence[Document]) -> int:
        """Add *docs* to the corpus and rebuild the BM25 index.

        Returns the number of documents actually added (duplicates by
        ``doc_id`` are silently skipped).
        """
        async with self._lock:
            added = 0
            for doc in docs:
                if doc.doc_id in self._doc_id_to_idx:
                    logger.debug("BM25: skipping duplicate doc_id=%s", doc.doc_id)
                    continue
                tokens = _tokenize(doc.content)
                idx = len(self._documents)
                self._documents.append(doc)
                self._tokenized_corpus.append(tokens)
                self._doc_id_to_idx[doc.doc_id] = idx
                added += 1

            if added:
                self._rebuild_index_unlocked()
            logger.info("BM25: added %d documents (corpus size=%d)", added, len(self._documents))
            return added

    async def remove_documents(self, doc_ids: Sequence[str]) -> int:
        """Remove documents by *doc_ids* and rebuild the index.

        Returns the number of documents actually removed.
        """
        async with self._lock:
            ids_to_remove = {did for did in doc_ids if did in self._doc_id_to_idx}
            if not ids_to_remove:
                return 0

            # Rebuild parallel arrays without the removed docs.
            new_docs: list[Document] = []
            new_tokens: list[list[str]] = []
            new_map: dict[str, int] = {}

            for doc, tokens in zip(self._documents, self._tokenized_corpus):
                if doc.doc_id in ids_to_remove:
                    continue
                new_map[doc.doc_id] = len(new_docs)
                new_docs.append(doc)
                new_tokens.append(tokens)

            self._documents = new_docs
            self._tokenized_corpus = new_tokens
            self._doc_id_to_idx = new_map
            self._rebuild_index_unlocked()

            removed = len(ids_to_remove)
            logger.info("BM25: removed %d documents (corpus size=%d)", removed, len(self._documents))
            return removed

    async def rebuild_index(self) -> None:
        """Force a full rebuild of the BM25 index."""
        async with self._lock:
            self._rebuild_index_unlocked()

    # ── Search ─────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[ScoredDocument]:
        """Search the BM25 index for *query*.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Maximum number of results to return.  Falls back to the
            configured ``bm25_top_k`` default.

        Returns
        -------
        list[ScoredDocument]
            Results sorted by descending BM25 score.
        """
        if top_k is None:
            top_k = self._default_top_k

        if self._index is None or not self._documents:
            return []

        t0 = time.perf_counter()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # BM25Okapi.get_scores is CPU-bound; run in executor to keep
        # the event loop responsive.
        loop = asyncio.get_running_loop()
        raw_scores: list[float] = await loop.run_in_executor(
            None, self._index.get_scores, query_tokens,
        )

        # Pair scores with indices and pick top-k.
        scored_indices = heapq.nlargest(
            top_k,
            enumerate(raw_scores),
            key=lambda t: t[1],
        )

        results: list[ScoredDocument] = []
        for idx, score in scored_indices:
            if score <= 0.0:
                break  # remaining are zero or negative
            results.append(ScoredDocument(
                document=self._documents[idx],
                bm25_score=float(score),
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.debug("BM25: query=%r top_k=%d results=%d elapsed=%.2fms",
                      query, top_k, len(results), elapsed_ms)
        return results

    # ── Introspection ──────────────────────────────────────────────

    @property
    def corpus_size(self) -> int:
        """Number of documents currently in the corpus."""
        return len(self._documents)

    # ── Private helpers ────────────────────────────────────────────

    def _rebuild_index_unlocked(self) -> None:
        """Rebuild the BM25Okapi index.  Caller must hold ``_lock``."""
        if self._tokenized_corpus:
            self._index = BM25Okapi(self._tokenized_corpus)
        else:
            self._index = None
        logger.debug("BM25: index rebuilt (corpus size=%d)", len(self._tokenized_corpus))
