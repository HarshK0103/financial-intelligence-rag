"""
Financial RAG System — L2 Semantic Similarity Cache

Stores query embeddings alongside responses and returns a cached
response when an incoming query embedding has cosine similarity ≥
threshold against a stored embedding.  Uses NumPy for fast vectorised
similarity computation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Final

import numpy as np
from numpy.typing import NDArray

from app.config import get_config
from app.models import CacheEntry, CacheLayer, QueryResponse

logger = logging.getLogger(__name__)


class SemanticCache:
    """L2 semantic similarity cache.

    On :meth:`get`, the incoming query embedding is compared against
    every stored embedding via cosine similarity.  If the highest
    similarity exceeds ``threshold`` **and** the entry is not expired,
    the cached response is returned.

    Internally the embeddings are kept in a contiguous NumPy matrix
    so that the cosine comparison of one query against *N* entries is a
    single vectorised operation — O(N·d) but with very small constant
    factor thanks to BLAS.

    Attributes:
        hits:   Number of cache hits since creation.
        misses: Number of cache misses since creation.
    """

    _EPSILON: Final[float] = 1e-10  # Avoid division-by-zero in norm

    def __init__(self) -> None:
        cfg = get_config().cache
        self._threshold: float = cfg.l2_similarity_threshold
        self._ttl: float = float(cfg.l2_ttl_seconds)
        self._max_entries: int = cfg.l2_max_entries

        # Parallel lists — kept in sync under lock
        self._entries: list[CacheEntry] = []
        self._embeddings: list[NDArray[np.float32]] = []
        self._embedding_matrix: NDArray[np.float32] | None = None
        self._embedding_norms: NDArray[np.float32] | None = None

        self._lock = asyncio.Lock()

        # Metrics
        self.hits: int = 0
        self.misses: int = 0

        logger.info(
            "SemanticCache initialised — threshold=%.3f, ttl=%ss, max_entries=%d",
            self._threshold,
            self._ttl,
            self._max_entries,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(
        self,
        query: str,
        embedding: list[float] | NDArray[np.float32],
    ) -> QueryResponse | None:
        """Return a cached response if a semantically similar query exists.

        Parameters:
            query:     The raw query string (used only for logging).
            embedding: The query embedding vector.

        Returns:
            The cached :class:`QueryResponse` if similarity ≥ threshold,
            otherwise ``None``.
        """
        query_vec = np.asarray(embedding, dtype=np.float32)

        async with self._lock:
            # Prune expired entries first
            self._prune_expired()

            if not self._embeddings:
                self.misses += 1
                return None

            # Vectorised cosine similarity against all stored embeddings
            matrix = self._embedding_matrix
            if matrix is None:
                self._rebuild_matrix_unlocked()
                matrix = self._embedding_matrix
            if matrix is None:
                self.misses += 1
                return None
            similarities = self._cosine_similarity_batch(query_vec, matrix)

            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= self._threshold:
                entry = self._entries[best_idx]
                entry.access_count += 1
                self.hits += 1
                logger.debug(
                    "SemanticCache HIT — sim=%.4f (threshold=%.3f) query=%r",
                    best_sim,
                    self._threshold,
                    query[:60],
                )
                return entry.response

        self.misses += 1
        logger.debug(
            "SemanticCache MISS — best_sim=%.4f (threshold=%.3f) query=%r",
            best_sim if self._embeddings else 0.0,
            self._threshold,
            query[:60],
        )
        return None

    async def set(
        self,
        query: str,
        embedding: list[float] | NDArray[np.float32],
        response: QueryResponse,
        tickers: list[str] | None = None,
    ) -> None:
        """Store a query-response pair with its embedding.

        Parameters:
            query:     The original query string.
            embedding: The query embedding vector.
            response:  The response to cache.
            tickers:   Optional list of tickers for invalidation.
        """
        query_vec = np.asarray(embedding, dtype=np.float32)

        entry = CacheEntry(
            query=query,
            response=response,
            embedding=query_vec.tolist(),
            created_at=time.time(),
            ttl_seconds=self._ttl,
            tickers=tickers or [],
        )

        async with self._lock:
            self._entries.append(entry)
            self._embeddings.append(query_vec)
            self._evict_if_needed()
            self._rebuild_matrix_unlocked()

        logger.debug("SemanticCache SET — query=%r (entries=%d)", query[:60], len(self._entries))

    async def invalidate_by_tickers(self, tickers: list[str]) -> int:
        """Remove all entries whose tickers overlap with *tickers*.

        Returns the count of removed entries.
        """
        if not tickers:
            return 0

        ticker_set = {t.upper() for t in tickers}
        removed = 0

        async with self._lock:
            keep_entries: list[CacheEntry] = []
            keep_embeddings: list[NDArray[np.float32]] = []
            for entry, emb in zip(self._entries, self._embeddings, strict=True):
                if ticker_set.intersection(t.upper() for t in entry.tickers):
                    removed += 1
                else:
                    keep_entries.append(entry)
                    keep_embeddings.append(emb)
            self._entries = keep_entries
            self._embeddings = keep_embeddings
            self._rebuild_matrix_unlocked()

        if removed:
            logger.info(
                "SemanticCache invalidated %d entries for tickers %s",
                removed,
                tickers,
            )
        return removed

    async def clear(self) -> None:
        """Remove all entries from the cache."""
        async with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._embeddings.clear()
            self._embedding_matrix = None
            self._embedding_norms = None
        logger.info("SemanticCache CLEARED %d entries", count)

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
        """Current number of cached entries."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine_similarity_batch(
        self,
        query: NDArray[np.float32],
        matrix: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute cosine similarity of *query* against each row of *matrix*.

        Parameters:
            query:  1-D vector of shape ``(d,)``.
            matrix: 2-D matrix of shape ``(N, d)``.

        Returns:
            1-D array of similarities of shape ``(N,)``.
        """
        query_norm = np.linalg.norm(query) + self._EPSILON
        matrix_norms = self._embedding_norms
        if matrix_norms is None:
            matrix_norms = np.linalg.norm(matrix, axis=1) + self._EPSILON
            self._embedding_norms = matrix_norms
        return (matrix @ query) / (matrix_norms * query_norm)

    def _prune_expired(self) -> None:
        """Remove all expired entries.  Must be called under lock."""
        now = time.time()
        keep_entries: list[CacheEntry] = []
        keep_embeddings: list[NDArray[np.float32]] = []
        pruned = 0
        for entry, emb in zip(self._entries, self._embeddings, strict=True):
            if (now - entry.created_at) > entry.ttl_seconds:
                pruned += 1
            else:
                keep_entries.append(entry)
                keep_embeddings.append(emb)
        if pruned:
            self._entries = keep_entries
            self._embeddings = keep_embeddings
            self._rebuild_matrix_unlocked()
            logger.debug("SemanticCache pruned %d expired entries", pruned)

    def _evict_if_needed(self) -> None:
        """Evict oldest entries when over capacity.  Must be called under lock."""
        while len(self._entries) > self._max_entries:
            self._entries.pop(0)
            self._embeddings.pop(0)
            logger.debug("SemanticCache EVICTED oldest entry")
        if not self._entries:
            self._embedding_matrix = None
            self._embedding_norms = None

    def _rebuild_matrix_unlocked(self) -> None:
        """Rebuild the embedding matrix. Must be called under lock."""
        if not self._embeddings:
            self._embedding_matrix = None
            self._embedding_norms = None
            return
        self._embedding_matrix = np.vstack(self._embeddings)
        self._embedding_norms = (
            np.linalg.norm(self._embedding_matrix, axis=1) + self._EPSILON
        )
