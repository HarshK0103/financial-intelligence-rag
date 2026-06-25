"""
Financial RAG System — Background Embedding Worker

Generates vector embeddings for newly ingested documents.  In this
prototype the embeddings are *simulated* (random unit vectors of
dimension 384) so the module has zero external model dependencies.

Key design choices
──────────────────
* **Rate limiting** — an asyncio.Semaphore caps the number of
  concurrent embedding calls so that retrieval latency is never
  starved of CPU.
* **Batch processing** — documents are embedded in configurable
  batches to amortise overhead and play nicely with future GPU
  back-ends.
* All public methods are async to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Sequence

from app.config import get_config
from app.embedding.embedding_service import EmbeddingService
from app.models import Document

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """Generate embeddings for :class:`Document` instances.

    Parameters
    ----------
    embedding_dim : int | None
        Dimensionality of the embedding vectors.  Falls back to
        ``config.retrieval.embedding_dim`` (384) when *None*.
    max_concurrency : int
        Maximum number of embedding batches that may run in parallel.
        Keeps CPU pressure bounded so the retrieval hot-path stays
        responsive.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        embedding_dim: int | None = None,
        max_concurrency: int = 4,
    ) -> None:
        cfg = get_config()
        self._dim: int = embedding_dim or cfg.retrieval.embedding_dim
        self._rate_limit: float = cfg.ingestion.rate_limit_per_second
        self._batch_size: int = cfg.ingestion.batch_size
        self._embedding_service = embedding_service or EmbeddingService()
        self._semaphore = asyncio.Semaphore(max_concurrency)

        # Token-bucket state for rate limiting
        self._tokens: float = self._rate_limit
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

        # Counters
        self._total_embedded: int = 0
        self._total_errors: int = 0

        logger.info(
            "EmbeddingWorker initialised  dim=%d  rate_limit=%.1f/s  "
            "batch_size=%d  max_concurrency=%d",
            self._dim,
            self._rate_limit,
            self._batch_size,
            max_concurrency,
        )

    # ── public API ────────────────────────────────────────────────

    async def embed_documents(
        self,
        docs: list[Document],
    ) -> list[Document]:
        """Embed a list of documents, returning them with ``.embedding`` populated.

        Documents that *already* carry an embedding are silently
        skipped (idempotent).  Batching and rate-limiting are applied
        transparently.

        Returns a *new* list of :class:`Document` instances (the
        originals are never mutated).
        """
        if not docs:
            return []

        results: list[Document] = []

        for batch_start in range(0, len(docs), self._batch_size):
            batch = docs[batch_start : batch_start + self._batch_size]
            embedded_batch = await self._embed_batch(batch)
            results.extend(embedded_batch)

        return results

    async def embed_single(self, doc: Document) -> Document:
        """Convenience wrapper around :meth:`embed_documents` for one doc."""
        embedded = await self.embed_documents([doc])
        return embedded[0]

    @property
    def total_embedded(self) -> int:
        """Number of documents successfully embedded so far."""
        return self._total_embedded

    @property
    def total_errors(self) -> int:
        """Number of embedding failures so far."""
        return self._total_errors

    # ── internals ─────────────────────────────────────────────────

    async def _embed_batch(
        self,
        batch: Sequence[Document],
    ) -> list[Document]:
        """Embed one batch, respecting the concurrency semaphore."""
        async with self._semaphore:
            await self._wait_for_token(len(batch))
            return await self._generate_embeddings(list(batch))

    async def _generate_embeddings(
        self,
        docs: list[Document],
    ) -> list[Document]:
        """Generate (simulated) embedding vectors for *docs*.

        In production this would call a model server (e.g.
        ``sentence-transformers`` via ``torch``).  Here we produce
        random unit vectors so the rest of the pipeline can be
        exercised end-to-end.
        """
        result: list[Document] = []
        pending_docs = [doc for doc in docs if doc.embedding is None]
        vectors = await self._embedding_service.embed_texts(
            [doc.content for doc in pending_docs]
        )
        vectors_by_id = {
            doc.doc_id: vector for doc, vector in zip(pending_docs, vectors, strict=True)
        }

        for doc in docs:
            try:
                if doc.embedding is not None:
                    # Already embedded — pass through unchanged.
                    result.append(doc)
                    continue

                # Return a *copy* with the embedding set.
                updated = doc.model_copy(update={"embedding": vectors_by_id[doc.doc_id]})
                result.append(updated)
                self._total_embedded += 1

            except Exception:
                logger.exception(
                    "Failed to embed doc_id=%s", doc.doc_id,
                )
                self._total_errors += 1
                # Still include the doc — downstream can decide what
                # to do with unembedded documents.
                result.append(doc)

        logger.debug(
            "Embedded batch of %d docs  (total=%d  errors=%d)",
            len(docs),
            self._total_embedded,
            self._total_errors,
        )
        # Simulate a small amount of compute latency.
        await asyncio.sleep(0.001 * len(docs))
        return result

    # ── rate limiting (token bucket) ──────────────────────────────

    async def _wait_for_token(self, count: int = 1) -> None:
        """Block until *count* tokens are available in the bucket."""
        async with self._lock:
            self._refill_tokens()
            while self._tokens < count:
                # Sleep just long enough to accumulate what we need.
                deficit = count - self._tokens
                sleep_s = deficit / self._rate_limit
                await asyncio.sleep(sleep_s)
                self._refill_tokens()
            self._tokens -= count

    def _refill_tokens(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._rate_limit,                 # bucket capacity
            self._tokens + elapsed * self._rate_limit,
        )
        self._last_refill = now
