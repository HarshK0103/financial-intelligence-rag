"""
Financial RAG System — FAISS Vector Retriever

Uses a FAISS IndexFlatIP (inner product on L2-normalised vectors ≡ cosine
similarity) for dense embedding retrieval over financial document chunks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Sequence

import faiss
import numpy as np
from numpy.typing import NDArray

from app.config import get_config
from app.models import Document, ScoredDocument

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Dense retriever backed by a FAISS inner-product index.

    Embeddings are L2-normalised before insertion so that inner-product
    scores are equivalent to cosine similarity (range [−1, 1]).

    Thread-safe for concurrent reads; writes are serialised through an
    asyncio lock.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._dim: int = cfg.retrieval.embedding_dim
        self._default_top_k: int = cfg.retrieval.vector_top_k

        # FAISS index — flat inner-product (exact search, no training).
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self._dim)

        # Mapping from FAISS *row position* → Document.
        self._position_to_doc: dict[int, Document] = {}
        self._doc_id_to_position: dict[str, int] = {}
        self._next_position: int = 0

        self._lock = asyncio.Lock()

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2-normalise each row in *vectors* in-place and return it."""
        faiss.normalize_L2(vectors)
        return vectors

    def _embedding_to_array(self, embedding: list[float]) -> NDArray[np.float32]:
        """Convert a single embedding list to a (1, dim) float32 array."""
        arr = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != self._dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self._dim}, " f"got {arr.shape[1]}")
        return arr

    # ── Corpus mutations ───────────────────────────────────────────

    async def add_documents(
        self,
        docs_with_embeddings: Sequence[tuple[Document, list[float]]],
    ) -> int:
        """Add documents along with their pre-computed embeddings.

        Parameters
        ----------
        docs_with_embeddings:
            Sequence of ``(Document, embedding)`` tuples.  Each
            *embedding* must be a float list of length ``embedding_dim``.

        Returns
        -------
        int
            Number of documents actually added (duplicates skipped).
        """
        async with self._lock:
            vectors_to_add: list[NDArray[np.float32]] = []
            docs_to_add: list[Document] = []

            for doc, emb in docs_with_embeddings:
                if doc.doc_id in self._doc_id_to_position:
                    logger.debug("Vector: skipping duplicate doc_id=%s", doc.doc_id)
                    continue
                arr = self._embedding_to_array(emb)
                vectors_to_add.append(arr)
                docs_to_add.append(doc)

            if not vectors_to_add:
                return 0

            batch = np.vstack(vectors_to_add)
            self._normalize(batch)

            # Run the FAISS add in the default executor (CPU-bound).
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._index.add, batch)

            for doc in docs_to_add:
                pos = self._next_position
                self._position_to_doc[pos] = doc
                self._doc_id_to_position[doc.doc_id] = pos
                self._next_position += 1

            logger.info(
                "Vector: added %d documents (index size=%d)",
                len(docs_to_add),
                self._index.ntotal,
            )
            return len(docs_to_add)

    async def remove_documents(self, doc_ids: Sequence[str]) -> int:
        """Remove documents by *doc_ids* and rebuild the FAISS index.

        FAISS ``IndexFlatIP`` does not support in-place deletion, so
        the entire index is rebuilt from the remaining vectors.

        Returns the number of documents actually removed.
        """
        async with self._lock:
            ids_to_remove = {did for did in doc_ids if did in self._doc_id_to_position}
            if not ids_to_remove:
                return 0

            # Collect surviving docs and their positions in the *old* index.
            surviving: list[tuple[int, Document]] = []
            for pos in sorted(self._position_to_doc):
                doc = self._position_to_doc[pos]
                if doc.doc_id not in ids_to_remove:
                    surviving.append((pos, doc))

            # Reconstruct vectors from old index.
            new_index = faiss.IndexFlatIP(self._dim)
            new_pos_to_doc: dict[int, Document] = {}
            new_id_to_pos: dict[str, int] = {}

            if surviving:
                old_positions = np.array([p for p, _ in surviving], dtype=np.int64)
                vectors = np.empty((len(surviving), self._dim), dtype=np.float32)
                for i, pos in enumerate(old_positions):
                    vectors[i] = self._index.reconstruct(int(pos))
                # Already normalised from insertion — no need to re-normalise.
                new_index.add(vectors)

                for new_pos, (_, doc) in enumerate(surviving):
                    new_pos_to_doc[new_pos] = doc
                    new_id_to_pos[doc.doc_id] = new_pos

            self._index = new_index
            self._position_to_doc = new_pos_to_doc
            self._doc_id_to_position = new_id_to_pos
            self._next_position = len(surviving)

            removed = len(ids_to_remove)
            logger.info("Vector: removed %d documents (index size=%d)", removed, self._index.ntotal)
            return removed

    # ── Search ─────────────────────────────────────────────────────

    async def search(
        self,
        embedding: list[float],
        top_k: int | None = None,
    ) -> list[ScoredDocument]:
        """Search the FAISS index for the nearest neighbours of *embedding*.

        Parameters
        ----------
        embedding:
            Query embedding (float list of length ``embedding_dim``).
        top_k:
            Maximum results to return.  Falls back to configured
            ``vector_top_k``.

        Returns
        -------
        list[ScoredDocument]
            Results sorted by descending cosine similarity.
        """
        if top_k is None:
            top_k = self._default_top_k

        if self._index.ntotal == 0:
            return []

        t0 = time.perf_counter()
        query_vec = self._embedding_to_array(embedding)
        self._normalize(query_vec)

        # Clamp top_k to available vectors.
        effective_k = min(top_k, self._index.ntotal)

        loop = asyncio.get_running_loop()
        distances, indices = await loop.run_in_executor(
            None,
            self._index.search,
            query_vec,
            effective_k,
        )

        results: list[ScoredDocument] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # FAISS sentinel for "no result"
            doc = self._position_to_doc.get(int(idx))
            if doc is None:
                logger.warning("Vector: position %d not found in mapping", idx)
                continue
            results.append(
                ScoredDocument(
                    document=doc,
                    vector_score=float(dist),
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.debug(
            "Vector: top_k=%d results=%d elapsed=%.2fms",
            top_k,
            len(results),
            elapsed_ms,
        )
        return results

    # ── Introspection ──────────────────────────────────────────────

    @property
    def index_size(self) -> int:
        """Number of vectors currently in the FAISS index."""
        return self._index.ntotal
