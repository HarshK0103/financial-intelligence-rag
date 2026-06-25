"""
Financial RAG System — Hybrid Retrieval Engine

Orchestrates parallel BM25 + vector retrieval, merges results with
Reciprocal Rank Fusion (RRF), and passes the fused top-K to the
lightweight reranker.  The entire pipeline is bounded by the hard
retrieval timeout from ``LatencyBudget.retrieval_ms``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Sequence

from app.config import get_config
from app.models import ScoredDocument
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)

# RRF constant — standard value from the literature (Cormack et al. 2009).
_RRF_K: int = 60


@dataclass
class RetrievalTimings:
    """Per-stage timings for the hybrid retrieval pipeline."""

    bm25_ms: float = 0.0
    vector_ms: float = 0.0
    fusion_ms: float = 0.0
    reranking_ms: float = 0.0


class RetrievalEngine:
    """Hybrid retrieval orchestrator.

    1. Fires BM25 and vector search **in parallel**.
    2. Merges the two ranked lists with **Reciprocal Rank Fusion (RRF)**.
    3. Sends the fused top-K to the :class:`Reranker`.
    4. Enforces a hard timeout of ``retrieval_ms`` across the whole pipeline.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        reranker: Reranker,
    ) -> None:
        cfg = get_config()
        self._bm25 = bm25
        self._vector = vector
        self._reranker = reranker

        self._retrieval_timeout_s: float = cfg.latency.retrieval_ms / 1_000
        self._bm25_top_k: int = cfg.retrieval.bm25_top_k
        self._vector_top_k: int = cfg.retrieval.vector_top_k
        self._rerank_top_k: int = cfg.retrieval.rerank_top_k
        self._weight_bm25: float = cfg.retrieval.fusion_weight_bm25
        self._weight_vector: float = cfg.retrieval.fusion_weight_vector

    # ── Public API ─────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        embedding: list[float],
        top_k: int | None = None,
    ) -> list[ScoredDocument]:
        """Run the full hybrid retrieval pipeline.

        Parameters
        ----------
        query:
            Natural-language query string for BM25.
        embedding:
            Dense embedding vector for the vector retriever.
        top_k:
            Final number of reranked results to return (defaults to
            ``rerank_top_k`` from config).

        Returns
        -------
        list[ScoredDocument]
            Top-k documents sorted by ``final_score`` descending.

        Raises
        ------
        asyncio.TimeoutError
            If the pipeline exceeds ``retrieval_ms``.  Callers should
            handle this and serve partial / cached results.
        """
        if top_k is None:
            top_k = self._rerank_top_k

        t0 = time.perf_counter()

        try:
            results = await asyncio.wait_for(
                self._pipeline(query, embedding, top_k),
                timeout=self._retrieval_timeout_s,
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - t0) * 1_000
            logger.warning(
                "RetrievalEngine: timeout after %.1fms (budget=%.0fms)",
                elapsed_ms, self._retrieval_timeout_s * 1_000,
            )
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.info(
            "RetrievalEngine: query=%r results=%d elapsed=%.2fms",
            query[:80], len(results), elapsed_ms,
        )
        return results

    async def retrieve_with_timings(
        self,
        query: str,
        embedding: list[float],
        top_k: int | None = None,
    ) -> tuple[list[ScoredDocument], RetrievalTimings]:
        """Run retrieval and return both results and per-stage timings."""
        if top_k is None:
            top_k = self._rerank_top_k

        timings = RetrievalTimings()
        t0 = time.perf_counter()

        try:
            results = await asyncio.wait_for(
                self._pipeline(query, embedding, top_k, timings),
                timeout=self._retrieval_timeout_s,
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - t0) * 1_000
            logger.warning(
                "RetrievalEngine: timeout after %.1fms (budget=%.0fms)",
                elapsed_ms, self._retrieval_timeout_s * 1_000,
            )
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.info(
            "RetrievalEngine: query=%r results=%d elapsed=%.2fms",
            query[:80], len(results), elapsed_ms,
        )
        return results, timings

    # ── Private pipeline ───────────────────────────────────────────

    async def _pipeline(
        self,
        query: str,
        embedding: list[float],
        top_k: int,
        timings: RetrievalTimings | None = None,
    ) -> list[ScoredDocument]:
        """Execute BM25 ∥ vector → fuse → rerank."""

        async def _timed_bm25() -> list[ScoredDocument]:
            started = time.perf_counter()
            results = await self._bm25.search(query, top_k=self._bm25_top_k)
            if timings is not None:
                timings.bm25_ms = round((time.perf_counter() - started) * 1_000, 2)
            return results

        async def _timed_vector() -> list[ScoredDocument]:
            started = time.perf_counter()
            results = await self._vector.search(embedding, top_k=self._vector_top_k)
            if timings is not None:
                timings.vector_ms = round((time.perf_counter() - started) * 1_000, 2)
            return results

        # 1. Parallel retrieval.
        bm25_results, vector_results = await asyncio.gather(
            _timed_bm25(),
            _timed_vector(),
            return_exceptions=True,
        )

        # Gracefully degrade if one retriever fails.
        if isinstance(bm25_results, BaseException):
            logger.error("BM25 retriever failed: %s", bm25_results)
            bm25_results = []
        if isinstance(vector_results, BaseException):
            logger.error("Vector retriever failed: %s", vector_results)
            vector_results = []

        # 2. Reciprocal Rank Fusion.
        fusion_start = time.perf_counter()
        fused = self._reciprocal_rank_fusion(
            bm25_results, vector_results,
        )
        if timings is not None:
            timings.fusion_ms = round((time.perf_counter() - fusion_start) * 1_000, 2)

        # 3. Rerank the fused candidates.
        # Pass more candidates to the reranker than needed so it has room
        # to re-order.
        rerank_candidates = fused[: max(top_k * 3, self._rerank_top_k * 2)]
        rerank_start = time.perf_counter()
        reranked = await self._reranker.rerank(query, rerank_candidates, top_k=top_k)
        if timings is not None:
            timings.reranking_ms = round((time.perf_counter() - rerank_start) * 1_000, 2)

        return reranked

    # ── Reciprocal Rank Fusion ─────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        bm25_results: Sequence[ScoredDocument],
        vector_results: Sequence[ScoredDocument],
    ) -> list[ScoredDocument]:
        """Merge two ranked lists with weighted RRF.

        RRF score for document *d*:

            score(d) = Σ  w_i / (k + rank_i(d))

        where *k* = 60 (standard constant) and *w_i* is the per-source
        weight (``fusion_weight_bm25``, ``fusion_weight_vector``).
        """
        # Collect per-document RRF score + best ScoredDocument record.
        doc_scores: dict[str, float] = {}
        doc_records: dict[str, ScoredDocument] = {}

        for rank, sdoc in enumerate(bm25_results, start=1):
            did = sdoc.document.doc_id
            rrf_contribution = self._weight_bm25 / (_RRF_K + rank)
            doc_scores[did] = doc_scores.get(did, 0.0) + rrf_contribution
            # Keep the record with the highest individual score.
            if did not in doc_records or sdoc.bm25_score > doc_records[did].bm25_score:
                doc_records[did] = sdoc

        for rank, sdoc in enumerate(vector_results, start=1):
            did = sdoc.document.doc_id
            rrf_contribution = self._weight_vector / (_RRF_K + rank)
            doc_scores[did] = doc_scores.get(did, 0.0) + rrf_contribution
            existing = doc_records.get(did)
            if existing is None:
                doc_records[did] = sdoc
            else:
                # Merge scores from both retrievers onto the same record.
                doc_records[did] = existing.model_copy(update={
                    "vector_score": sdoc.vector_score,
                })

        # Build the fused list sorted by RRF score.
        fused: list[ScoredDocument] = []
        for did in sorted(doc_scores, key=doc_scores.__getitem__, reverse=True):
            record = doc_records[did]
            fused.append(record.model_copy(update={
                "final_score": round(doc_scores[did], 6),
            }))

        return fused
