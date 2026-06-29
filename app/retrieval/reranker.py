"""
Financial RAG System — Lightweight Reranker

A fast heuristic reranker that stays within a 20 ms budget.  It scores
documents using three signals:

1. **Query-term overlap** — fraction of query tokens found in the document,
   with bonus weight for ticker symbols and financial numbers.
2. **Position score** — documents already ranked higher by the retriever
   receive a small positional bonus (decays with rank).
3. **Freshness score** — exponential decay based on document age, using the
   configurable half-life from ``ConsistencyConfig``.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Sequence

from app.config import get_config
from app.models import ScoredDocument

logger = logging.getLogger(__name__)

# ── Token helpers ──────────────────────────────────────────────────

_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Za-z0-9$%]+", re.ASCII)
_TICKER_RE: re.Pattern[str] = re.compile(r"^\$?[A-Z]{1,5}$", re.ASCII)
_NUMBER_RE: re.Pattern[str] = re.compile(r"^\d", re.ASCII)


def _tokenize_lower(text: str) -> list[str]:
    """Quick whitespace-aware tokenisation, lower-cased."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def _is_high_value_token(token: str) -> bool:
    """Return True if *token* is a ticker symbol or numeric value."""
    upper = token.upper()
    return bool(_TICKER_RE.match(upper) or _NUMBER_RE.match(token))


# ── Scoring weights ───────────────────────────────────────────────

_WEIGHT_OVERLAP: float = 0.50
_WEIGHT_POSITION: float = 0.15
_WEIGHT_FRESHNESS: float = 0.35

# Bonus multiplier for high-value (ticker / number) token matches
_HIGH_VALUE_BONUS: float = 1.5


class Reranker:
    """Lightweight heuristic reranker for financial document scoring.

    Designed to add < 20 ms overhead even on large candidate lists.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._default_top_k: int = cfg.retrieval.rerank_top_k
        self._freshness_halflife: float = cfg.consistency.freshness_decay_halflife_seconds
        self._token_cache_key: str = "_reranker_token_set"

    # ── Public API ─────────────────────────────────────────────────

    async def rerank(
        self,
        query: str,
        documents: Sequence[ScoredDocument],
        top_k: int | None = None,
    ) -> list[ScoredDocument]:
        """Rerank *documents* using heuristic signals.

        Parameters
        ----------
        query:
            The original natural-language query.
        documents:
            Candidate ``ScoredDocument``s produced by the retriever stage.
        top_k:
            How many to return after reranking.  Falls back to the
            configured ``rerank_top_k``.

        Returns
        -------
        list[ScoredDocument]
            Top-k documents with ``rerank_score``, ``freshness_score``,
            and ``final_score`` populated, sorted by ``final_score``
            descending.
        """
        if top_k is None:
            top_k = self._default_top_k

        if not documents:
            return []

        t0 = time.perf_counter()
        now = time.time()

        query_tokens = _tokenize_lower(query)

        scored: list[ScoredDocument] = []
        n_docs = len(documents)

        for rank, sdoc in enumerate(documents):
            overlap = self._compute_overlap(
                query_tokens,
                sdoc,
                self._token_cache_key,
            )
            position = self._compute_position_score(rank, n_docs)
            freshness = self._compute_freshness(sdoc.document.timestamp, now)

            rerank = _WEIGHT_OVERLAP * overlap + _WEIGHT_POSITION * position + _WEIGHT_FRESHNESS * freshness

            # Blend retriever score and reranker score.
            retriever_score = max(sdoc.bm25_score, sdoc.vector_score, sdoc.final_score)
            final = 0.4 * retriever_score + 0.6 * rerank

            scored.append(
                sdoc.model_copy(
                    update={
                        "rerank_score": round(rerank, 6),
                        "freshness_score": round(freshness, 6),
                        "final_score": round(final, 6),
                    }
                )
            )

        scored.sort(key=lambda s: s.final_score, reverse=True)
        results = scored[:top_k]

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.debug(
            "Reranker: query=%r candidates=%d returned=%d elapsed=%.2fms",
            query,
            n_docs,
            len(results),
            elapsed_ms,
        )
        return results

    # ── Scoring components ─────────────────────────────────────────

    @staticmethod
    def _compute_overlap(
        query_tokens: list[str],
        sdoc: ScoredDocument,
        token_cache_key: str,
    ) -> float:
        """Compute weighted query-term overlap score in [0, 1]."""
        if not query_tokens:
            return 0.0

        cached_tokens = sdoc.document.metadata.get(token_cache_key)
        if not isinstance(cached_tokens, list):
            cached_tokens = _tokenize_lower(sdoc.document.content)
            sdoc.document.metadata[token_cache_key] = cached_tokens
        doc_tokens = set(cached_tokens)
        total_weight = 0.0
        matched_weight = 0.0

        for token in query_tokens:
            weight = _HIGH_VALUE_BONUS if _is_high_value_token(token) else 1.0
            total_weight += weight
            if token in doc_tokens:
                matched_weight += weight

        return matched_weight / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _compute_position_score(rank: int, total: int) -> float:
        """Positional score that decays with rank.  Range [0, 1]."""
        if total <= 1:
            return 1.0
        return 1.0 - (rank / (total - 1))

    def _compute_freshness(self, doc_timestamp: float, now: float) -> float:
        """Exponential freshness decay.  Range (0, 1]."""
        age_seconds = max(now - doc_timestamp, 0.0)
        if self._freshness_halflife <= 0:
            return 1.0
        # f(t) = 2^(-t / half_life)
        return math.pow(2.0, -age_seconds / self._freshness_halflife)
