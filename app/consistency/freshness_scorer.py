"""
Financial RAG System — Freshness Scorer

Computes a time-decay freshness score for each document based on
its timestamp relative to query time.  Uses exponential decay
parameterised by the half-life from ``ConsistencyConfig``.
"""

from __future__ import annotations

import logging
import math

from app.config import get_config
from app.models import Document, ScoredDocument

logger = logging.getLogger(__name__)


class FreshnessScorer:
    """Timestamp-aware freshness scoring.

    Each document receives a freshness score in ``[0.0, 1.0]`` computed
    via exponential decay::

        score = exp(-λ · age_seconds)

    where ``λ = ln(2) / halflife``.  Documents older than
    ``stale_threshold_seconds`` are clamped to a near-zero score.

    The half-life and stale threshold are read from
    :class:`~app.config.ConsistencyConfig` at construction time but can
    be overridden via constructor arguments for testing.
    """

    def __init__(
        self,
        *,
        halflife_seconds: float | None = None,
        stale_threshold_seconds: float | None = None,
    ) -> None:
        cfg = get_config().consistency
        self._halflife: float = (
            halflife_seconds
            if halflife_seconds is not None
            else cfg.freshness_decay_halflife_seconds
        )
        self._stale_threshold: float = (
            stale_threshold_seconds
            if stale_threshold_seconds is not None
            else cfg.stale_threshold_seconds
        )

        # Pre-compute the decay constant  λ = ln(2) / t½
        if self._halflife > 0:
            self._lambda: float = math.log(2) / self._halflife
        else:
            # Edge case: halflife ≤ 0 ⟹ everything is instantly stale
            self._lambda = float("inf")

        logger.info(
            "FreshnessScorer initialised — halflife=%.1fs, stale_threshold=%.1fs, λ=%.6f",
            self._halflife,
            self._stale_threshold,
            self._lambda if math.isfinite(self._lambda) else -1.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, document: Document, query_time: float) -> float:
        """Compute the freshness score for *document* at *query_time*.

        Parameters:
            document:   The document to score.
            query_time: The epoch timestamp of the incoming query
                        (typically ``time.time()``).

        Returns:
            A float in ``[0.0, 1.0]`` where **1.0** means perfectly
            fresh and **0.0** means completely stale.
        """
        age_seconds = query_time - document.timestamp

        # Future-dated documents are treated as perfectly fresh
        if age_seconds <= 0:
            return 1.0

        # Beyond stale threshold → hard floor
        if age_seconds >= self._stale_threshold:
            return 0.0

        # Exponential decay
        freshness = math.exp(-self._lambda * age_seconds)

        # Clamp to valid range (should already be, but guard precision)
        return max(0.0, min(1.0, freshness))

    def apply(
        self,
        scored_documents: list[ScoredDocument],
        query_time: float,
    ) -> list[ScoredDocument]:
        """Compute and apply freshness scores to a list of scored documents.

        Each :class:`ScoredDocument` has its ``freshness_score`` field
        updated **in-place** and returned.

        Parameters:
            scored_documents: Documents to score.
            query_time:       Epoch timestamp of the query.

        Returns:
            The same list with ``freshness_score`` populated.
        """
        for sd in scored_documents:
            sd.freshness_score = self.score(sd.document, query_time)
        return scored_documents

    def apply_and_rerank(
        self,
        scored_documents: list[ScoredDocument],
        query_time: float,
        freshness_weight: float = 0.3,
    ) -> list[ScoredDocument]:
        """Score, blend freshness into ``final_score``, and re-sort.

        The blended score is::

            final = (1 - w) * original_final + w * freshness

        Parameters:
            scored_documents: Documents to process.
            query_time:       Epoch timestamp of the query.
            freshness_weight: Weight given to the freshness component
                              in ``[0.0, 1.0]``.

        Returns:
            The list sorted **descending** by the updated ``final_score``.
        """
        w = max(0.0, min(1.0, freshness_weight))

        for sd in scored_documents:
            sd.freshness_score = self.score(sd.document, query_time)
            sd.final_score = (1.0 - w) * sd.final_score + w * sd.freshness_score

        scored_documents.sort(key=lambda sd: sd.final_score, reverse=True)
        return scored_documents

    def is_stale(self, document: Document, query_time: float) -> bool:
        """Return ``True`` if *document* is older than the stale threshold."""
        return (query_time - document.timestamp) >= self._stale_threshold

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def explain(self, document: Document, query_time: float) -> dict[str, float]:
        """Return a diagnostic dict explaining the freshness calculation.

        Useful for debugging and observability dashboards.
        """
        age = query_time - document.timestamp
        fresh = self.score(document, query_time)
        return {
            "age_seconds": round(age, 3),
            "halflife_seconds": self._halflife,
            "stale_threshold_seconds": self._stale_threshold,
            "lambda": round(self._lambda, 6) if math.isfinite(self._lambda) else -1.0,
            "freshness_score": round(fresh, 6),
            "is_stale": float(self.is_stale(document, query_time)),
        }
