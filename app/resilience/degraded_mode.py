"""
Financial RAG System — Degraded Mode

Generates structured fallback responses when the main pipeline cannot
satisfy its SLA.  This sits on the *hot path* — any call to this
module must itself be effectively instantaneous (< 1 ms).

Strategy
────────
1. If **cached data** is available (from a prior L1/L2/L3 hit or a
   stale but still-present entry), it is returned directly with a
   staleness disclaimer.
2. If no cached data is available, a **canned degraded message** is
   returned that is still machine-parseable by downstream consumers.
3. Output is always *compressed* — no optional whitespace, minimal
   boilerplate — following Approach 2's hot-path optimisation.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_config
from app.models import (
    CacheLayer,
    CircuitState,
    QueryResponse,
    QueryType,
    ResponseMetrics,
    ScoredDocument,
)

logger = logging.getLogger(__name__)

# ── Pre-built canned messages (kept short for hot-path) ──────────

_DEGRADED_MESSAGES: dict[str, str] = {
    "timeout": ("The system could not complete your request within the " "latency budget. Please try again shortly."),
    "circuit_open": (
        "A downstream dependency is temporarily unavailable. " "The system is serving cached or partial results."
    ),
    "overload": ("The system is experiencing high load. " "Your request has been served in a reduced capacity."),
    "unknown": ("The system is operating in degraded mode. " "Results may be incomplete or stale."),
}


class DegradedMode:
    """Generate safe fallback responses when the pipeline is degraded.

    All methods are *synchronous* on purpose — they must never await
    anything so they cannot contribute to a timeout cascade.

    Parameters
    ----------
    max_tokens : int | None
        Maximum output length (approx tokens ≈ chars / 4).  Defaults
        to ``config.resilience.degraded_response_max_tokens``.
    """

    _CHARS_PER_TOKEN: int = 4

    def __init__(self, max_tokens: int | None = None) -> None:
        cfg = get_config()
        self._max_tokens = max_tokens or cfg.resilience.degraded_response_max_tokens
        self._max_chars = self._max_tokens * self._CHARS_PER_TOKEN
        self._cache_only = cfg.resilience.degraded_cache_only

        # Counters
        self._total_degraded: int = 0
        self._cached_served: int = 0
        self._canned_served: int = 0

        logger.info(
            "DegradedMode initialised  max_tokens=%d  cache_only=%s",
            self._max_tokens,
            self._cache_only,
        )

    # ── public API ────────────────────────────────────────────────

    def generate_degraded_response(
        self,
        query: str,
        cached_data: dict[str, Any] | None = None,
        reason: str = "unknown",
    ) -> QueryResponse:
        """Build a degraded :class:`QueryResponse`.

        Parameters
        ----------
        query : str
            The original user query (included for context in the
            response body).
        cached_data : dict | None
            Previously cached response data, if any.  Expected keys:
            ``"answer"`` (str), ``"sources"`` (list of dict),
            ``"query_type"`` (str), ``"cache_layer"`` (str).
        reason : str
            Why the pipeline is degraded.  Must be one of
            ``"timeout"``, ``"circuit_open"``, ``"overload"``, or
            ``"unknown"``.

        Returns
        -------
        QueryResponse
            A response with ``is_degraded=True``.
        """
        self._total_degraded += 1

        if cached_data and self._has_usable_answer(cached_data):
            return self._from_cache(query, cached_data, reason)

        return self._canned_response(query, reason)

    # ── cached path ───────────────────────────────────────────────

    def _from_cache(
        self,
        query: str,
        cached_data: dict[str, Any],
        reason: str,
    ) -> QueryResponse:
        """Construct a response from stale cached data."""
        self._cached_served += 1

        answer = self._compress(str(cached_data.get("answer", "")))
        answer = self._enforce_budget(answer)

        # Prepend a staleness notice.
        notice = f"[Degraded — {self._reason_label(reason)}] "
        answer = notice + answer

        # Reconstruct sources if present.
        sources = self._rebuild_sources(cached_data.get("sources"))

        # Determine cache layer.
        raw_layer = cached_data.get("cache_layer", CacheLayer.MISS.value)
        try:
            cache_layer = CacheLayer(raw_layer)
        except ValueError:
            cache_layer = CacheLayer.MISS

        # Determine query type.
        raw_qt = cached_data.get("query_type", QueryType.GENERAL.value)
        try:
            query_type = QueryType(raw_qt)
        except ValueError:
            query_type = QueryType.GENERAL

        logger.info(
            "Degraded response (cached)  reason=%s  cache_layer=%s  " "answer_len=%d",
            reason,
            cache_layer.value,
            len(answer),
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            query_type=query_type,
            cache_layer=cache_layer,
            is_degraded=True,
            metrics=self._build_metrics(reason, cache_layer),
        )

    # ── canned path ───────────────────────────────────────────────

    def _canned_response(
        self,
        query: str,
        reason: str,
    ) -> QueryResponse:
        """Return a minimal canned message when no cache is available."""
        self._canned_served += 1

        message = _DEGRADED_MESSAGES.get(reason, _DEGRADED_MESSAGES["unknown"])
        answer = self._enforce_budget(message)

        logger.info(
            "Degraded response (canned)  reason=%s  answer_len=%d",
            reason,
            len(answer),
        )

        return QueryResponse(
            answer=answer,
            sources=[],
            query_type=QueryType.GENERAL,
            cache_layer=CacheLayer.MISS,
            is_degraded=True,
            metrics=self._build_metrics(reason, CacheLayer.MISS),
        )

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _has_usable_answer(cached_data: dict[str, Any]) -> bool:
        """Return *True* if *cached_data* contains a non-empty answer."""
        answer = cached_data.get("answer")
        return isinstance(answer, str) and len(answer.strip()) > 0

    @staticmethod
    def _reason_label(reason: str) -> str:
        """Human-readable label for the degradation reason."""
        labels = {
            "timeout": "request timed out",
            "circuit_open": "dependency unavailable",
            "overload": "high load",
            "unknown": "service degraded",
        }
        return labels.get(reason, reason)

    @staticmethod
    def _rebuild_sources(
        raw_sources: list[dict[str, Any]] | Any | None,
    ) -> list[ScoredDocument]:
        """Best-effort reconstruction of :class:`ScoredDocument` list."""
        if not isinstance(raw_sources, list):
            return []

        rebuilt: list[ScoredDocument] = []
        for src in raw_sources:
            if not isinstance(src, dict):
                continue
            try:
                rebuilt.append(ScoredDocument.model_validate(src))
            except Exception:
                # Malformed cache entry — skip silently.
                continue
        return rebuilt

    @staticmethod
    def _compress(text: str) -> str:
        """Strip excessive whitespace for hot-path compactness."""
        lines = text.splitlines()
        compressed = " ".join(line.strip() for line in lines if line.strip())
        return compressed

    def _enforce_budget(self, text: str) -> str:
        """Truncate to the degraded token budget."""
        if len(text) <= self._max_chars:
            return text
        truncated = text[: self._max_chars]
        last_space = truncated.rfind(" ")
        if last_space > self._max_chars * 0.6:
            truncated = truncated[:last_space]
        return truncated.rstrip() + " …"

    @staticmethod
    def _build_metrics(
        reason: str,
        cache_layer: CacheLayer,
    ) -> ResponseMetrics:
        """Attach minimal metrics to a degraded response."""
        circuit = CircuitState.OPEN if reason == "circuit_open" else CircuitState.CLOSED
        return ResponseMetrics(
            total_latency_ms=0.0,
            cache_hit=cache_layer != CacheLayer.MISS,
            cache_layer=cache_layer,
            circuit_state=circuit,
        )

    # ── diagnostics ───────────────────────────────────────────────

    @property
    def total_degraded(self) -> int:
        """Total degraded responses generated."""
        return self._total_degraded

    @property
    def cached_served(self) -> int:
        """Degraded responses served from cache."""
        return self._cached_served

    @property
    def canned_served(self) -> int:
        """Degraded responses served with canned messages."""
        return self._canned_served
