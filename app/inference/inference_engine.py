"""
Financial RAG System — Inference Engine

Generates human-readable answers from retrieved documents.  In this
prototype actual LLM calls are replaced by **template-based** generation
so the module runs without a model server and meets the 80 ms
inference budget.

Optimisations
─────────────
* **Query classification** — a fast keyword classifier routes queries
  to specialised templates, eliminating prompt-engineering overhead.
* **Prompt compression** — excess whitespace is stripped and context
  is truncated before hitting the token budget.
* **Token budget** — output is hard-capped at ``max_output_tokens``
  (approximated as characters ÷ 4).
"""

from __future__ import annotations

import logging
import re
import textwrap
from typing import Final

from app.config import get_config
from app.models import QueryType, ScoredDocument

logger = logging.getLogger(__name__)

# ── Keyword maps for classification ──────────────────────────────

_PRICE_KEYWORDS: Final[frozenset[str]] = frozenset({
    "price", "stock price", "share price", "trading", "market cap",
    "valuation", "pe ratio", "p/e", "52-week", "high", "low",
    "close", "open", "volume", "bid", "ask", "quote",
})

_EARNINGS_KEYWORDS: Final[frozenset[str]] = frozenset({
    "earnings", "revenue", "profit", "eps", "ebitda", "margin",
    "quarterly", "annual", "q1", "q2", "q3", "q4", "fiscal",
    "income", "operating", "net income", "gross", "guidance",
    "forecast", "beat", "miss",
})

_NEWS_KEYWORDS: Final[frozenset[str]] = frozenset({
    "news", "headline", "breaking", "report", "announce",
    "announcement", "release", "press", "update", "latest",
    "today", "yesterday", "recent", "filed",
})

_COMPARISON_KEYWORDS: Final[frozenset[str]] = frozenset({
    "compare", "comparison", "versus", "vs", "vs.", "differ",
    "difference", "better", "worse", "outperform", "underperform",
    "relative", "against",
})

_ANALYSIS_KEYWORDS: Final[frozenset[str]] = frozenset({
    "analysis", "analyze", "analyst", "forecast", "outlook",
    "risk", "opportunity", "strategy", "recommendation", "target",
    "rating", "upgrade", "downgrade", "bull", "bear", "sentiment",
})


# ── Templates ─────────────────────────────────────────────────────

_TEMPLATES: dict[QueryType, str] = {
    QueryType.PRICE: (
        "Based on the available data:\n\n"
        "{context}\n\n"
        "Note: Financial data may be delayed. "
        "Verify with your broker for real-time pricing."
    ),
    QueryType.EARNINGS: (
        "Earnings Summary:\n\n"
        "{context}\n\n"
        "Key takeaway: {takeaway}"
    ),
    QueryType.NEWS: (
        "Recent developments:\n\n"
        "{context}\n\n"
        "This summary is based on the most recent sources available."
    ),
    QueryType.COMPARISON: (
        "Comparative Analysis:\n\n"
        "{context}\n\n"
        "Please note that comparisons are based on the data available "
        "at the time of retrieval."
    ),
    QueryType.ANALYSIS: (
        "Analysis Overview:\n\n"
        "{context}\n\n"
        "This analysis reflects data as of the latest available date. "
        "Consult a licensed financial advisor for personalised advice."
    ),
    QueryType.GENERAL: (
        "Here is what the available data shows:\n\n"
        "{context}"
    ),
}


class InferenceEngine:
    """Template-based inference engine for financial queries.

    Parameters
    ----------
    max_output_tokens : int | None
        Hard cap on output length (in approximate tokens ≈ chars / 4).
        Falls back to ``config.inference.max_output_tokens``.
    """

    # Approximate chars-per-token ratio.  Conservative so we never
    # exceed a downstream LLM context window.
    _CHARS_PER_TOKEN: Final[int] = 4

    def __init__(self, max_output_tokens: int | None = None) -> None:
        cfg = get_config()
        self._max_tokens: int = max_output_tokens or cfg.inference.max_output_tokens
        self._max_chars: int = self._max_tokens * self._CHARS_PER_TOKEN
        self._compression: bool = cfg.inference.prompt_compression_enabled
        self._template_mode: bool = cfg.inference.template_mode_enabled

        logger.info(
            "InferenceEngine initialised  max_tokens=%d  compression=%s  "
            "template_mode=%s",
            self._max_tokens,
            self._compression,
            self._template_mode,
        )

    # ── public API ────────────────────────────────────────────────

    async def generate(
        self,
        query: str,
        context_docs: list[ScoredDocument],
        query_type: QueryType | None = None,
    ) -> str:
        """Generate an answer using context documents.

        Parameters
        ----------
        query : str
            The user's raw query text.
        context_docs : list[ScoredDocument]
            Retrieved documents, sorted by relevance (highest first).
        query_type : QueryType | None
            Pre-classified type.  If *None*, the engine classifies
            the query automatically.

        Returns
        -------
        str
            The generated answer, length-capped to the token budget.
        """
        if query_type is None:
            query_type = self.classify_query(query)

        context_text = self._build_context(context_docs)

        if self._compression:
            context_text = self._compress(context_text)

        if self._template_mode:
            answer = self._render_template(query_type, context_text, query)
        else:
            # Fallback: plain concatenation when templates are disabled.
            answer = f"Query: {query}\n\n{context_text}"

        answer = self._enforce_budget(answer)

        logger.debug(
            "Generated answer  query_type=%s  chars=%d  approx_tokens=%d",
            query_type.value,
            len(answer),
            len(answer) // self._CHARS_PER_TOKEN,
        )
        return answer

    def classify_query(self, query: str) -> QueryType:
        """Classify a query string into a :class:`QueryType`.

        Uses simple keyword overlap scoring — fast and predictable.
        """
        lower = query.lower()
        scores: dict[QueryType, int] = {
            QueryType.PRICE: self._keyword_score(lower, _PRICE_KEYWORDS),
            QueryType.EARNINGS: self._keyword_score(lower, _EARNINGS_KEYWORDS),
            QueryType.NEWS: self._keyword_score(lower, _NEWS_KEYWORDS),
            QueryType.COMPARISON: self._keyword_score(lower, _COMPARISON_KEYWORDS),
            QueryType.ANALYSIS: self._keyword_score(lower, _ANALYSIS_KEYWORDS),
        }

        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_type]

        if best_score == 0:
            return QueryType.GENERAL

        logger.debug(
            "Classified query as %s  (score=%d)  query=%r",
            best_type.value,
            best_score,
            query[:80],
        )
        return best_type

    # ── context building ──────────────────────────────────────────

    def _build_context(self, docs: list[ScoredDocument]) -> str:
        """Concatenate scored-document content into a single context string."""
        if not docs:
            return "No relevant documents were found for this query."

        parts: list[str] = []
        for idx, sd in enumerate(docs, 1):
            source_tag = sd.document.source or "unknown"
            ticker_tag = sd.document.ticker or "N/A"
            header = f"[{idx}] ({source_tag} | {ticker_tag})"
            parts.append(f"{header}\n{sd.document.content}")

        return "\n\n".join(parts)

    # ── template rendering ────────────────────────────────────────

    def _render_template(
        self,
        query_type: QueryType,
        context: str,
        query: str,
    ) -> str:
        """Fill the template for *query_type* with context text."""
        template = _TEMPLATES.get(query_type, _TEMPLATES[QueryType.GENERAL])

        # For earnings we generate a simple one-line takeaway.
        takeaway = self._extract_takeaway(context, query)

        return template.format(context=context, takeaway=takeaway)

    @staticmethod
    def _extract_takeaway(context: str, query: str) -> str:
        """Produce a single-sentence takeaway for earnings-style queries.

        In a real system this would be LLM-generated; here we use
        a deterministic heuristic.
        """
        # Use the first sentence of the first document as the takeaway.
        sentences = re.split(r"(?<=[.!?])\s+", context)
        for sent in sentences:
            stripped = sent.strip()
            if len(stripped) > 20:
                return stripped
        return "Refer to the sources above for detailed figures."

    # ── prompt compression ────────────────────────────────────────

    @staticmethod
    def _compress(text: str) -> str:
        """Strip redundant whitespace and normalise the text."""
        # Collapse multiple blank lines into one.
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse runs of spaces/tabs (not newlines).
        text = re.sub(r"[^\S\n]+", " ", text)
        return text.strip()

    # ── budget enforcement ────────────────────────────────────────

    def _enforce_budget(self, text: str) -> str:
        """Truncate *text* to ``self._max_chars`` on a word boundary."""
        if len(text) <= self._max_chars:
            return text

        truncated = text[: self._max_chars]
        # Try to cut at the last space so we don't split a word.
        last_space = truncated.rfind(" ")
        if last_space > self._max_chars * 0.8:
            truncated = truncated[:last_space]

        truncated = truncated.rstrip() + " …"
        logger.debug(
            "Enforced token budget: %d chars → %d chars",
            len(text),
            len(truncated),
        )
        return truncated

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _keyword_score(text: str, keywords: frozenset[str]) -> int:
        """Count how many *keywords* appear in *text*."""
        return sum(1 for kw in keywords if kw in text)
