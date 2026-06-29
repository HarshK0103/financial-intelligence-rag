"""
Financial RAG System — Data Router

Routes documents and queries between the :class:`HotStore` (real-time)
and :class:`ColdStore` (historical) based on data temperature.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from app.config import get_config
from app.models import DataTemperature, Document, QueryRequest
from app.data.cold_store import ColdStore
from app.data.hot_store import HotStore

logger = logging.getLogger(__name__)

# ── Query classification heuristics ───────────────────────────────

# Keywords / patterns strongly signalling real-time ("hot") intent.
_HOT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(current|live|now|latest|real[- ]?time|today|right now)\b", re.IGNORECASE),
    re.compile(r"\b(price|bid|ask|spread|volume|order\s*book)\b", re.IGNORECASE),
    re.compile(r"\b(breaking|just\s+announced|flash)\b", re.IGNORECASE),
    re.compile(r"\b(pre[- ]?market|after[- ]?hours|intraday)\b", re.IGNORECASE),
]

# Keywords signalling historical ("cold") intent.
_COLD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(10-[KQ]|annual\s+report|SEC\s+filing|proxy)\b", re.IGNORECASE),
    re.compile(r"\b(historical|past|last\s+year|quarter|fiscal\s+year)\b", re.IGNORECASE),
    re.compile(r"\b(earnings\s+call|transcript|analyst\s+report)\b", re.IGNORECASE),
    re.compile(r"\bQ[1-4]\s*\d{4}\b", re.IGNORECASE),
    re.compile(r"\b(20\d{2}|19\d{2})\b"),  # explicit year references
]


class DataRouter:
    """Routes documents and queries to the appropriate data store.

    Uses lightweight keyword heuristics to classify a
    :class:`QueryRequest` as **hot** or **cold**, then dispatches
    reads / writes to the corresponding store.
    """

    def __init__(self, hot_store: HotStore, cold_store: ColdStore) -> None:
        self._hot = hot_store
        self._cold = cold_store

        cfg = get_config()
        # Hot tickers from cache config — queries mentioning these
        # tickers get a slight bias toward the hot store.
        self._hot_tickers: set[str] = set(cfg.cache.l3_tickers)

    @property
    def hot_store(self) -> HotStore:
        return self._hot

    @property
    def cold_store(self) -> ColdStore:
        return self._cold

    # ── Query routing ──────────────────────────────────────────────

    async def route_query(self, request: QueryRequest) -> DataTemperature:
        """Classify a query as **hot** (real-time) or **cold** (historical).

        The decision is based on:
        1. ``require_fresh`` flag — forces HOT.
        2. Keyword / pattern matching against the query text.
        3. Ticker membership in the hot-ticker list (weak signal).
        """
        # 1. Explicit freshness requirement.
        if request.require_fresh:
            logger.debug("DataRouter: require_fresh=True → HOT")
            return DataTemperature.HOT

        query = request.query
        hot_score = 0.0
        cold_score = 0.0

        # 2. Pattern matching.
        for pattern in _HOT_PATTERNS:
            if pattern.search(query):
                hot_score += 1.0

        for pattern in _COLD_PATTERNS:
            if pattern.search(query):
                cold_score += 1.0

        # 3. Hot-ticker bias.
        for ticker in request.tickers:
            if ticker.upper() in self._hot_tickers:
                hot_score += 0.3

        temperature = DataTemperature.HOT if hot_score >= cold_score else DataTemperature.COLD

        logger.debug(
            "DataRouter: query=%r hot=%.1f cold=%.1f → %s",
            query[:60],
            hot_score,
            cold_score,
            temperature.value,
        )
        return temperature

    # ── Document retrieval ─────────────────────────────────────────

    async def get_documents(
        self,
        temperature: DataTemperature,
        ticker: str | None = None,
    ) -> list[Document]:
        """Retrieve documents from the store matching *temperature*.

        Parameters
        ----------
        temperature:
            Which store to query.
        ticker:
            If provided, filter by ticker symbol.

        Returns
        -------
        list[Document]
            Documents from the selected store.
        """
        if temperature == DataTemperature.HOT:
            if ticker:
                return await self._hot.get_by_ticker(ticker)
            return await self._hot.get_all()

        # COLD
        if ticker:
            return await self._cold.get_by_ticker(ticker)
        # Without a ticker filter the cold store could be huge — return
        # an empty list and let callers use search_by_content instead.
        return []

    async def get_documents_both(
        self,
        ticker: str | None = None,
    ) -> list[Document]:
        """Retrieve from **both** stores (useful for ambiguous queries).

        Results are returned hot-first, then cold.
        """
        hot_docs = await self.get_documents(DataTemperature.HOT, ticker)
        cold_docs = await self.get_documents(DataTemperature.COLD, ticker)
        return hot_docs + cold_docs

    # ── Document ingestion ─────────────────────────────────────────

    async def add_document(self, doc: Document) -> None:
        """Route a single document to the correct store based on its temperature."""
        if doc.temperature == DataTemperature.HOT:
            await self._hot.add_document(doc)
        else:
            await self._cold.add_document(doc)
        logger.debug(
            "DataRouter: routed doc_id=%s → %s store",
            doc.doc_id,
            doc.temperature.value,
        )

    async def add_documents(self, docs: Sequence[Document]) -> dict[str, int]:
        """Batch-route documents.  Returns ``{"hot": n, "cold": m}``."""
        hot_batch: list[Document] = []
        cold_batch: list[Document] = []

        for doc in docs:
            if doc.temperature == DataTemperature.HOT:
                hot_batch.append(doc)
            else:
                cold_batch.append(doc)

        hot_count = 0
        cold_count = 0

        if hot_batch:
            hot_count = await self._hot.add_documents(hot_batch)
        if cold_batch:
            cold_count = await self._cold.add_documents(cold_batch)

        logger.info(
            "DataRouter: batch routed hot=%d cold=%d",
            hot_count,
            cold_count,
        )
        return {"hot": hot_count, "cold": cold_count}

    # ── Introspection ──────────────────────────────────────────────

    async def counts(self) -> dict[str, int]:
        """Return document counts for each store."""
        return {
            "hot": await self._hot.count(),
            "cold": await self._cold.count(),
        }
