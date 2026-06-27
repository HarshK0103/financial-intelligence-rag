"""Tests for the data layer: HotStore, ColdStore, and DataRouter."""

import time

import pytest

from app.data.cold_store import ColdStore
from app.data.data_router import DataRouter
from app.data.hot_store import HotStore
from app.models import DataTemperature, Document, QueryRequest


def _hot_doc(doc_id: str = "hot_1", ticker: str = "AAPL") -> Document:
    return Document(
        doc_id=doc_id,
        content="AAPL is trading at $195.50",
        source="market",
        ticker=ticker,
        temperature=DataTemperature.HOT,
        timestamp=time.time(),
    )


def _cold_doc(doc_id: str = "cold_1", ticker: str = "AAPL") -> Document:
    return Document(
        doc_id=doc_id,
        content="AAPL 10-K annual filing for fiscal year 2024",
        source="sec_filing",
        ticker=ticker,
        temperature=DataTemperature.COLD,
        timestamp=time.time() - 86400,
    )


# ══════════════════════════════════════════════════════════════════
# HotStore
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_hot_store_add_and_get() -> None:
    store = HotStore(ttl_seconds=60)
    doc = _hot_doc()
    await store.add_document(doc)

    result = await store.get_document("hot_1")
    assert result is not None
    assert result.doc_id == "hot_1"


@pytest.mark.asyncio
async def test_hot_store_get_by_ticker() -> None:
    store = HotStore(ttl_seconds=60)
    await store.add_document(_hot_doc("h1", "AAPL"))
    await store.add_document(_hot_doc("h2", "AAPL"))
    await store.add_document(_hot_doc("h3", "NVDA"))

    aapl_docs = await store.get_by_ticker("AAPL")
    assert len(aapl_docs) == 2

    nvda_docs = await store.get_by_ticker("NVDA")
    assert len(nvda_docs) == 1


@pytest.mark.asyncio
async def test_hot_store_count() -> None:
    store = HotStore(ttl_seconds=60)
    await store.add_document(_hot_doc("a"))
    await store.add_document(_hot_doc("b"))
    assert await store.count() == 2


@pytest.mark.asyncio
async def test_hot_store_missing_doc_returns_none() -> None:
    store = HotStore(ttl_seconds=60)
    assert await store.get_document("nonexistent") is None


@pytest.mark.asyncio
async def test_hot_store_clear() -> None:
    store = HotStore(ttl_seconds=60)
    await store.add_document(_hot_doc())
    await store.clear()
    assert await store.count() == 0


# ══════════════════════════════════════════════════════════════════
# ColdStore
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_cold_store_add_and_get() -> None:
    store = ColdStore()
    doc = _cold_doc()
    await store.add_document(doc)

    result = await store.get_document("cold_1")
    assert result is not None
    assert result.doc_id == "cold_1"


@pytest.mark.asyncio
async def test_cold_store_get_by_ticker() -> None:
    store = ColdStore()
    await store.add_document(_cold_doc("c1", "MSFT"))
    await store.add_document(_cold_doc("c2", "MSFT"))

    docs = await store.get_by_ticker("MSFT")
    assert len(docs) == 2


@pytest.mark.asyncio
async def test_cold_store_count() -> None:
    store = ColdStore()
    await store.add_document(_cold_doc("a"))
    await store.add_document(_cold_doc("b"))
    await store.add_document(_cold_doc("c"))
    assert await store.count() == 3


@pytest.mark.asyncio
async def test_cold_store_missing_doc_returns_none() -> None:
    store = ColdStore()
    assert await store.get_document("nope") is None


# ══════════════════════════════════════════════════════════════════
# DataRouter
# ══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_router_stores_hot_doc_in_hot_store() -> None:
    hot, cold = HotStore(ttl_seconds=60), ColdStore()
    router = DataRouter(hot, cold)

    await router.add_document(_hot_doc("route_hot"))
    assert await hot.get_document("route_hot") is not None
    assert await cold.get_document("route_hot") is None


@pytest.mark.asyncio
async def test_router_stores_cold_doc_in_cold_store() -> None:
    hot, cold = HotStore(ttl_seconds=60), ColdStore()
    router = DataRouter(hot, cold)

    await router.add_document(_cold_doc("route_cold"))
    assert await cold.get_document("route_cold") is not None
    assert await hot.get_document("route_cold") is None


@pytest.mark.asyncio
async def test_router_classifies_price_query_as_hot() -> None:
    hot, cold = HotStore(ttl_seconds=60), ColdStore()
    router = DataRouter(hot, cold)
    request = QueryRequest(query="What is the current price of AAPL?")
    temp = await router.route_query(request)
    assert temp == DataTemperature.HOT


@pytest.mark.asyncio
async def test_router_classifies_filing_query_as_cold() -> None:
    hot, cold = HotStore(ttl_seconds=60), ColdStore()
    router = DataRouter(hot, cold)
    request = QueryRequest(query="What did the 10-K annual report say about risk?")
    temp = await router.route_query(request)
    assert temp == DataTemperature.COLD


@pytest.mark.asyncio
async def test_router_require_fresh_forces_hot() -> None:
    hot, cold = HotStore(ttl_seconds=60), ColdStore()
    router = DataRouter(hot, cold)
    request = QueryRequest(query="historical analysis of AAPL", require_fresh=True)
    temp = await router.route_query(request)
    assert temp == DataTemperature.HOT
