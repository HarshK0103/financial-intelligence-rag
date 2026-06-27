"""Tests for the lightweight heuristic reranker."""

import time

import pytest

from app.models import Document, ScoredDocument
from app.retrieval.reranker import Reranker


@pytest.fixture
def reranker() -> Reranker:
    return Reranker()


def _scored_doc(
    content: str,
    ticker: str = "AAPL",
    age_seconds: float = 0.0,
) -> ScoredDocument:
    return ScoredDocument(
        document=Document(
            doc_id=f"doc_{hash(content) % 10000}",
            content=content,
            source="test",
            ticker=ticker,
            timestamp=time.time() - age_seconds,
        ),
        bm25_score=0.5,
        vector_score=0.5,
    )


# ── Basic behavior ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_top_k(reranker: Reranker) -> None:
    docs = [_scored_doc(f"Document {i} about AAPL revenue") for i in range(10)]
    result = await reranker.rerank("AAPL revenue", docs, top_k=3)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_empty_input_returns_empty(reranker: Reranker) -> None:
    result = await reranker.rerank("test query", [], top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_fewer_docs_than_top_k(reranker: Reranker) -> None:
    docs = [_scored_doc("Only two docs"), _scored_doc("Second doc")]
    result = await reranker.rerank("test", docs, top_k=10)
    assert len(result) == 2


# ── Scoring quality ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_overlap_boosts_score(reranker: Reranker) -> None:
    relevant = _scored_doc("AAPL revenue grew 15% in Q3 2024 earnings report")
    irrelevant = _scored_doc("The weather forecast for tomorrow is sunny")
    result = await reranker.rerank(
        "AAPL revenue earnings", [irrelevant, relevant], top_k=2
    )
    # The relevant doc should be ranked first
    assert "revenue" in result[0].document.content.lower()


@pytest.mark.asyncio
async def test_results_sorted_by_final_score(reranker: Reranker) -> None:
    docs = [_scored_doc(f"Content {i}") for i in range(5)]
    result = await reranker.rerank("test query", docs, top_k=5)
    final_scores = [d.final_score for d in result]
    assert final_scores == sorted(final_scores, reverse=True)


@pytest.mark.asyncio
async def test_rerank_score_is_populated(reranker: Reranker) -> None:
    docs = [_scored_doc("Some content about revenue")]
    result = await reranker.rerank("revenue", docs, top_k=1)
    assert result[0].rerank_score >= 0.0


@pytest.mark.asyncio
async def test_fresher_docs_score_higher(reranker: Reranker) -> None:
    fresh = _scored_doc("AAPL data", age_seconds=1.0)
    stale = _scored_doc("AAPL data", age_seconds=600.0)
    result = await reranker.rerank("AAPL", [stale, fresh], top_k=2)
    # The fresher doc should rank higher (all else equal)
    assert result[0].document.timestamp > result[1].document.timestamp
