"""Tests for the semantic similarity cache (L2)."""

import numpy as np
import pytest

from app.cache.semantic_cache import SemanticCache
from app.models import QueryResponse


@pytest.fixture
def cache() -> SemanticCache:
    return SemanticCache()


def _embedding(seed: int, dim: int = 384) -> list[float]:
    """Deterministic embedding vector from a seed."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _similar_embedding(base: list[float], noise: float = 0.02) -> list[float]:
    """Create an embedding very similar to *base* (cosine > 0.99)."""
    arr = np.array(base, dtype=np.float32)
    arr += np.random.RandomState(42).randn(len(base)).astype(np.float32) * noise
    arr /= np.linalg.norm(arr)
    return arr.tolist()


def _different_embedding(dim: int = 384) -> list[float]:
    """Create a random embedding likely dissimilar to any seeded one."""
    rng = np.random.RandomState(9999)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


# ── Hit / miss ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exact_embedding_returns_hit(cache: SemanticCache) -> None:
    emb = _embedding(1)
    response = QueryResponse(answer="cached answer")
    await cache.set("test query", emb, response, tickers=["AAPL"])

    result = await cache.get("test query", emb)
    assert result is not None
    assert result.answer == "cached answer"
    assert cache.hits == 1


@pytest.mark.asyncio
async def test_similar_embedding_returns_hit(cache: SemanticCache) -> None:
    emb = _embedding(2)
    similar = _similar_embedding(emb, noise=0.01)
    response = QueryResponse(answer="similar hit")
    await cache.set("original", emb, response)

    result = await cache.get("similar query", similar)
    assert result is not None
    assert result.answer == "similar hit"


@pytest.mark.asyncio
async def test_dissimilar_embedding_returns_miss(cache: SemanticCache) -> None:
    emb = _embedding(3)
    different = _different_embedding()
    await cache.set("query A", emb, QueryResponse(answer="A"))

    result = await cache.get("query B", different)
    assert result is None
    assert cache.misses >= 1


@pytest.mark.asyncio
async def test_empty_cache_returns_miss(cache: SemanticCache) -> None:
    result = await cache.get("anything", _embedding(10))
    assert result is None
    assert cache.misses == 1


# ── Invalidation ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalidate_by_tickers_removes_entries(
    cache: SemanticCache,
) -> None:
    emb = _embedding(4)
    await cache.set("q1", emb, QueryResponse(answer="A"), tickers=["AAPL"])
    await cache.set("q2", _embedding(5), QueryResponse(answer="B"), tickers=["NVDA"])

    removed = await cache.invalidate_by_tickers(["AAPL"])
    assert removed == 1

    result = await cache.get("q1", emb)
    assert result is None

    result_nvda = await cache.get("q2", _embedding(5))
    assert result_nvda is not None


# ── Clear ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clear_removes_all(cache: SemanticCache) -> None:
    await cache.set("q1", _embedding(6), QueryResponse(answer="A"))
    await cache.set("q2", _embedding(7), QueryResponse(answer="B"))

    await cache.clear()

    assert await cache.get("q1", _embedding(6)) is None
    assert await cache.get("q2", _embedding(7)) is None


# ── Counters ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hit_miss_counters(cache: SemanticCache) -> None:
    emb = _embedding(8)
    await cache.set("q", emb, QueryResponse(answer="x"))

    await cache.get("q", emb)             # hit
    await cache.get("q", _different_embedding())  # miss

    assert cache.hits >= 1
    assert cache.misses >= 1
