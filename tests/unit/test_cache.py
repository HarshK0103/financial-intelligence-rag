import pytest

from app.cache.cache_manager import CacheManager
from app.cache.exact_cache import ExactCache
from app.models import CacheLayer, QueryResponse


@pytest.mark.asyncio
async def test_exact_cache_hit() -> None:
    exact_cache = ExactCache()
    exact_cache._redis_enabled = False
    manager = CacheManager(exact_cache=exact_cache)
    response = QueryResponse(answer="cached")

    await manager.set("What is AAPL?", None, response, tickers=["AAPL"])
    layer, cached = await manager.get("What is AAPL?")

    assert layer == CacheLayer.L1_EXACT
    assert cached is not None
    assert cached.answer == "cached"


@pytest.mark.asyncio
async def test_cache_invalidation_by_ticker() -> None:
    exact_cache = ExactCache()
    exact_cache._redis_enabled = False
    manager = CacheManager(exact_cache=exact_cache)

    await manager.set(
        "What is AAPL?",
        None,
        QueryResponse(answer="cached"),
        tickers=["AAPL"],
    )
    counts = await manager.invalidate_by_tickers(["AAPL"])
    layer, cached = await manager.get("What is AAPL?")

    assert counts["l1"] == 1
    assert layer == CacheLayer.MISS
    assert cached is None
