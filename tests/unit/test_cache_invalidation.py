"""Tests for multi-layer cache invalidation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.cache.cache_manager import CacheManager
from app.cache.exact_cache import ExactCache
from app.models import CacheLayer, QueryResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cache_config():
    cfg = MagicMock()
    cfg.cache.l1_ttl_seconds = 30
    cfg.cache.l1_max_entries = 10000
    cfg.cache.l2_similarity_threshold = 0.92
    cfg.cache.l2_ttl_seconds = 60
    cfg.cache.l2_max_entries = 5000
    cfg.cache.l3_tickers = ["AAPL", "NVDA"]
    cfg.cache.l3_refresh_interval_seconds = 30
    cfg.redis.host = "localhost"
    cfg.redis.port = 6379
    cfg.redis.db = 0
    cfg.redis.password = None
    cfg.redis.decode_responses = True
    cfg.redis.socket_timeout = 0.5
    return cfg


@patch("app.cache.exact_cache._redis_available", False)
@patch("app.cache.exact_cache.get_config")
@patch("app.cache.semantic_cache.get_config")
@patch("app.cache.hot_ticker_cache.get_config")
def _make_cache_manager(mock_htc, mock_sc, mock_ec):
    """Create a CacheManager backed by a pure in-memory ExactCache."""
    cfg = _make_cache_config()
    mock_htc.return_value = cfg
    mock_sc.return_value = cfg
    mock_ec.return_value = cfg
    exact = ExactCache()
    exact._redis_enabled = False
    return CacheManager(exact_cache=exact)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l1_invalidation_by_ticker():
    """Invalidating by ticker removes matching L1 entries."""
    manager = _make_cache_manager()
    response = QueryResponse(answer="AAPL data")

    await manager.set("What is AAPL?", None, response, tickers=["AAPL"])
    counts = await manager.invalidate_by_tickers(["AAPL"])

    assert counts["l1"] >= 1
    layer, cached = await manager.get("What is AAPL?")
    assert layer == CacheLayer.MISS
    assert cached is None


@pytest.mark.asyncio
async def test_l1_invalidation_preserves_other_tickers():
    """Invalidating AAPL preserves MSFT entries."""
    manager = _make_cache_manager()

    await manager.set(
        "What is AAPL?",
        None,
        QueryResponse(answer="AAPL data"),
        tickers=["AAPL"],
    )
    await manager.set(
        "What is MSFT?",
        None,
        QueryResponse(answer="MSFT data"),
        tickers=["MSFT"],
    )

    await manager.invalidate_by_tickers(["AAPL"])

    # AAPL should be gone
    layer_aapl, cached_aapl = await manager.get("What is AAPL?")
    assert layer_aapl == CacheLayer.MISS

    # MSFT should remain
    layer_msft, cached_msft = await manager.get("What is MSFT?")
    assert layer_msft == CacheLayer.L1_EXACT
    assert cached_msft is not None
    assert cached_msft.answer == "MSFT data"


@pytest.mark.asyncio
async def test_invalidation_returns_counts():
    """invalidate_by_tickers returns a dict with per-layer counts."""
    manager = _make_cache_manager()
    await manager.set("q1", None, QueryResponse(answer="a"), tickers=["AAPL"])
    await manager.set("q2", None, QueryResponse(answer="b"), tickers=["AAPL"])

    counts = await manager.invalidate_by_tickers(["AAPL"])

    assert isinstance(counts, dict)
    assert "l1" in counts
    assert counts["l1"] == 2


@pytest.mark.asyncio
async def test_empty_ticker_list_no_op():
    """Invalidating with empty ticker list removes nothing."""
    manager = _make_cache_manager()
    await manager.set("query", None, QueryResponse(answer="data"), tickers=["AAPL"])

    counts = await manager.invalidate_by_tickers([])

    assert counts["l1"] == 0
    layer, cached = await manager.get("query")
    assert layer == CacheLayer.L1_EXACT
    assert cached is not None


@pytest.mark.asyncio
async def test_cache_invalidator_removes_entries():
    """CacheInvalidator.invalidate removes matching cache entries."""
    from app.consistency.cache_invalidator import CacheInvalidator

    manager = _make_cache_manager()
    invalidator = CacheInvalidator(manager)

    await manager.set("AAPL info", None, QueryResponse(answer="data"), tickers=["AAPL"])

    record = await invalidator.invalidate(["AAPL"], source="test")

    assert record.total_invalidated >= 1
    layer, cached = await manager.get("AAPL info")
    assert layer == CacheLayer.MISS
