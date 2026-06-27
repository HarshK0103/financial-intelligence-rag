"""Tests for Redis integration in ExactCache."""

from __future__ import annotations

import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.cache.exact_cache import ExactCache
from app.models import QueryResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config():
    cfg = MagicMock()
    cfg.cache.l1_ttl_seconds = 30
    cfg.cache.l1_max_entries = 10000
    cfg.redis.host = "localhost"
    cfg.redis.port = 6379
    cfg.redis.db = 0
    cfg.redis.password = None
    cfg.redis.decode_responses = True
    cfg.redis.socket_timeout = 0.5
    return cfg


@patch("app.cache.exact_cache._redis_available", False)
@patch("app.cache.exact_cache.get_config")
def _make_cache(mock_config) -> ExactCache:
    """Create an ExactCache with Redis disabled at module level."""
    mock_config.return_value = _make_config()
    cache = ExactCache()
    cache._redis_enabled = False
    return cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_to_memory_when_redis_unavailable():
    """set/get works in pure in-memory mode when Redis is unavailable."""
    cache = _make_cache()
    response = QueryResponse(answer="cached answer")

    await cache.set("test query", response, tickers=["AAPL"])
    result = await cache.get("test query")

    assert result is not None
    assert result.answer == "cached answer"


@pytest.mark.asyncio
async def test_health_check_disconnected():
    """health_check reports disconnected when Redis is not connected."""
    cache = _make_cache()
    health = await cache.health_check()

    assert health["redis_status"] == "disconnected"
    assert health["redis_connected"] is False
    assert "redis_host" in health


@pytest.mark.asyncio
async def test_health_check_connected_status():
    """health_check reports connected when _redis_connected is True."""
    cache = _make_cache()
    cache._redis_connected = True

    health = await cache.health_check()

    assert health["redis_status"] == "connected"
    assert health["redis_connected"] is True


@pytest.mark.asyncio
@patch("app.cache.exact_cache._redis_available", True)
async def test_reconnect_after_interval():
    """_get_redis retries connection after reconnect_interval has elapsed."""
    cache = _make_cache()
    cache._redis_enabled = False
    cache._redis_last_failure = time.time() - 60  # 60s ago, past 30s interval
    cache._redis_reconnect_interval = 30.0

    # Mock the Redis constructor to fail
    with patch("app.cache.exact_cache.aioredis") as mock_redis_mod:
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping = AsyncMock(side_effect=ConnectionError("still down"))
        mock_redis_mod.Redis.return_value = mock_redis_instance

        result = await cache._get_redis()

    # Should have attempted reconnect (returns None on failure)
    assert result is None
    # _redis_enabled set back to False after failure
    assert cache._redis_enabled is False


@pytest.mark.asyncio
@patch("app.cache.exact_cache._redis_available", True)
async def test_no_reconnect_before_interval():
    """_get_redis returns None immediately if reconnect interval hasn't elapsed."""
    cache = _make_cache()
    cache._redis_enabled = False
    cache._redis_last_failure = time.time() - 5  # Only 5s ago
    cache._redis_reconnect_interval = 30.0

    result = await cache._get_redis()

    assert result is None


@pytest.mark.asyncio
async def test_redis_metrics_tracking():
    """hits/misses counters increment correctly on in-memory operations."""
    cache = _make_cache()
    response = QueryResponse(answer="test")

    await cache.set("query1", response)
    await cache.get("query1")  # hit
    await cache.get("query1")  # hit
    await cache.get("nonexistent")  # miss

    assert cache.hits == 2
    assert cache.misses == 1
