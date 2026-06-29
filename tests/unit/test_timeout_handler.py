"""Tests for the timeout handler."""

import asyncio

import pytest

from app.resilience.timeout_handler import TimeoutHandler


@pytest.fixture
def handler() -> TimeoutHandler:
    return TimeoutHandler(timeout_multiplier=1.0)


# ── Fast coroutine completes ──────────────────────────────────────


@pytest.mark.asyncio
async def test_fast_coro_returns_result(handler: TimeoutHandler) -> None:
    async def fast() -> str:
        return "done"

    result = await handler.execute_with_timeout(fast(), timeout_ms=1000, stage_name="test")
    assert result == "done"


@pytest.mark.asyncio
async def test_fast_coro_counter(handler: TimeoutHandler) -> None:
    async def fast() -> int:
        return 42

    await handler.execute_with_timeout(fast(), timeout_ms=1000, stage_name="test")
    assert handler._total_calls == 1
    assert handler._total_timeouts == 0


# ── Slow coroutine times out ──────────────────────────────────────


@pytest.mark.asyncio
async def test_slow_coro_returns_fallback(handler: TimeoutHandler) -> None:
    async def slow() -> str:
        await asyncio.sleep(5)
        return "too late"

    result = await handler.execute_with_timeout(
        slow(),
        timeout_ms=10,  # 10ms — will timeout
        stage_name="slow_test",
        fallback_value="fallback",
    )
    assert result == "fallback"


@pytest.mark.asyncio
async def test_timeout_increments_counter(handler: TimeoutHandler) -> None:
    async def slow() -> None:
        await asyncio.sleep(5)

    await handler.execute_with_timeout(slow(), timeout_ms=10, stage_name="test", fallback_value=None)
    assert handler._total_timeouts == 1


# ── Fallback value defaults ──────────────────────────────────────


@pytest.mark.asyncio
async def test_default_fallback_is_none(handler: TimeoutHandler) -> None:
    async def slow() -> str:
        await asyncio.sleep(5)
        return "never"

    result = await handler.execute_with_timeout(slow(), timeout_ms=10, stage_name="test")
    assert result is None


@pytest.mark.asyncio
async def test_custom_fallback_value(handler: TimeoutHandler) -> None:
    async def slow() -> list:
        await asyncio.sleep(5)
        return [1, 2, 3]

    result = await handler.execute_with_timeout(slow(), timeout_ms=10, stage_name="test", fallback_value=[])
    assert result == []


# ── Exception propagation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_exception_propagates(handler: TimeoutHandler) -> None:
    async def failing() -> None:
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await handler.execute_with_timeout(failing(), timeout_ms=1000, stage_name="test")


# ── Multiplier ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiplier_extends_deadline() -> None:
    handler = TimeoutHandler(timeout_multiplier=10.0)

    async def moderate() -> str:
        await asyncio.sleep(0.05)  # 50ms
        return "ok"

    # Budget is 10ms but multiplier is 10× → effective is 100ms
    result = await handler.execute_with_timeout(moderate(), timeout_ms=10, stage_name="test")
    assert result == "ok"
