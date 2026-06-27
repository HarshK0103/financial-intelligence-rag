"""Tests for the resilience chain: CircuitBreaker + DegradedMode + TimeoutHandler."""

from __future__ import annotations

import asyncio

import pytest

from app.models import CircuitState, CacheLayer, QueryResponse
from app.resilience.circuit_breaker import CircuitBreaker
from app.resilience.degraded_mode import DegradedMode
from app.resilience.timeout_handler import TimeoutHandler


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


async def failing_func():
    raise RuntimeError("service down")


async def succeeding_func():
    return "ok"


async def slow_func():
    await asyncio.sleep(5)
    return "too slow"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_trips_after_failures():
    """Circuit trips OPEN after consecutive failures exceed threshold."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(failing_func)

    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_degraded_mode_after_circuit_trips():
    """DegradedMode produces is_degraded=True responses."""
    degraded = DegradedMode()
    response = degraded.generate_degraded_response(
        query="What is AAPL price?",
        cached_data=None,
        reason="Circuit breaker is OPEN",
    )

    assert isinstance(response, QueryResponse)
    assert response.is_degraded is True
    assert len(response.answer) > 0


@pytest.mark.asyncio
async def test_timeout_triggers_fallback_value():
    """TimeoutHandler returns fallback when coroutine exceeds budget."""
    handler = TimeoutHandler()
    result = await handler.execute_with_timeout(
        coro=slow_func(),
        timeout_ms=50,
        stage_name="test",
        fallback_value="fallback_result",
    )

    assert result == "fallback_result"


@pytest.mark.asyncio
async def test_full_resilience_chain():
    """Full chain: circuit trips → degraded response."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30)
    degraded = DegradedMode()

    # Trip the circuit
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(failing_func)

    assert cb.state == CircuitState.OPEN

    # Now generate degraded response
    response = degraded.generate_degraded_response(
        query="What is NVDA revenue?",
        cached_data=None,
        reason=f"Circuit breaker is {cb.state.value}",
    )

    assert response.is_degraded is True
    assert len(response.answer) > 0


@pytest.mark.asyncio
async def test_circuit_recovery_after_success():
    """Circuit recovers from OPEN to CLOSED after successful calls in HALF_OPEN."""
    cb = CircuitBreaker(
        failure_threshold=2, recovery_timeout=0.05, half_open_max_calls=1
    )

    # Trip the circuit
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.call(failing_func)

    assert cb.state == CircuitState.OPEN

    # Wait for recovery timeout so it transitions to HALF_OPEN
    await asyncio.sleep(0.15)

    # Successful call should reset to CLOSED
    result = await cb.call(succeeding_func)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED
