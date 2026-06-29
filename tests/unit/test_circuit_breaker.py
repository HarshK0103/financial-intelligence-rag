"""Tests for the circuit breaker pattern."""

import asyncio

import pytest

from app.models import CircuitState
from app.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError


@pytest.fixture
def breaker() -> CircuitBreaker:
    """Create a circuit breaker with low thresholds for fast testing."""
    return CircuitBreaker(
        name="test",
        failure_threshold=3,
        recovery_timeout=0.1,  # 100ms
        half_open_max_calls=2,
    )


async def _ok() -> str:
    return "ok"


async def _fail() -> str:
    raise RuntimeError("boom")


# ── State: CLOSED ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_starts_closed(breaker: CircuitBreaker) -> None:
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0
    assert not breaker.is_open


@pytest.mark.asyncio
async def test_successful_call_stays_closed(breaker: CircuitBreaker) -> None:
    result = await breaker.call(_ok)
    assert result == "ok"
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


@pytest.mark.asyncio
async def test_fewer_than_threshold_failures_stays_closed(
    breaker: CircuitBreaker,
) -> None:
    for _ in range(2):  # threshold is 3
        with pytest.raises(RuntimeError):
            await breaker.call(_fail)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 2


@pytest.mark.asyncio
async def test_success_resets_failure_count(breaker: CircuitBreaker) -> None:
    with pytest.raises(RuntimeError):
        await breaker.call(_fail)
    assert breaker.failure_count == 1

    await breaker.call(_ok)
    assert breaker.failure_count == 0


# ── Transition: CLOSED → OPEN ─────────────────────────────────────


@pytest.mark.asyncio
async def test_trips_open_after_threshold(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(_fail)
    assert breaker.state == CircuitState.OPEN
    assert breaker.is_open


@pytest.mark.asyncio
async def test_open_rejects_calls(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(_fail)

    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        await breaker.call(_ok)
    assert exc_info.value.name == "test"
    assert exc_info.value.retry_after >= 0


# ── Transition: OPEN → HALF_OPEN ──────────────────────────────────


@pytest.mark.asyncio
async def test_transitions_to_half_open_after_recovery(
    breaker: CircuitBreaker,
) -> None:
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(_fail)
    assert breaker.state == CircuitState.OPEN

    # Wait for recovery timeout
    await asyncio.sleep(0.15)
    assert breaker.state == CircuitState.HALF_OPEN


# ── Transition: HALF_OPEN → CLOSED ────────────────────────────────


@pytest.mark.asyncio
async def test_half_open_recovers_to_closed(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(_fail)
    await asyncio.sleep(0.15)
    assert breaker.state == CircuitState.HALF_OPEN

    # Enough successes to close
    for _ in range(2):  # half_open_max_calls = 2
        await breaker.call(_ok)
    assert breaker.state == CircuitState.CLOSED


# ── Transition: HALF_OPEN → OPEN ──────────────────────────────────


@pytest.mark.asyncio
async def test_half_open_failure_reopens(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(_fail)
    await asyncio.sleep(0.15)
    assert breaker.state == CircuitState.HALF_OPEN

    with pytest.raises(RuntimeError):
        await breaker.call(_fail)
    assert breaker.state == CircuitState.OPEN


# ── Reset ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_manual_reset(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(_fail)
    assert breaker.state == CircuitState.OPEN

    breaker.reset()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


# ── Counters ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_diagnostic_counters(breaker: CircuitBreaker) -> None:
    await breaker.call(_ok)
    with pytest.raises(RuntimeError):
        await breaker.call(_fail)

    assert breaker.total_calls == 2
    assert breaker.total_failures == 1

    # Trip it open
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(_fail)

    # Try calling while open → rejection
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(_ok)

    assert breaker.total_rejections >= 1
