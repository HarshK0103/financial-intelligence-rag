"""
Financial RAG System — Circuit Breaker

Implements the classic circuit-breaker pattern to protect downstream
dependencies (e.g. an LLM API, embedding service, or external data
feed) from cascading failures.

State machine
─────────────
    CLOSED  ──failure_threshold──▶  OPEN
      ▲                               │
      │ success                        │ recovery_timeout
      │                               ▼
    CLOSED  ◀──── success ────  HALF_OPEN
                                   │
                                   │ failure
                                   ▼
                                  OPEN

Configuration is pulled from :pydata:`ResilienceConfig` at init time
so all thresholds are centrally tuneable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TypeVar

from app.config import get_config
from app.models import CircuitState

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""

    def __init__(self, name: str, retry_after: float) -> None:
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{name}' is OPEN — "
            f"retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """Wraps any async callable with circuit-breaker protection.

    Parameters
    ----------
    name : str
        Human-readable label for logging (e.g. ``"llm_api"``).
    failure_threshold : int | None
        Consecutive failures before the circuit trips.  Defaults to
        ``config.resilience.cb_failure_threshold``.
    recovery_timeout : float | None
        Seconds to wait in OPEN before moving to HALF_OPEN.  Defaults
        to ``config.resilience.cb_recovery_timeout_seconds``.
    half_open_max_calls : int | None
        Maximum probe calls allowed while HALF_OPEN.  Defaults to
        ``config.resilience.cb_half_open_max_calls``.
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
        half_open_max_calls: int | None = None,
    ) -> None:
        cfg = get_config().resilience

        self._name = name
        self._failure_threshold = failure_threshold or cfg.cb_failure_threshold
        self._recovery_timeout = recovery_timeout or cfg.cb_recovery_timeout_seconds
        self._half_open_max = half_open_max_calls or cfg.cb_half_open_max_calls

        # Mutable state
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._half_open_calls: int = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

        # Counters (diagnostic)
        self._total_calls: int = 0
        self._total_failures: int = 0
        self._total_rejections: int = 0

        logger.info(
            "CircuitBreaker '%s' created  threshold=%d  "
            "recovery=%.1fs  half_open_max=%d",
            self._name,
            self._failure_threshold,
            self._recovery_timeout,
            self._half_open_max,
        )

    # ── public API ────────────────────────────────────────────────

    async def call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute *func* through the circuit breaker.

        Parameters
        ----------
        func : Callable[..., Awaitable[T]]
            An async callable to protect.
        *args, **kwargs
            Forwarded to *func*.

        Returns
        -------
        T
            Whatever *func* returns on success.

        Raises
        ------
        CircuitBreakerOpenError
            If the circuit is OPEN and the recovery timeout has not
            yet elapsed.
        Exception
            Any exception raised by *func* is re-raised after the
            failure is recorded.
        """
        async with self._lock:
            self._maybe_transition_to_half_open()
            self._assert_callable()

        self._total_calls += 1

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            await self._record_failure()
            raise exc
        else:
            await self._record_success()
            return result

    # ── properties ────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may auto-transition on read)."""
        self._maybe_transition_to_half_open()
        return self._state

    @property
    def failure_count(self) -> int:
        """Number of consecutive failures since last reset."""
        return self._failure_count

    @property
    def is_open(self) -> bool:
        """True when the circuit is OPEN (rejecting calls)."""
        return self.state == CircuitState.OPEN

    @property
    def name(self) -> str:
        return self._name

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def total_failures(self) -> int:
        return self._total_failures

    @property
    def total_rejections(self) -> int:
        return self._total_rejections

    def reset(self) -> None:
        """Force the breaker back to CLOSED (e.g. after a manual fix)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        logger.info("CircuitBreaker '%s' manually reset to CLOSED.", self._name)

    # ── state transitions ─────────────────────────────────────────

    def _maybe_transition_to_half_open(self) -> None:
        """If OPEN and recovery timeout has elapsed, move to HALF_OPEN."""
        if self._state != CircuitState.OPEN:
            return
        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= self._recovery_timeout:
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            logger.info(
                "CircuitBreaker '%s' → HALF_OPEN after %.1fs recovery",
                self._name,
                elapsed,
            )

    def _assert_callable(self) -> None:
        """Raise if the circuit should reject the call."""
        if self._state == CircuitState.OPEN:
            retry_after = max(
                0.0,
                self._recovery_timeout
                - (time.monotonic() - self._last_failure_time),
            )
            self._total_rejections += 1
            raise CircuitBreakerOpenError(self._name, retry_after)

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self._half_open_max:
                # Already exhausted probe budget — treat as OPEN.
                self._total_rejections += 1
                raise CircuitBreakerOpenError(self._name, self._recovery_timeout)
            self._half_open_calls += 1

    async def _record_success(self) -> None:
        """Handle a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._half_open_max:
                    # Enough successful probes — circuit is healthy.
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_calls = 0
                    logger.info(
                        "CircuitBreaker '%s' → CLOSED  "
                        "(recovered after %d successes)",
                        self._name,
                        self._half_open_max,
                    )
            else:
                # CLOSED — reset consecutive failure count.
                self._failure_count = 0
                self._success_count = 0

    async def _record_failure(self) -> None:
        """Handle a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure during probing re-opens the circuit.
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                self._success_count = 0
                logger.warning(
                    "CircuitBreaker '%s' → OPEN  "
                    "(probe failed, failure_count=%d)",
                    self._name,
                    self._failure_count,
                )
            elif self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker '%s' → OPEN  "
                    "(threshold %d reached)",
                    self._name,
                    self._failure_threshold,
                )
