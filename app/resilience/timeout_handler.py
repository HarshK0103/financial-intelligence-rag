"""
Financial RAG System — Timeout Handler

Enforces per-stage latency budgets so that no single pipeline stage
can blow the overall SLA.  Uses :func:`asyncio.wait_for` under the
hood and returns a caller-specified fallback value on timeout rather
than raising.

Usage example
─────────────
.. code-block:: python

    handler = TimeoutHandler()
    result = await handler.execute_with_timeout(
        coro=retrieval.search(query),
        timeout_ms=config.latency.retrieval_ms,
        stage_name="retrieval",
        fallback_value=[],
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from typing import TypeVar

from app.config import get_config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutHandler:
    """Wrap coroutines with hard timeout enforcement and fallback values.

    Parameters
    ----------
    timeout_multiplier : float | None
        Multiplied against ``timeout_ms`` to give a slight grace
        period (e.g. 1.5×).  Defaults to
        ``config.resilience.stage_timeout_multiplier``.
    """

    def __init__(self, timeout_multiplier: float | None = None) -> None:
        cfg = get_config()
        self._multiplier = (
            timeout_multiplier
            if timeout_multiplier is not None
            else cfg.resilience.stage_timeout_multiplier
        )

        # Diagnostic counters
        self._total_calls: int = 0
        self._total_timeouts: int = 0

        logger.info(
            "TimeoutHandler initialised  multiplier=%.2f",
            self._multiplier,
        )

    # ── public API ────────────────────────────────────────────────

    async def execute_with_timeout(
        self,
        coro: Coroutine[None, None, T],
        timeout_ms: float,
        stage_name: str = "unknown",
        fallback_value: T | None = None,
    ) -> T | None:
        """Run *coro* with a hard timeout, returning *fallback_value* on expiry.

        Parameters
        ----------
        coro : Coroutine
            The awaitable to execute.
        timeout_ms : float
            Budget in **milliseconds**.  The actual deadline applied
            is ``timeout_ms * self._multiplier``.
        stage_name : str
            Label used in log messages (e.g. ``"retrieval"``).
        fallback_value : T | None
            Returned in place of the coroutine result when a timeout
            occurs.

        Returns
        -------
        T | None
            The coroutine's result, or *fallback_value* on timeout.
        """
        self._total_calls += 1
        effective_ms = timeout_ms * self._multiplier
        effective_s = effective_ms / 1000.0

        t0 = time.perf_counter()

        try:
            result: T = await asyncio.wait_for(coro, timeout=effective_s)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if elapsed_ms > timeout_ms:
                # Completed but exceeded the soft budget (within
                # the multiplied grace period).
                logger.warning(
                    "Stage '%s' completed but exceeded soft budget: "
                    "%.1fms > %.1fms budget  (hard limit %.1fms)",
                    stage_name,
                    elapsed_ms,
                    timeout_ms,
                    effective_ms,
                )

            return result

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._total_timeouts += 1
            logger.error(
                "Stage '%s' TIMED OUT after %.1fms  "
                "(budget=%.1fms  hard_limit=%.1fms) — "
                "returning fallback value",
                stage_name,
                elapsed_ms,
                timeout_ms,
                effective_ms,
            )
            return fallback_value

        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.exception(
                "Stage '%s' raised an exception after %.1fms",
                stage_name,
                elapsed_ms,
            )
            raise

    # ── convenience wrappers ──────────────────────────────────────

    async def execute_retrieval(
        self,
        coro: Coroutine[None, None, T],
        fallback_value: T | None = None,
    ) -> T | None:
        """Shortcut for the retrieval stage using the configured budget."""
        cfg = get_config()
        return await self.execute_with_timeout(
            coro=coro,
            timeout_ms=cfg.latency.retrieval_ms,
            stage_name="retrieval",
            fallback_value=fallback_value,
        )

    async def execute_inference(
        self,
        coro: Coroutine[None, None, T],
        fallback_value: T | None = None,
    ) -> T | None:
        """Shortcut for the inference stage using the configured budget."""
        cfg = get_config()
        return await self.execute_with_timeout(
            coro=coro,
            timeout_ms=cfg.latency.inference_ms,
            stage_name="inference",
            fallback_value=fallback_value,
        )

    async def execute_cache_lookup(
        self,
        coro: Coroutine[None, None, T],
        fallback_value: T | None = None,
    ) -> T | None:
        """Shortcut for cache lookup using the configured budget."""
        cfg = get_config()
        return await self.execute_with_timeout(
            coro=coro,
            timeout_ms=cfg.latency.cache_lookup_ms,
            stage_name="cache_lookup",
            fallback_value=fallback_value,
        )

    # ── diagnostics ───────────────────────────────────────────────

    @property
    def total_calls(self) -> int:
        """Total number of ``execute_with_timeout`` invocations."""
        return self._total_calls

    @property
    def total_timeouts(self) -> int:
        """Total number of timeout events recorded."""
        return self._total_timeouts

    @property
    def timeout_rate(self) -> float:
        """Fraction of calls that timed out (0.0 – 1.0)."""
        if self._total_calls == 0:
            return 0.0
        return self._total_timeouts / self._total_calls
