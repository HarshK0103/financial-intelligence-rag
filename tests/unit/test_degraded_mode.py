"""Tests for the degraded mode response generator."""

import pytest

from app.models import CacheLayer
from app.resilience.degraded_mode import DegradedMode


@pytest.fixture
def degraded() -> DegradedMode:
    return DegradedMode(max_tokens=100)


# ── Canned response (no cached data) ─────────────────────────────


def test_canned_response_is_degraded(degraded: DegradedMode) -> None:
    response = degraded.generate_degraded_response("What is AAPL?", reason="timeout")
    assert response.is_degraded is True


def test_canned_response_has_answer(degraded: DegradedMode) -> None:
    response = degraded.generate_degraded_response("test query", reason="unknown")
    assert len(response.answer) > 0


def test_canned_response_for_timeout(degraded: DegradedMode) -> None:
    response = degraded.generate_degraded_response("test", reason="timeout")
    assert "latency" in response.answer.lower() or "degraded" in response.answer.lower()


def test_canned_response_for_circuit_open(degraded: DegradedMode) -> None:
    response = degraded.generate_degraded_response("test", reason="circuit_open")
    assert response.is_degraded is True
    assert len(response.answer) > 0


def test_canned_response_for_unknown_reason(degraded: DegradedMode) -> None:
    response = degraded.generate_degraded_response("test", reason="some_new_reason")
    assert response.is_degraded is True


# ── Cached data path ─────────────────────────────────────────────


def test_cached_data_is_used(degraded: DegradedMode) -> None:
    cached = {
        "answer": "AAPL is trading at $195.50",
        "sources": [],
        "query_type": "price",
        "cache_layer": "l1_exact",
    }
    response = degraded.generate_degraded_response("AAPL price", cached_data=cached, reason="timeout")
    assert response.is_degraded is True
    assert "195.50" in response.answer


def test_cached_data_with_empty_answer_falls_back(degraded: DegradedMode) -> None:
    cached = {"answer": "", "sources": []}
    response = degraded.generate_degraded_response("test", cached_data=cached, reason="timeout")
    assert response.is_degraded is True
    assert len(response.answer) > 0  # Should use canned message


# ── Counters ──────────────────────────────────────────────────────


def test_counters_increment(degraded: DegradedMode) -> None:
    assert degraded._total_degraded == 0
    degraded.generate_degraded_response("q1", reason="timeout")
    degraded.generate_degraded_response("q2", reason="overload")
    assert degraded._total_degraded == 2
