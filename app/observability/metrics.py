"""Prometheus metrics integration with graceful fallback."""

from __future__ import annotations

import logging

from app.models import CacheLayer, QueryResponse

logger = logging.getLogger(__name__)

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    metrics_available = True
except ImportError:  # pragma: no cover - fallback for minimal local envs
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    Counter = Gauge = Histogram = None
    generate_latest = None
    metrics_available = False


if metrics_available:
    QUERY_LATENCY = Histogram(
        "query_latency",
        "End-to-end query latency in seconds",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
    )
    RETRIEVAL_LATENCY = Histogram(
        "retrieval_latency",
        "Retrieval stage latency in seconds",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5),
    )
    INFERENCE_LATENCY = Histogram(
        "inference_latency",
        "Inference stage latency in seconds",
        buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    )
    CACHE_HIT_RATE = Gauge(
        "cache_hit_rate",
        "Rolling cache hit rate across all cache layers",
    )
    FRESHNESS_SCORE = Histogram(
        "freshness_score",
        "Average freshness score of documents returned in a response",
        buckets=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
    )
    CACHE_REQUESTS = Counter(
        "cache_requests_total",
        "Cache lookups by layer and result",
        labelnames=("layer", "result"),
    )
    CONNECTOR_SYNC_DURATION = Histogram(
        "connector_sync_duration_seconds",
        "Connector sync duration in seconds",
        labelnames=("connector",),
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    )
    CONNECTOR_DOCUMENTS = Counter(
        "connector_documents_total",
        "Documents produced by connector sync",
        labelnames=("connector",),
    )
    CONNECTOR_ERRORS = Counter(
        "connector_errors_total",
        "Connector sync failures",
        labelnames=("connector",),
    )
else:
    QUERY_LATENCY = None
    RETRIEVAL_LATENCY = None
    INFERENCE_LATENCY = None
    CACHE_HIT_RATE = None
    FRESHNESS_SCORE = None
    CACHE_REQUESTS = None
    CONNECTOR_SYNC_DURATION = None
    CONNECTOR_DOCUMENTS = None
    CONNECTOR_ERRORS = None


def observe_query_response(response: QueryResponse, health_tracker) -> None:
    """Record query metrics from a completed response."""
    if not metrics_available or response.metrics is None:
        return

    metrics = response.metrics
    QUERY_LATENCY.observe(metrics.total_latency_ms / 1000.0)
    RETRIEVAL_LATENCY.observe(metrics.retrieval_ms / 1000.0)
    INFERENCE_LATENCY.observe(metrics.inference_ms / 1000.0)

    if metrics.documents_reranked > 0:
        avg_freshness = (metrics.freshness_min + metrics.freshness_max) / 2.0
        FRESHNESS_SCORE.observe(avg_freshness)

    total_hits = sum(health_tracker.cache_hits.values())
    total_misses = health_tracker.cache_hits.get(CacheLayer.MISS, 0)
    if total_hits:
        CACHE_HIT_RATE.set((total_hits - total_misses) / total_hits)

    result = "hit" if metrics.cache_hit else "miss"
    CACHE_REQUESTS.labels(layer=response.cache_layer.value, result=result).inc()


def observe_connector_sync(
    connector_name: str,
    *,
    duration_seconds: float,
    document_count: int,
    error: bool = False,
) -> None:
    """Record connector sync activity."""
    if not metrics_available:
        return
    CONNECTOR_SYNC_DURATION.labels(connector=connector_name).observe(duration_seconds)
    if document_count:
        CONNECTOR_DOCUMENTS.labels(connector=connector_name).inc(document_count)
    if error:
        CONNECTOR_ERRORS.labels(connector=connector_name).inc()


def render_prometheus_metrics() -> bytes:
    """Render the Prometheus exposition payload."""
    if not metrics_available or generate_latest is None:
        return b""
    return generate_latest()
