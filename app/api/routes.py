"""
Financial RAG System — API Routes

All HTTP endpoints for querying, health monitoring, metrics,
and data ingestion.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models import (
    CacheLayer,
    IngestionEvent,
    QueryRequest,
    QueryResponse,
    ResponseMetrics,
)
from app.observability.metrics import observe_query_response

logger = logging.getLogger("financial_rag.api")

router = APIRouter(prefix="/api", tags=["rag"])

# ── Module-level references (set by main.py on startup) ───────────
_pipeline: Any = None
_health_tracker: Any = None


class _HealthTracker:
    """Tracks aggregate system health metrics."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.total_queries = 0
        self.latencies: deque[float] = deque(maxlen=1000)
        self.cache_hits = {
            CacheLayer.L1_EXACT: 0,
            CacheLayer.L2_SEMANTIC: 0,
            CacheLayer.L3_HOT_TICKER: 0,
            CacheLayer.MISS: 0,
        }
        self.recent_queries: deque[dict] = deque(maxlen=50)

    def record_query(
        self,
        query: str,
        response: QueryResponse,
        latency_ms: float,
    ) -> None:
        self.total_queries += 1
        self.latencies.append(latency_ms)
        self.cache_hits[response.cache_layer] = self.cache_hits.get(response.cache_layer, 0) + 1
        self.recent_queries.appendleft(
            {
                "query": query[:80],
                "latency_ms": round(latency_ms, 2),
                "cache_layer": response.cache_layer.value,
                "is_degraded": response.is_degraded,
                "timestamp": time.time(),
                "query_type": response.query_type.value,
            }
        )

    def get_percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * p / 100)
        idx = min(idx, len(sorted_lat) - 1)
        return sorted_lat[idx]

    def get_health(self) -> dict:
        total_hits = sum(self.cache_hits.values())
        cache_misses = self.cache_hits.get(CacheLayer.MISS, 0)
        cache_hit_rate = (total_hits - cache_misses) / total_hits * 100 if total_hits > 0 else 0.0
        l1 = self.cache_hits.get(CacheLayer.L1_EXACT, 0)
        l2 = self.cache_hits.get(CacheLayer.L2_SEMANTIC, 0)
        l3 = self.cache_hits.get(CacheLayer.L3_HOT_TICKER, 0)

        return {
            "status": "healthy",
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "total_queries": self.total_queries,
            "avg_latency_ms": round(sum(self.latencies) / len(self.latencies), 2) if self.latencies else 0.0,
            "p50_latency_ms": round(self.get_percentile(50), 2),
            "p75_latency_ms": round(self.get_percentile(75), 2),
            "p95_latency_ms": round(self.get_percentile(95), 2),
            "p99_latency_ms": round(self.get_percentile(99), 2),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "l1_hit_rate": round(l1 / total_hits * 100, 2) if total_hits else 0.0,
            "l2_hit_rate": round(l2 / total_hits * 100, 2) if total_hits else 0.0,
            "l3_hit_rate": round(l3 / total_hits * 100, 2) if total_hits else 0.0,
            "l1_hits": l1,
            "l2_hits": l2,
            "l3_hits": l3,
            "cache_misses": cache_misses,
            "circuit_state": "closed",
        }


# Create the global tracker
_health_tracker = _HealthTracker()


def set_pipeline(pipeline: Any) -> None:
    """Set the query pipeline reference (called from main.py)."""
    global _pipeline
    _pipeline = pipeline


def get_health_tracker() -> _HealthTracker:
    """Return the health tracker singleton."""
    return _health_tracker


# ── Endpoints ──────────────────────────────────────────────────────


@router.post("/query")
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Process a financial query through the full RAG pipeline.

    The pipeline runs: cache lookup → hybrid retrieval → reranking → inference.
    Each stage has a hard latency budget to stay under 200ms total.
    """
    start_time = time.perf_counter()

    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Server is still starting.",
        )

    try:
        response = await _pipeline.process_query(request)
    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        # Return a degraded response rather than a 500
        response = QueryResponse(
            answer=f"Service temporarily unavailable. Error: {type(e).__name__}",
            cache_layer=CacheLayer.MISS,
            is_degraded=True,
            metrics=ResponseMetrics(
                total_latency_ms=(time.perf_counter() - start_time) * 1000,
            ),
        )

    latency_ms = (time.perf_counter() - start_time) * 1000

    # Ensure metrics reflect actual total latency
    if response.metrics:
        response.metrics.total_latency_ms = round(latency_ms, 2)
    else:
        response.metrics = ResponseMetrics(total_latency_ms=round(latency_ms, 2))

    serialization_start = time.perf_counter()
    response.model_dump(mode="json")
    response.metrics.serialization_ms = round((time.perf_counter() - serialization_start) * 1000, 2)

    _health_tracker.record_query(request.query, response, latency_ms)
    observe_query_response(response, _health_tracker)

    return response


@router.get("/health")
async def health_endpoint() -> dict:
    """
    Return aggregate system health metrics.

    Used by the monitoring dashboard to display real-time status.
    """
    health = _health_tracker.get_health()

    # Add pipeline-specific info if available
    if _pipeline is not None:
        try:
            health["hot_store_docs"] = await _pipeline.data_router.hot_store.count()
            health["cold_store_docs"] = await _pipeline.data_router.cold_store.count()
            health["circuit_state"] = _pipeline.circuit_breaker.state.value
            health["ingestion_queue_size"] = _pipeline.stream_processor.queue_size
            health["inference_backend"] = getattr(_pipeline, "inference_backend", "unknown")
            health["active_tickers"] = (
                list(_pipeline.hot_ticker_cache.get_cached_tickers()) if hasattr(_pipeline, "hot_ticker_cache") else []
            )
            if hasattr(_pipeline, "connector_sync_service"):
                health["connectors"] = _pipeline.connector_sync_service.status_snapshot()

            # Redis status
            if hasattr(_pipeline.cache_manager, "l1"):
                redis_info = await _pipeline.cache_manager.l1.health_check()
                health["redis_status"] = redis_info.get("redis_status", "unknown")
                health["redis_connected"] = redis_info.get("redis_connected", False)
                health["redis_host"] = redis_info.get("redis_host", "")
                health["redis_hits"] = redis_info.get("redis_hits", 0)
                health["redis_misses"] = redis_info.get("redis_misses", 0)
                health["redis_reconnects"] = redis_info.get("redis_reconnects", 0)

            # Ollama status
            inference = _pipeline.inference_engine
            if hasattr(inference, "health_check"):
                ollama_info = await inference.health_check()
                health["ollama_status"] = ollama_info.get("ollama_status", "unknown")
                health["ollama_model"] = ollama_info.get("ollama_model", "")
            else:
                health["ollama_status"] = "disabled"
                health["ollama_model"] = ""

        except Exception:
            pass

    return health


@router.get("/metrics")
async def metrics_endpoint() -> dict:
    """
    Return detailed performance metrics.

    Includes per-layer cache stats, latency distribution,
    and pipeline component health.
    """
    health = _health_tracker.get_health()

    metrics = {
        **health,
        "latency_distribution": {
            "p50": health.get("p50_latency_ms", 0),
            "p75": health.get("p75_latency_ms", 0),
            "p95": health.get("p95_latency_ms", 0),
            "p99": health.get("p99_latency_ms", 0),
        },
        "cache_breakdown": {
            "l1_exact": health.get("l1_hits", 0),
            "l2_semantic": health.get("l2_hits", 0),
            "l3_hot_ticker": health.get("l3_hits", 0),
            "miss": health.get("cache_misses", 0),
        },
        "sla_budget_ms": 200,
    }

    return metrics


@router.get("/recent-queries")
async def recent_queries_endpoint() -> dict:
    """
    Return the most recent queries with their performance data.

    Used by the dashboard's live query log panel.
    """
    return {"queries": list(_health_tracker.recent_queries)}


class IngestRequest(BaseModel):
    """Request body for the ingest endpoint."""

    documents: list[dict]
    source: str = "manual"


@router.post("/ingest")
async def ingest_endpoint(request: IngestRequest) -> dict:
    """
    Ingest new financial data into the system.

    Documents are processed asynchronously — this endpoint returns
    immediately without blocking the retrieval path.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    try:
        from app.models import Document

        docs = []
        for d in request.documents:
            docs.append(Document(**d))

        event = IngestionEvent(
            event_id=hashlib.md5(f"{time.time()}_{len(docs)}".encode()).hexdigest()[:12],
            documents=docs,
            source=request.source,
        )

        await _pipeline.ingest(event)

        return {
            "status": "accepted",
            "event_id": event.event_id,
            "documents_queued": len(docs),
        }
    except Exception as e:
        logger.error("Ingestion error: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/invalidate-cache")
async def invalidate_cache_endpoint(tickers: list[str]) -> dict:
    """
    Manually invalidate cache entries for specific tickers.

    Used for testing cache invalidation behavior.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    try:
        count = await _pipeline.cache_manager.invalidate_by_tickers(tickers)
        return {
            "status": "invalidated",
            "tickers": tickers,
            "entries_invalidated": count,
        }
    except Exception as e:
        logger.error("Cache invalidation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connectors/status")
async def connector_status_endpoint() -> dict:
    """Return connector runtime status."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    if not hasattr(_pipeline, "connector_sync_service"):
        return {"connectors": {}}
    return {"connectors": _pipeline.connector_sync_service.status_snapshot()}


@router.post("/connectors/sync")
async def connector_sync_endpoint(connector_name: str | None = None) -> dict:
    """Trigger one immediate connector sync."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    if not hasattr(_pipeline, "connector_sync_service"):
        raise HTTPException(status_code=503, detail="Connector sync service not available.")
    counts = await _pipeline.connector_sync_service.sync_once(connector_name)
    return {"status": "ok", "counts": counts}
