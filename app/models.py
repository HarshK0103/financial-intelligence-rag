"""
Financial RAG System — Pydantic Data Models

All request/response schemas, internal data structures, and metrics
models used across the system.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ── Enums ──────────────────────────────────────────────────────────


class DataTemperature(str, Enum):
    """Whether data is hot (real-time) or cold (historical)."""

    HOT = "hot"
    COLD = "cold"


class CacheLayer(str, Enum):
    """Which cache layer served the response."""

    L1_EXACT = "l1_exact"
    L2_SEMANTIC = "l2_semantic"
    L3_HOT_TICKER = "l3_hot_ticker"
    MISS = "miss"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class QueryType(str, Enum):
    """Classification of incoming financial queries."""

    PRICE = "price"
    EARNINGS = "earnings"
    NEWS = "news"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    GENERAL = "general"


# ── Documents ──────────────────────────────────────────────────────


class Document(BaseModel):
    """A retrievable financial document chunk."""

    doc_id: str
    content: str
    source: str = ""  # e.g., "sec_filing", "news", "price_feed"
    ticker: str | None = None
    timestamp: float = Field(default_factory=time.time)
    temperature: DataTemperature = DataTemperature.COLD
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    version: int = 1

    class Config:
        json_encoders = {float: lambda v: round(v, 6)}


class ScoredDocument(BaseModel):
    """A document with retrieval and freshness scores."""

    document: Document
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0
    freshness_score: float = 1.0
    final_score: float = 0.0


# ── API Request / Response ─────────────────────────────────────────


class QueryRequest(BaseModel):
    """Incoming query from the client."""

    query: str = Field(..., min_length=1, max_length=1000)
    tickers: list[str] = Field(default_factory=list)
    max_results: int = Field(default=5, ge=1, le=20)
    require_fresh: bool = False  # Force bypass stale caches
    timeout_ms: int | None = None  # Client-specified SLA override

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "What was AAPL revenue in Q3 2024?", "tickers": ["AAPL"]},
                {"query": "Compare NVDA and AMD GPU revenue growth", "tickers": ["NVDA", "AMD"]},
            ]
        }
    }


class QueryResponse(BaseModel):
    """Response returned to the client."""

    answer: str
    sources: list[ScoredDocument] = Field(default_factory=list)
    query_type: QueryType = QueryType.GENERAL
    cache_layer: CacheLayer = CacheLayer.MISS
    is_degraded: bool = False  # True if served from degraded mode
    metrics: ResponseMetrics | None = None


class ResponseMetrics(BaseModel):
    """Latency and performance metrics attached to every response."""

    total_latency_ms: float = 0.0
    query_embedding_ms: float = 0.0
    cache_lookup_ms: float = 0.0
    l1_cache_ms: float = 0.0
    l2_cache_ms: float = 0.0
    l3_cache_ms: float = 0.0
    retrieval_ms: float = 0.0
    bm25_retrieval_ms: float = 0.0
    vector_retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    freshness_scoring_ms: float = 0.0
    inference_ms: float = 0.0
    serialization_ms: float = 0.0
    cache_hit: bool = False
    cache_layer: CacheLayer = CacheLayer.MISS
    documents_retrieved: int = 0
    documents_reranked: int = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    freshness_min: float = 0.0
    freshness_max: float = 0.0


# ── Cache Entries ──────────────────────────────────────────────────


class CacheEntry(BaseModel):
    """Entry stored in any cache layer."""

    query: str
    query_hash: str = ""
    response: QueryResponse
    embedding: list[float] | None = None  # For semantic cache
    created_at: float = Field(default_factory=time.time)
    ttl_seconds: float = 30.0
    access_count: int = 0
    tickers: list[str] = Field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


# ── Ingestion ──────────────────────────────────────────────────────


class IngestionEvent(BaseModel):
    """An event representing new data to ingest."""

    event_id: str
    documents: list[Document]
    source: str
    timestamp: float = Field(default_factory=time.time)
    priority: int = 0  # Higher = more urgent


class IngestionResult(BaseModel):
    """Result of processing an ingestion event."""

    event_id: str
    documents_indexed: int = 0
    documents_failed: int = 0
    caches_invalidated: int = 0
    processing_time_ms: float = 0.0


# ── System Health ──────────────────────────────────────────────────


class SystemHealth(BaseModel):
    """Aggregate system health metrics for the dashboard."""

    status: str = "healthy"
    uptime_seconds: float = 0.0
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0
    l3_hit_rate: float = 0.0
    circuit_state: CircuitState = CircuitState.CLOSED
    hot_store_docs: int = 0
    cold_store_docs: int = 0
    ingestion_queue_size: int = 0
    active_tickers: list[str] = Field(default_factory=list)
