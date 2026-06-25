"""
Financial RAG System - Configuration & Constants

All latency budgets, cache TTLs, connector settings, inference runtime,
and system parameters are centralized here for easy tuning.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _env_json_dict(name: str, default: dict[str, str]) -> dict[str, str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return default
    if not isinstance(parsed, dict):
        return default
    return {str(key).upper(): str(value) for key, value in parsed.items()}


@dataclass(frozen=True)
class LatencyBudget:
    """Hard latency budgets per pipeline stage (milliseconds)."""

    total_sla_ms: int = 200
    api_gateway_ms: int = 5
    cache_lookup_ms: int = 15
    retrieval_ms: int = 80
    reranking_ms: int = 20
    inference_ms: int = 80
    serialization_ms: int = 5
    buffer_ms: int = 50


@dataclass(frozen=True)
class CacheConfig:
    """Multi-layer cache configuration."""

    l1_ttl_seconds: int = 30
    l1_max_entries: int = 10_000
    l2_similarity_threshold: float = 0.92
    l2_ttl_seconds: int = 60
    l2_max_entries: int = 5_000
    l3_refresh_interval_seconds: int = 30
    l3_tickers: tuple[str, ...] = (
        "AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META",
        "BTC", "ETH", "SPY", "QQQ", "JPM", "GS", "BAC",
    )


@dataclass(frozen=True)
class RetrievalConfig:
    """Hybrid retrieval parameters."""

    bm25_top_k: int = 20
    vector_top_k: int = 20
    rerank_top_k: int = 5
    fusion_weight_bm25: float = 0.4
    fusion_weight_vector: float = 0.6
    embedding_dim: int = 384
    hot_index_type: str = "flat"
    cold_index_type: str = "hnsw"


@dataclass(frozen=True)
class ConsistencyConfig:
    """Retrieval consistency parameters."""

    freshness_decay_halflife_seconds: float = 60.0
    stale_threshold_seconds: float = 300.0
    version_tracking_enabled: bool = True
    invalidation_propagation_ms: int = 50


@dataclass(frozen=True)
class ResilienceConfig:
    """Circuit breaker and degraded mode parameters."""

    cb_failure_threshold: int = 5
    cb_recovery_timeout_seconds: float = 30.0
    cb_half_open_max_calls: int = 3
    stage_timeout_multiplier: float = 1.5
    partial_result_enabled: bool = True
    degraded_cache_only: bool = True
    degraded_response_max_tokens: int = 100


@dataclass(frozen=True)
class InferenceConfig:
    """Inference backend configuration."""

    backend: str = os.getenv("INFERENCE_BACKEND", "ollama")
    max_output_tokens: int = _env_int("INFERENCE_MAX_OUTPUT_TOKENS", 300)
    temperature: float = _env_float("INFERENCE_TEMPERATURE", 0.1)
    prompt_compression_enabled: bool = _env_bool(
        "INFERENCE_PROMPT_COMPRESSION",
        True,
    )
    template_mode_enabled: bool = _env_bool(
        "INFERENCE_TEMPLATE_MODE",
        True,
    )


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama runtime configuration."""

    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    request_timeout_seconds: float = _env_float("OLLAMA_TIMEOUT_SECONDS", 60.0)
    keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
    stream: bool = False
    enabled: bool = _env_bool("OLLAMA_ENABLED", False)


@dataclass(frozen=True)
class RedisConfig:
    """Redis connection configuration."""

    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = _env_int("REDIS_PORT", 6379)
    db: int = _env_int("REDIS_DB", 0)
    password: str | None = os.getenv("REDIS_PASSWORD")
    decode_responses: bool = True
    socket_timeout: float = 0.5
    retry_on_timeout: bool = False


@dataclass(frozen=True)
class IngestionConfig:
    """Async ingestion pipeline parameters."""

    queue_max_size: int = 10_000
    batch_size: int = 32
    worker_count: int = 2
    rate_limit_per_second: float = 100.0
    background_index_interval_seconds: float = 5.0


@dataclass(frozen=True)
class ConnectorConfig:
    """External data connector configuration."""

    enabled: bool = _env_bool("CONNECTORS_ENABLED", False)
    startup_sync: bool = _env_bool("CONNECTORS_STARTUP_SYNC", False)
    market_symbols: tuple[str, ...] = tuple(
        symbol.strip().upper()
        for symbol in os.getenv(
            "CONNECTOR_MARKET_SYMBOLS",
            "AAPL,NVDA,TSLA,MSFT,GOOGL,AMZN,META,JPM,GS,BAC",
        ).split(",")
        if symbol.strip()
    )
    sec_enabled: bool = _env_bool("SEC_CONNECTOR_ENABLED", True)
    sec_poll_interval_seconds: int = _env_int("SEC_POLL_INTERVAL_SECONDS", 1800)
    sec_user_agent: str = os.getenv(
        "SEC_USER_AGENT",
        "FinancialRAG/1.0 contact=admin@example.com",
    )
    sec_cik_map: dict[str, str] = field(default_factory=lambda: _env_json_dict(
        "SEC_CIK_MAP",
        {
            "AAPL": "0000320193",
            "MSFT": "0000789019",
            "NVDA": "0001045810",
            "AMZN": "0001018724",
            "META": "0001326801",
            "GOOGL": "0001652044",
            "TSLA": "0001318605",
            "JPM": "0000019617",
            "GS": "0000886982",
            "BAC": "0000070858",
        },
    ))
    sec_forms: tuple[str, ...] = tuple(
        form.strip().upper()
        for form in os.getenv("SEC_FORMS", "10-K,10-Q,8-K").split(",")
        if form.strip()
    )
    sec_full_text: bool = _env_bool("CONNECTORS_SEC_FULL_TEXT", False)
    finnhub_enabled: bool = _env_bool("FINNHUB_ENABLED", True)
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    finnhub_base_url: str = os.getenv("FINNHUB_BASE_URL", "https://finnhub.io/api/v1")
    finnhub_poll_interval_seconds: int = _env_int("FINNHUB_POLL_INTERVAL_SECONDS", 300)
    alpha_enabled: bool = _env_bool("ALPHA_VANTAGE_ENABLED", True)
    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    alpha_vantage_base_url: str = os.getenv(
        "ALPHA_VANTAGE_BASE_URL",
        "https://www.alphavantage.co/query",
    )
    alpha_poll_interval_seconds: int = _env_int("ALPHA_POLL_INTERVAL_SECONDS", 300)
    request_timeout_seconds: float = _env_float("CONNECTOR_TIMEOUT_SECONDS", 20.0)


@dataclass(frozen=True)
class ObservabilityConfig:
    """Observability and monitoring configuration."""

    prometheus_enabled: bool = _env_bool("PROMETHEUS_ENABLED", True)
    grafana_admin_user: str = os.getenv("GRAFANA_ADMIN_USER", "admin")
    grafana_admin_password: str = os.getenv("GRAFANA_ADMIN_PASSWORD", "admin")


@dataclass
class SystemConfig:
    """Top-level system configuration aggregating all sub-configs."""

    latency: LatencyBudget = field(default_factory=LatencyBudget)
    cache: CacheConfig = field(default_factory=CacheConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    connectors: ConnectorConfig = field(default_factory=ConnectorConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = _env_int("APP_PORT", 8000)
    debug: bool = _env_bool("RAG_DEBUG", False)
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


_config: SystemConfig | None = None


def get_config() -> SystemConfig:
    """Return the global system configuration (lazily created)."""
    global _config
    if _config is None:
        _config = SystemConfig()
    return _config
