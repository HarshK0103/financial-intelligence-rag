# Architecture Diagram

The system now operates as a financial intelligence platform rather than a benchmark-only MVP.

```mermaid
flowchart TB
    subgraph Clients
        API["REST API"]
        UI["Dashboard"]
    end

    subgraph App["Financial RAG Platform"]
        Query["Query Pipeline"]
        Sync["Connector Sync Service"]
        Ingest["Async Stream Processor"]
        Infer["Ollama / Template Fallback"]
        Metrics["Prometheus /metrics"]
    end

    subgraph Connectors
        SEC["SEC EDGAR"]
        Finnhub["Finnhub News"]
        Alpha["Alpha Vantage Quotes"]
    end

    subgraph Storage
        Hot["Hot Store"]
        Cold["Cold Store"]
        Cache["L1 / L2 / L3 Cache"]
        Redis["Redis"]
        Index["BM25 + FAISS"]
    end

    subgraph Ops
        Prom["Prometheus"]
        Graf["Grafana"]
        Ollama["Ollama llama3.1:8b"]
    end

    API --> Query
    UI --> Query
    Query --> Cache
    Query --> Index
    Query --> Infer
    Infer --> Ollama
    Sync --> Ingest
    SEC --> Sync
    Finnhub --> Sync
    Alpha --> Sync
    Ingest --> Hot
    Ingest --> Cold
    Ingest --> Index
    Ingest --> Cache
    Cache --> Redis
    Metrics --> Prom
    Prom --> Graf
```

## Integration Approach

- Connectors normalize provider data into internal `Document` objects.
- `ConnectorSyncService` polls providers and emits `IngestionEvent`s.
- `StreamProcessor` remains the single ingestion path for embedding, indexing, and cache invalidation.
- Query processing still uses hybrid retrieval, semantic caching, freshness scoring, and reranking.
- Ollama replaces template-only generation while preserving the existing inference interface and fallback behavior.
