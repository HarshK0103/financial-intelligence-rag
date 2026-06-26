# API Documentation

## Core Endpoints

### `POST /api/query`

Submit a financial intelligence query.

Example:

```json
{
  "query": "Summarize the latest filings and news for NVDA",
  "tickers": ["NVDA"],
  "max_results": 5,
  "require_fresh": false
}
```

### `POST /api/ingest`

Manually enqueue documents into the ingestion pipeline.

### `POST /api/invalidate-cache`

Invalidate cache entries by ticker symbols.

### `GET /api/health`

Returns:

- latency percentiles
- cache hit rates
- document counts
- connector status
- inference backend
- ingestion queue size

### `GET /api/metrics`

Returns JSON application metrics for dashboards and debugging.

### `GET /metrics`

Prometheus exposition endpoint.

### `GET /api/connectors/status`

Returns per-connector runtime status.

### `POST /api/connectors/sync`

Trigger one immediate connector sync.

Optional query/body parameter:

- `connector_name`

## Query Response Metrics

Responses include:

- `query_embedding_ms`
- `l1_cache_ms`
- `l2_cache_ms`
- `l3_cache_ms`
- `retrieval_ms`
- `bm25_retrieval_ms`
- `vector_retrieval_ms`
- `reranking_ms`
- `freshness_scoring_ms`
- `inference_ms`
- `serialization_ms`
