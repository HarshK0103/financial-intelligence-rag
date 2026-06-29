# Latency Breakdown

## Scope

This report profiles the Financial RAG request path across:

1. Query embedding
2. L1 cache
3. L2 semantic cache
4. L3 ticker cache
5. BM25 retrieval
6. Vector retrieval
7. Reranking
8. Freshness scoring
9. Inference
10. Serialization

Architecture preserved:

- Hybrid retrieval kept
- Semantic cache kept
- Freshness scoring kept
- Reranking kept

## Benchmark Setup

- Harness: in-process `fastapi.testclient.TestClient`
- Query set: `evaluation/benchmark_queries.json`
- Workload: 20-request and 30-request loops
- Environment note: sample-data startup loads the embedding model and probes the optional Redis backend

## Before

### User-reported benchmark

- P50: `19ms`
- P95: `563ms`
- P99: `563ms`

### Reproduced baseline before optimization

30-request run:

- P50: `16.641ms`
- P95: `21.682ms`
- P99: `532.691ms`

Observed pattern:

- Warm requests were already fast
- A single cold-path outlier dominated the tail
- First request metrics showed `cache_lookup_ms ~= 512.62ms`

## Root Causes

### Largest latency contributors

1. Cold-path Redis probe in L1 exact cache
   - `ExactCache.get()` attempted Redis connection/ping on the first live request.
   - When Redis was unavailable, the request paid the failure cost before falling back to memory.
   - This was the main reason P99 and the user-reported P95 were above 500ms.

2. Unnecessary query embedding before cheap caches
   - The pipeline embedded every query before any cache short-circuit.
   - Exact-cache and hot-ticker hits still paid embedding latency.

3. Semantic cache lookup allocation overhead
   - L2 semantic cache rebuilt `np.stack(self._embeddings)` on every lookup.
   - This is an avoidable O(N*d) allocation on top of the similarity scan itself.

### Cold path bottlenecks

- Redis fallback probe on first request
- Embedding model/network warm-up during startup
- Early request embedding cost before a potential cache hit

### O(N) operations identified

- L2 semantic cache similarity scan: `O(N*d)` by design
  - Preserved, but per-request matrix rebuild removed
- BM25 result ordering: previously full `O(N log N)` sort of all scores
  - Reduced to top-k selection
- Reranker document tokenization: repeated per-request work over retrieved docs
  - Token list is now reused via lightweight metadata caching
- Cache invalidation by ticker in L1/L2 remains O(N)
  - Not on the request hot path

### Blocking I/O identified

- Redis `ping()` on request path before optimization
- Startup-time Hugging Face/model resolution and local model load
- JSON/file reads during startup sample ingestion

### Unnecessary embedding calls

Before:

- Every request embedded before cache evaluation

After:

- L1 exact and L3 hot-ticker checks happen before embedding
- Query embedding only happens when L1/L3 miss and L2/vector retrieval is needed

## Changes Implemented

### Instrumentation

Added per-request metrics for:

- `query_embedding_ms`
- `l1_cache_ms`
- `l2_cache_ms`
- `l3_cache_ms`
- `bm25_retrieval_ms`
- `vector_retrieval_ms`
- `reranking_ms`
- `freshness_scoring_ms`
- `inference_ms`
- `serialization_ms`

Updated `evaluation/latency_benchmark.py` to emit aggregated stage metrics.

### Optimizations

1. Moved Redis availability resolution to startup
   - `ExactCache.warm()`
   - called from `CacheManager.start()`
   - first request no longer blocks on Redis fallback

2. Reordered cheap cache path
   - L1 exact -> L3 hot ticker -> embed -> L2 semantic -> retrieval
   - avoids unnecessary embeddings on cheap cache hits

3. Cached semantic-cache embedding matrix
   - removed repeated `np.stack(...)` during L2 lookups

4. Reduced BM25 top-k selection cost
   - replaced full sort with `heapq.nlargest(...)`

5. Cached reranker document tokenization
   - avoids repeated tokenization of the same document content across requests

## After

### 20-request run

- P50: `7.775ms`
- P95: `21.129ms`
- P99: `25.758ms`

### 30-request run

- P50: `7.541ms`
- P95: `23.081ms`
- P99: `24.421ms`

### Stage metrics after optimization (30-request run)

- `query_embedding_ms`: avg `1.807ms`, p95 `13.47ms`, max `14.46ms`
- `l1_cache_ms`: avg `0.03ms`, p95 `0.08ms`, max `0.10ms`
- `l2_cache_ms`: avg `0.013ms`, p95 `0.09ms`, max `0.10ms`
- `l3_cache_ms`: avg `0.002ms`, p95 `0.01ms`, max `0.01ms`
- `cache_lookup_ms`: avg `1.864ms`, p95 `13.61ms`, max `14.63ms`
- `bm25_retrieval_ms`: avg `0.126ms`, p95 `0.85ms`, max `1.01ms`
- `vector_retrieval_ms`: avg `0.13ms`, p95 `0.87ms`, max `1.15ms`
- `reranking_ms`: avg `0.094ms`, p95 `0.58ms`, max `0.87ms`
- `freshness_scoring_ms`: avg `0.005ms`, p95 `0.03ms`, max `0.05ms`
- `inference_ms`: avg `0.106ms`, p95 `0.74ms`, max `0.74ms`
- `serialization_ms`: avg `2.636ms`, p95 `3.17ms`, max `3.20ms`

## Outcome

Target:

- `P95 < 200ms`

Result:

- Achieved with substantial headroom

## Remaining Notes

- The model still performs network/model initialization at startup; this no longer affects steady-state request latency but does affect readiness time.
- L2 semantic cache remains linear in cache size by architecture. If the semantic cache grows materially, the next architecture-preserving improvement would be a maintained ANN-style semantic-cache index or shard-by-query-family strategy.
- Serialization is now one of the largest steady-state contributors simply because the rest of the pipeline is very fast.
