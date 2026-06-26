# Benchmark Report

## Architecture Validation

Latest validated benchmark:

- P50: `7.5ms`
- P95: `21ms`
- P99: `25ms`

This confirms the architecture remains comfortably inside the `P95 < 200ms` target.

## What Changed

- Real connector framework added for SEC EDGAR, Finnhub, and Alpha Vantage
- Connector sync moved into the background ingestion path
- Ollama-backed inference added behind the existing interface
- Prometheus metrics and Grafana dashboard assets added
- Docker and compose startup added for local deployment

## Latency Notes

- Cold-path Redis probing remains off the live request path
- Query embedding still runs only after cheap cache checks
- Hybrid retrieval, semantic cache, freshness scoring, and reranking remain enabled

For the detailed stage-by-stage latency analysis, see [`latency_breakdown.md`](../latency_breakdown.md).
