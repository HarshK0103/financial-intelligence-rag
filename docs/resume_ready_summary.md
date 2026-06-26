# Resume-Ready Project Summary

Built a low-latency Financial Intelligence Platform that combines hybrid RAG retrieval, multi-layer caching, real-time market/news ingestion, SEC filing ingestion, and local LLM inference.

Highlights:

- Designed a sub-200ms financial RAG architecture and validated it at `P95 21ms`
- Implemented hybrid BM25 + FAISS retrieval with semantic caching and freshness-aware reranking
- Added real data connector architecture for SEC EDGAR, Finnhub, and Alpha Vantage
- Integrated Ollama with `llama3.1:8b` behind a production-safe inference interface
- Added Dockerized local deployment with Redis, Ollama, Prometheus, and Grafana
- Exposed platform metrics for query latency, retrieval latency, cache hit rate, freshness, and inference timing
