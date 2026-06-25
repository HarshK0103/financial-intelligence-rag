"""
Financial Intelligence RAG System - Main Application Entry Point

FastAPI application with lifecycle management. Initialises all subsystems
on startup and orchestrates the query path, ingestion pipeline, connector
sync, inference backend selection, and observability endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router, set_pipeline
from app.cache.cache_manager import CacheManager
from app.cache.exact_cache import ExactCache
from app.cache.hot_ticker_cache import HotTickerCache, HotTickerEntry
from app.cache.semantic_cache import SemanticCache
from app.config import get_config
from app.connectors import ConnectorSyncService, build_connectors
from app.consistency.cache_invalidator import CacheInvalidator
from app.consistency.freshness_scorer import FreshnessScorer
from app.data.cold_store import ColdStore
from app.data.data_router import DataRouter
from app.data.hot_store import HotStore
from app.embedding.embedding_service import EmbeddingService
from app.inference import InferenceEngine, OllamaInferenceEngine
from app.ingestion.embedding_worker import EmbeddingWorker
from app.ingestion.stream_processor import StreamProcessor
from app.models import (
    CacheLayer,
    Document,
    IngestionEvent,
    QueryRequest,
    QueryResponse,
    QueryType,
    ResponseMetrics,
    ScoredDocument,
)
from app.observability.metrics import CONTENT_TYPE_LATEST, render_prometheus_metrics
from app.resilience.circuit_breaker import CircuitBreaker
from app.resilience.degraded_mode import DegradedMode
from app.resilience.timeout_handler import TimeoutHandler
from app.retrieval.retrieval_engine import RetrievalEngine, RetrievalTimings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("financial_rag")


class QueryPipeline:
    """Orchestrates the full RAG query pipeline."""

    def __init__(
        self,
        *,
        cache_manager: CacheManager,
        retrieval_engine: RetrievalEngine,
        inference_engine,
        circuit_breaker: CircuitBreaker,
        timeout_handler: TimeoutHandler,
        degraded_mode: DegradedMode,
        freshness_scorer: FreshnessScorer,
        data_router: DataRouter,
        embedding_worker: EmbeddingWorker,
        stream_processor: StreamProcessor,
        cache_invalidator: CacheInvalidator,
        hot_ticker_cache: HotTickerCache,
        embedding_service: EmbeddingService,
        connector_sync_service: ConnectorSyncService,
        inference_backend: str,
    ) -> None:
        self.cache_manager = cache_manager
        self.retrieval_engine = retrieval_engine
        self.inference_engine = inference_engine
        self.circuit_breaker = circuit_breaker
        self.timeout_handler = timeout_handler
        self.degraded_mode = degraded_mode
        self.freshness_scorer = freshness_scorer
        self.data_router = data_router
        self.embedding_worker = embedding_worker
        self.stream_processor = stream_processor
        self.cache_invalidator = cache_invalidator
        self.hot_ticker_cache = hot_ticker_cache
        self.embedding_service = embedding_service
        self.connector_sync_service = connector_sync_service
        self.inference_backend = inference_backend
        self.config = get_config()

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query through the full pipeline."""
        metrics = ResponseMetrics()
        query_type = self.inference_engine.classify_query(request.query)
        cache_total_start = time.perf_counter()

        l1_start = time.perf_counter()
        cached_response = await self.cache_manager.l1.get(request.query)
        metrics.l1_cache_ms = round((time.perf_counter() - l1_start) * 1000, 2)
        if cached_response is not None and not request.require_fresh:
            metrics.cache_hit = True
            metrics.cache_layer = CacheLayer.L1_EXACT
            metrics.cache_lookup_ms = round(
                (time.perf_counter() - cache_total_start) * 1000, 2
            )
            cached_response.cache_layer = CacheLayer.L1_EXACT
            cached_response.query_type = query_type
            cached_response.metrics = metrics
            return cached_response

        l3_start = time.perf_counter()
        cached_response = await self.cache_manager.l3.get(request.query)
        metrics.l3_cache_ms = round((time.perf_counter() - l3_start) * 1000, 2)
        if cached_response is not None and not request.require_fresh:
            metrics.cache_hit = True
            metrics.cache_layer = CacheLayer.L3_HOT_TICKER
            metrics.cache_lookup_ms = round(
                (time.perf_counter() - cache_total_start) * 1000, 2
            )
            cached_response.cache_layer = CacheLayer.L3_HOT_TICKER
            cached_response.query_type = query_type
            cached_response.metrics = metrics
            return cached_response

        embedding_start = time.perf_counter()
        query_embedding = await self._generate_query_embedding(request.query)
        metrics.query_embedding_ms = round(
            (time.perf_counter() - embedding_start) * 1000, 2
        )

        l2_start = time.perf_counter()
        cached_response = await self.cache_manager.l2.get(request.query, query_embedding)
        metrics.l2_cache_ms = round((time.perf_counter() - l2_start) * 1000, 2)
        metrics.cache_lookup_ms = round(
            (time.perf_counter() - cache_total_start) * 1000, 2
        )
        if cached_response is not None and not request.require_fresh:
            metrics.cache_hit = True
            metrics.cache_layer = CacheLayer.L2_SEMANTIC
            cached_response.cache_layer = CacheLayer.L2_SEMANTIC
            cached_response.query_type = query_type
            cached_response.metrics = metrics
            return cached_response

        retrieval_start = time.perf_counter()
        retrieval_timings = RetrievalTimings()
        try:
            scored_docs, retrieval_timings = await self.circuit_breaker.call(
                self.timeout_handler.execute_with_timeout,
                coro=self._do_retrieval_with_timings(
                    request.query,
                    query_embedding,
                    request.max_results,
                ),
                timeout_ms=self.config.latency.retrieval_ms,
                stage_name="retrieval",
                fallback_value=([], retrieval_timings),
            )
        except Exception as exc:
            logger.warning("Retrieval failed, using degraded mode: %s", exc)
            return self.degraded_mode.generate_degraded_response(
                request.query,
                None,
                f"Retrieval failed: {exc}",
            )

        metrics.retrieval_ms = round(
            (time.perf_counter() - retrieval_start) * 1000, 2
        )
        metrics.bm25_retrieval_ms = retrieval_timings.bm25_ms
        metrics.vector_retrieval_ms = retrieval_timings.vector_ms
        metrics.reranking_ms = retrieval_timings.reranking_ms
        metrics.documents_retrieved = len(scored_docs)

        freshness_start = time.perf_counter()
        query_time = time.time()
        for sdoc in scored_docs:
            sdoc.freshness_score = self.freshness_scorer.score(
                sdoc.document,
                query_time,
            )
            sdoc.final_score = sdoc.final_score * 0.7 + sdoc.freshness_score * 0.3

        scored_docs.sort(key=lambda doc: doc.final_score, reverse=True)
        scored_docs = scored_docs[: request.max_results]
        metrics.documents_reranked = len(scored_docs)
        metrics.freshness_scoring_ms = round(
            (time.perf_counter() - freshness_start) * 1000, 2
        )
        if scored_docs:
            freshness_vals = [sdoc.freshness_score for sdoc in scored_docs]
            metrics.freshness_min = round(min(freshness_vals), 3)
            metrics.freshness_max = round(max(freshness_vals), 3)

        inference_start = time.perf_counter()
        try:
            answer = await self.timeout_handler.execute_with_timeout(
                coro=self._do_inference(request.query, scored_docs, query_type),
                timeout_ms=self.config.latency.inference_ms,
                stage_name="inference",
                fallback_value=(
                    "I couldn't generate a complete response within the time budget. "
                    "Here are the most relevant sources retrieved."
                ),
            )
        except Exception as exc:
            logger.warning("Inference failed: %s", exc)
            answer = "Inference unavailable. Retrieved documents are listed below."

        metrics.inference_ms = round(
            (time.perf_counter() - inference_start) * 1000, 2
        )
        metrics.cache_layer = CacheLayer.MISS
        metrics.circuit_state = self.circuit_breaker.state

        response = QueryResponse(
            answer=answer,
            sources=scored_docs,
            query_type=query_type,
            cache_layer=CacheLayer.MISS,
            is_degraded=False,
            metrics=metrics,
        )

        asyncio.create_task(
            self.cache_manager.set(
                request.query,
                query_embedding,
                response,
                request.tickers,
            )
        )
        return response

    async def ingest(self, event: IngestionEvent) -> None:
        """Submit an ingestion event for async processing."""
        await self.stream_processor.submit(event)

    async def _generate_query_embedding(self, query: str) -> list[float]:
        return await self.embedding_service.embed_text(query)

    async def _do_retrieval(
        self,
        query: str,
        embedding: list[float],
        top_k: int,
    ) -> list[ScoredDocument]:
        return await self.retrieval_engine.retrieve(query, embedding, top_k)

    async def _do_retrieval_with_timings(
        self,
        query: str,
        embedding: list[float],
        top_k: int,
    ) -> tuple[list[ScoredDocument], RetrievalTimings]:
        return await self.retrieval_engine.retrieve_with_timings(
            query,
            embedding,
            top_k,
        )

    async def _do_inference(
        self,
        query: str,
        docs: list[ScoredDocument],
        query_type: QueryType,
    ) -> str:
        return await self.inference_engine.generate(query, docs, query_type)


async def _load_sample_data(
    data_router: DataRouter,
    embedding_worker: EmbeddingWorker,
) -> list[Document]:
    data_dir = Path(__file__).parent.parent / "data"
    loaded_docs: list[Document] = []

    for filename in ["sample_filings.json", "sample_news.json", "sample_prices.json"]:
        filepath = data_dir / filename
        if not filepath.exists():
            logger.warning("Sample data file not found: %s", filepath)
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as handle:
                raw_docs = json.load(handle)
            docs = [Document(**raw) for raw in raw_docs]
            embedded_docs = await embedding_worker.embed_documents(docs)
            await data_router.add_documents(embedded_docs)
            loaded_docs.extend(embedded_docs)
            logger.info("Loaded %d documents from %s", len(raw_docs), filename)
        except Exception as exc:
            logger.error("Error loading %s: %s", filename, exc)

    return loaded_docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    config = get_config()
    logger.info("=" * 60)
    logger.info("  Financial RAG Platform - Starting up")
    logger.info("=" * 60)

    exact_cache = ExactCache()
    semantic_cache = SemanticCache()
    hot_ticker_cache = HotTickerCache()
    cache_manager = CacheManager(
        exact_cache=exact_cache,
        semantic_cache=semantic_cache,
        hot_ticker_cache=hot_ticker_cache,
    )

    hot_store = HotStore()
    cold_store = ColdStore()
    data_router = DataRouter(hot_store, cold_store)

    embedding_service = EmbeddingService()
    embedding_worker = EmbeddingWorker(embedding_service=embedding_service)

    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.reranker import Reranker
    from app.retrieval.vector_retriever import VectorRetriever

    bm25_retriever = BM25Retriever()
    vector_retriever = VectorRetriever()
    reranker = Reranker()
    retrieval_engine = RetrievalEngine(
        bm25_retriever,
        vector_retriever,
        reranker,
    )

    template_engine = InferenceEngine()
    if config.inference.backend.lower() == "ollama":
        inference_engine = OllamaInferenceEngine(fallback_engine=template_engine)
        inference_backend = "ollama"
    else:
        inference_engine = template_engine
        inference_backend = "template"

    circuit_breaker = CircuitBreaker()
    timeout_handler = TimeoutHandler()
    degraded_mode = DegradedMode()
    freshness_scorer = FreshnessScorer()
    cache_invalidator = CacheInvalidator(cache_manager)

    async def index_documents(docs: list[Document]) -> int:
        await data_router.add_documents(docs)
        await bm25_retriever.add_documents(docs)
        await vector_retriever.add_documents([
            (doc, doc.embedding)
            for doc in docs
            if doc.embedding is not None
        ])
        return len(docs)

    async def invalidate_tickers(tickers: list[str]) -> int:
        record = await cache_invalidator.invalidate(tickers, source="ingestion")
        return record.total_invalidated

    async def refresh_hot_ticker(ticker: str) -> HotTickerEntry:
        docs = await data_router.get_documents_both(ticker)
        scored_docs = [
            ScoredDocument(
                document=doc,
                freshness_score=freshness_scorer.score(doc, time.time()),
                final_score=freshness_scorer.score(doc, time.time()),
            )
            for doc in docs
        ]
        response = QueryResponse(
            answer=await inference_engine.generate(
                f"What is the latest information for {ticker}?",
                scored_docs,
                QueryType.GENERAL,
            ),
            sources=scored_docs,
            query_type=QueryType.GENERAL,
            cache_layer=CacheLayer.L3_HOT_TICKER,
        )
        return HotTickerEntry(
            ticker=ticker,
            documents=scored_docs,
            precomputed_response=response,
        )

    cache_manager.register_hot_ticker_callback(refresh_hot_ticker)

    stream_processor = StreamProcessor(
        embedding_worker=embedding_worker,
        on_index=index_documents,
        on_cache_invalidate=invalidate_tickers,
    )

    logger.info("Loading sample data...")
    all_docs = await _load_sample_data(data_router, embedding_worker)
    logger.info("Loaded %d total documents", len(all_docs))

    if all_docs:
        await bm25_retriever.add_documents(all_docs)
        await vector_retriever.add_documents([
            (doc, doc.embedding)
            for doc in all_docs
            if doc.embedding is not None
        ])
        logger.info(
            "Built retrieval indexes: BM25=%d docs, Vector=%d docs",
            bm25_retriever.corpus_size,
            vector_retriever.index_size,
        )
    await hot_ticker_cache.refresh()

    connectors = build_connectors()
    connector_sync_service = ConnectorSyncService(
        connectors,
        submit_event=lambda event: stream_processor.submit(event),
        startup_sync=config.connectors.startup_sync,
    )

    pipeline = QueryPipeline(
        cache_manager=cache_manager,
        retrieval_engine=retrieval_engine,
        inference_engine=inference_engine,
        circuit_breaker=circuit_breaker,
        timeout_handler=timeout_handler,
        degraded_mode=degraded_mode,
        freshness_scorer=freshness_scorer,
        data_router=data_router,
        embedding_worker=embedding_worker,
        stream_processor=stream_processor,
        cache_invalidator=cache_invalidator,
        hot_ticker_cache=hot_ticker_cache,
        embedding_service=embedding_service,
        connector_sync_service=connector_sync_service,
        inference_backend=inference_backend,
    )
    set_pipeline(pipeline)

    await cache_manager.start()
    stream_processor.start_processing()
    await connector_sync_service.start()
    logger.info("Background services started")

    logger.info("=" * 60)
    logger.info("  Financial RAG Platform - Ready")
    logger.info("  Dashboard: http://localhost:%d", config.port)
    logger.info("  API: http://localhost:%d/api/query", config.port)
    logger.info("=" * 60)

    try:
        yield
    finally:
        logger.info("Shutting down...")
        await connector_sync_service.stop()
        if hasattr(inference_engine, "close"):
            await inference_engine.close()
        await stream_processor.stop()
        await cache_manager.stop()
        logger.info("Financial RAG Platform - Stopped")


app = FastAPI(
    title="Financial RAG Intelligence Platform",
    description="Financial intelligence platform with hybrid retrieval, real data connectors, and Ollama-backed inference.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the monitoring dashboard."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "service": "Financial RAG Intelligence Platform",
        "version": "2.0.0",
        "docs": "/docs",
        "dashboard": "/static/index.html",
    }


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(
        content=render_prometheus_metrics(),
        media_type=CONTENT_TYPE_LATEST,
    )
