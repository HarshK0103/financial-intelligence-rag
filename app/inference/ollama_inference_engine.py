"""Ollama-backed inference engine."""

from __future__ import annotations

import logging

import httpx

from app.config import get_config
from app.inference.inference_engine import InferenceEngine
from app.models import QueryType, ScoredDocument

logger = logging.getLogger(__name__)


class OllamaInferenceEngine:
    """Drop-in inference backend using Ollama's chat API."""

    def __init__(
        self,
        *,
        fallback_engine: InferenceEngine | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        cfg = get_config()
        self._ollama_cfg = cfg.ollama
        self._fallback = fallback_engine or InferenceEngine()
        self._temperature = cfg.inference.temperature
        self._enabled = self._ollama_cfg.enabled
        self._client = client or httpx.AsyncClient(
            base_url=self._ollama_cfg.base_url.rstrip("/"),
            timeout=self._ollama_cfg.request_timeout_seconds,
        )

        logger.info(
            "OllamaInferenceEngine initialised model=%s enabled=%s",
            self._ollama_cfg.model,
            self._enabled,
        )
        self._last_healthy: bool = False

    # ------------------------------------------------------------------
    # Health probe
    # ------------------------------------------------------------------

    async def health_check(self) -> dict:
        """Probe Ollama availability via ``GET /api/tags``.

        Returns a dict suitable for embedding in ``/api/health``.
        """
        if not self._enabled:
            return {
                "ollama_status": "disabled",
                "ollama_model": self._ollama_cfg.model,
            }
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            self._last_healthy = True
            return {
                "ollama_status": "connected",
                "ollama_model": self._ollama_cfg.model,
            }
        except Exception as exc:
            logger.debug("Ollama health check failed: %s", exc)
            self._last_healthy = False
            return {
                "ollama_status": "fallback",
                "ollama_model": self._ollama_cfg.model,
            }

    def classify_query(self, query: str) -> QueryType:
        """Reuse the current fast local classifier."""
        return self._fallback.classify_query(query)

    async def generate(
        self,
        query: str,
        context_docs: list[ScoredDocument],
        query_type: QueryType | None = None,
    ) -> str:
        """Generate an answer from Ollama, with fallback on failure."""
        if not self._enabled:
            return await self._fallback.generate(query, context_docs, query_type)

        if query_type is None:
            query_type = self.classify_query(query)

        prompt = self._build_prompt(query, context_docs, query_type)
        payload = {
            "model": self._ollama_cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a financial intelligence assistant. Answer using only "
                        "the provided context. If the context is insufficient, say so."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": self._ollama_cfg.keep_alive,
            "options": {
                "temperature": self._temperature,
            },
        }

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "") if isinstance(data, dict) else ""
            if content and content.strip():
                return content.strip()
            logger.warning("Ollama returned empty content, using fallback engine")
        except Exception as exc:
            logger.warning("Ollama inference failed, using fallback engine: %s", exc)

        return await self._fallback.generate(query, context_docs, query_type)

    async def close(self) -> None:
        """Release HTTP resources."""
        await self._client.aclose()

    def _build_prompt(
        self,
        query: str,
        context_docs: list[ScoredDocument],
        query_type: QueryType,
    ) -> str:
        context = self._fallback._build_context(context_docs)
        if get_config().inference.prompt_compression_enabled:
            context = self._fallback._compress(context)
        return (
            f"Query type: {query_type.value}\n"
            f"User query: {query}\n\n"
            f"Context documents:\n{context}\n\n"
            "Respond with a concise analyst-style answer and mention uncertainty "
            "when the context is incomplete."
        )
