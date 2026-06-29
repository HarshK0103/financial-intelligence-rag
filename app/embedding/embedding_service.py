"""Shared sentence-transformers embedding service."""

from __future__ import annotations

import asyncio
from functools import partial

from sentence_transformers import SentenceTransformer

from app.config import get_config


class EmbeddingService:
    """Generate normalized embeddings from one shared model instance."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or get_config().embedding_model
        self._model: SentenceTransformer | None = None
        self._model_lock = asyncio.Lock()

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await self._get_model()
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(
            None,
            partial(
                model.encode,
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )
        return [vector.tolist() for vector in vectors]

    async def _get_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is None:
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(None, SentenceTransformer, self._model_name)
        return self._model
