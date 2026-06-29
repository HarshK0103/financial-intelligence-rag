"""Tests for OllamaInferenceEngine."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(enabled: bool = True):
    cfg = MagicMock()
    cfg.ollama.enabled = enabled
    cfg.ollama.model = "llama3:latest"
    cfg.ollama.base_url = "http://localhost:11434"
    cfg.ollama.request_timeout_seconds = 60.0
    cfg.ollama.keep_alive = "10m"
    cfg.inference.temperature = 0.1
    cfg.inference.max_output_tokens = 300
    cfg.inference.prompt_compression_enabled = False
    cfg.inference.template_mode_enabled = True
    return cfg


def _build_engine(enabled: bool = True, mock_client=None):
    """Create an OllamaInferenceEngine with mocked config and client."""
    with (
        patch("app.inference.ollama_inference_engine.get_config") as mock_cfg,
        patch("app.inference.inference_engine.get_config") as mock_cfg2,
    ):
        cfg = _make_config(enabled)
        mock_cfg.return_value = cfg
        mock_cfg2.return_value = cfg

        from app.inference.ollama_inference_engine import OllamaInferenceEngine

        client = mock_client or AsyncMock(spec=httpx.AsyncClient)
        engine = OllamaInferenceEngine(client=client)
        return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_disabled_uses_fallback():
    """When Ollama is disabled, generate() delegates to the template engine."""
    engine = _build_engine(enabled=False)
    result = await engine.generate("What is AAPL trading at?", [])
    # Should return template output (not empty, not error)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_generate_ollama_success():
    """A valid Ollama chat response is returned as the answer."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"message": {"content": "AAPL is trading at $195.50 based on latest data."}}
    mock_client.post = AsyncMock(return_value=mock_response)

    engine = _build_engine(enabled=True, mock_client=mock_client)
    result = await engine.generate("What is AAPL price?", [])

    assert "195.50" in result
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_generate_ollama_failure_falls_back():
    """When Ollama HTTP call fails, the template fallback is used."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    engine = _build_engine(enabled=True, mock_client=mock_client)
    result = await engine.generate("What is AAPL price?", [])

    # Should still get a valid response from template fallback
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_generate_ollama_empty_response_falls_back():
    """When Ollama returns empty content, the template fallback is used."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"message": {"content": ""}}
    mock_client.post = AsyncMock(return_value=mock_response)

    engine = _build_engine(enabled=True, mock_client=mock_client)
    result = await engine.generate("What is AAPL price?", [])

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_classify_query_delegates_to_fallback():
    """classify_query delegates to the template engine's classifier."""
    engine = _build_engine(enabled=True)
    from app.models import QueryType

    result = engine.classify_query("What is AAPL stock price?")
    assert isinstance(result, QueryType)
    assert result == QueryType.PRICE


@pytest.mark.asyncio
async def test_health_check_connected():
    """health_check returns 'connected' when Ollama responds to /api/tags."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    engine = _build_engine(enabled=True, mock_client=mock_client)
    result = await engine.health_check()

    assert result["ollama_status"] == "connected"
    assert result["ollama_model"] == "llama3:latest"


@pytest.mark.asyncio
async def test_health_check_fallback_on_error():
    """health_check returns 'fallback' when Ollama is unreachable."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    engine = _build_engine(enabled=True, mock_client=mock_client)
    result = await engine.health_check()

    assert result["ollama_status"] == "fallback"
    assert result["ollama_model"] == "llama3:latest"
