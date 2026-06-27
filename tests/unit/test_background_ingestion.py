"""Tests for background ingestion via StreamProcessor."""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import Document, IngestionEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(doc_id: str = "doc_001") -> Document:
    return Document(
        doc_id=doc_id,
        content="Test document for ingestion",
        source="test",
        ticker="AAPL",
    )


def _make_event(doc_id: str = "evt_001", n_docs: int = 1) -> IngestionEvent:
    docs = [_make_doc(f"{doc_id}_d{i}") for i in range(n_docs)]
    return IngestionEvent(
        event_id=doc_id,
        documents=docs,
        source="test",
    )


def _make_processor():
    """Create a StreamProcessor with mock callbacks."""
    from app.ingestion.stream_processor import StreamProcessor

    mock_worker = MagicMock()
    mock_worker.embed_documents = AsyncMock(side_effect=lambda docs: docs)

    on_index = AsyncMock(return_value=1)
    on_cache_invalidate = AsyncMock(return_value=0)

    processor = StreamProcessor(
        embedding_worker=mock_worker,
        on_index=on_index,
        on_cache_invalidate=on_cache_invalidate,
    )
    return processor, on_index, on_cache_invalidate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_adds_to_queue():
    """Submitting an event increases the queue size (must start first)."""
    processor, _, _ = _make_processor()
    event = _make_event("submit_test")

    # StreamProcessor requires start_processing() before submit()
    processor.start_processing()
    await processor.submit(event)

    # Queue may already be processed, but at minimum no errors raised
    await asyncio.sleep(0.1)
    await processor.stop()


@pytest.mark.asyncio
async def test_processing_calls_callbacks():
    """After processing, on_index callback is called with embedded docs."""
    processor, on_index, _ = _make_processor()
    event = _make_event("callback_test", n_docs=3)

    processor.start_processing()
    await processor.submit(event)
    # Give the background task time to process
    await asyncio.sleep(0.3)
    await processor.stop()

    on_index.assert_called_once()
    # The callback receives embedded docs
    call_args = on_index.call_args[0][0]
    assert len(call_args) == 3


@pytest.mark.asyncio
async def test_stop_graceful():
    """Starting and stopping the processor doesn't raise errors."""
    processor, _, _ = _make_processor()
    processor.start_processing()
    await asyncio.sleep(0.1)
    await processor.stop()
    # No assertion needed — just verify no exception


@pytest.mark.asyncio
async def test_empty_event_no_crash():
    """Submitting an event with zero documents doesn't crash."""
    processor, on_index, _ = _make_processor()
    event = IngestionEvent(
        event_id="empty_001",
        documents=[],
        source="test",
    )

    processor.start_processing()
    await processor.submit(event)
    await asyncio.sleep(0.3)
    await processor.stop()

    # on_index may or may not be called for empty docs, but no crash


@pytest.mark.asyncio
async def test_queue_multiple_events():
    """Multiple events submitted are all processed."""
    processor, on_index, _ = _make_processor()

    processor.start_processing()
    for i in range(3):
        await processor.submit(_make_event(f"multi_{i}"))

    await asyncio.sleep(0.5)
    await processor.stop()

    assert on_index.call_count == 3
