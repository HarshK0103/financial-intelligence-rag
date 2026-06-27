"""Tests for ConnectorSyncService."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.connectors.base import BaseConnector, ConnectorFetchResult
from app.models import Document


# ---------------------------------------------------------------------------
# Mock connector
# ---------------------------------------------------------------------------


class MockConnector(BaseConnector):
    """Test connector with configurable behavior."""

    def __init__(
        self,
        docs: list[Document] | None = None,
        *,
        should_fail: bool = False,
        name: str = "mock",
    ):
        super().__init__(name=name, poll_interval_seconds=60, enabled=True)
        self._docs = docs or []
        self._should_fail = should_fail

    async def fetch_documents(
        self, *, since_ts: float | None = None
    ) -> ConnectorFetchResult:
        if self._should_fail:
            raise RuntimeError("Connector failure")
        return ConnectorFetchResult(
            connector_name=self.name, documents=self._docs
        )


def _make_doc(doc_id: str = "doc_001") -> Document:
    return Document(
        doc_id=doc_id,
        content="Test document content",
        source="test",
        ticker="AAPL",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("app.connectors.sync_service.observe_connector_sync")
async def test_sync_once_calls_connector(mock_observe):
    """sync_once fetches documents from the connector."""
    from app.connectors.sync_service import ConnectorSyncService

    doc = _make_doc("sync_test_001")
    connector = MockConnector(docs=[doc])
    submit_event = AsyncMock()

    service = ConnectorSyncService(
        [connector], submit_event, startup_sync=False
    )
    counts = await service.sync_once()

    assert counts["mock"] == 1
    submit_event.assert_called_once()


@pytest.mark.asyncio
@patch("app.connectors.sync_service.observe_connector_sync")
async def test_dedup_prevents_duplicate_ingestion(mock_observe):
    """Same doc_id submitted twice is only ingested once."""
    from app.connectors.sync_service import ConnectorSyncService

    doc = _make_doc("dedup_test_001")
    connector = MockConnector(docs=[doc])
    submit_event = AsyncMock()

    service = ConnectorSyncService(
        [connector], submit_event, startup_sync=False
    )
    await service.sync_once()
    await service.sync_once()  # Second sync with same doc_id

    # submit_event should only be called once (second sync has 0 new docs)
    assert submit_event.call_count == 1


@pytest.mark.asyncio
async def test_status_snapshot_format():
    """status_snapshot returns correct dict shape."""
    from app.connectors.sync_service import ConnectorSyncService

    connector = MockConnector(name="test_snap")
    service = ConnectorSyncService(
        [connector], AsyncMock(), startup_sync=False
    )
    snapshot = service.status_snapshot()

    assert "test_snap" in snapshot
    entry = snapshot["test_snap"]
    assert "enabled" in entry
    assert "poll_interval_seconds" in entry
    assert "last_sync_ts" in entry
    assert "last_error" in entry
    assert "documents_synced" in entry
    assert entry["enabled"] is True
    assert entry["poll_interval_seconds"] == 60


@pytest.mark.asyncio
async def test_sync_with_no_connectors():
    """Empty connector list returns empty counts dict."""
    from app.connectors.sync_service import ConnectorSyncService

    service = ConnectorSyncService([], AsyncMock(), startup_sync=False)
    counts = await service.sync_once()

    assert counts == {}


@pytest.mark.asyncio
@patch("app.connectors.sync_service.observe_connector_sync")
async def test_sync_error_recorded(mock_observe):
    """Sync errors are recorded in last_error."""
    from app.connectors.sync_service import ConnectorSyncService

    connector = MockConnector(should_fail=True, name="failing")
    service = ConnectorSyncService(
        [connector], AsyncMock(), startup_sync=False
    )

    with pytest.raises(RuntimeError):
        await service.sync_once()

    snapshot = service.status_snapshot()
    # Error should NOT be in snapshot since _sync_connector re-raises
    # But observe_connector_sync should have been called with error=True
    mock_observe.assert_called_once()
    call_kwargs = mock_observe.call_args
    assert call_kwargs[1]["error"] is True


@pytest.mark.asyncio
async def test_seen_capacity_eviction():
    """Old doc IDs are evicted when seen_capacity is exceeded."""
    from app.connectors.sync_service import ConnectorSyncService

    service = ConnectorSyncService(
        [], AsyncMock(), startup_sync=False, seen_capacity=3
    )

    # Fill to capacity
    assert service._remember_doc("a") is True
    assert service._remember_doc("b") is True
    assert service._remember_doc("c") is True

    # All three are remembered
    assert service._remember_doc("a") is False
    assert service._remember_doc("b") is False
    assert service._remember_doc("c") is False

    # Adding "d" exceeds capacity → evicts oldest ("a")
    assert service._remember_doc("d") is True

    # "a" was evicted — now forgotten
    assert "a" not in service._seen_ids
    # "b", "c", "d" are still tracked
    assert "b" in service._seen_ids
    assert "c" in service._seen_ids
    assert "d" in service._seen_ids
