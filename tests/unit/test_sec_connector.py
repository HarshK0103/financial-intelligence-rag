"""Tests for SEC connector metadata and full-text modes."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.connectors.base import ConnectorFetchResult
from app.models import Document

# Sample SEC EDGAR API response
SAMPLE_SEC_RESPONSE = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K"],
            "filingDate": ["2025-01-15", "2024-10-20", "2024-09-01"],
            "accessionNumber": [
                "0001-25-000001",
                "0001-24-000002",
                "0001-24-000003",
            ],
            "primaryDocument": ["filing.htm", "filing.htm", "report.htm"],
            "primaryDocDescription": [
                "Annual Report",
                "Quarterly Report",
                "Current Report",
            ],
        }
    }
}


def _make_config(full_text: bool = False, enabled: bool = True):
    cfg = MagicMock()
    cfg.connectors.enabled = enabled
    cfg.connectors.sec_enabled = True
    cfg.connectors.sec_poll_interval_seconds = 1800
    cfg.connectors.sec_cik_map = {"AAPL": "320193"}
    cfg.connectors.sec_forms = ["10-K", "10-Q", "8-K"]
    cfg.connectors.sec_full_text = full_text
    cfg.connectors.request_timeout_seconds = 20
    cfg.connectors.sec_user_agent = "Test/1.0 contact=test@example.com"
    return cfg


def _mock_sec_response():
    """Mock httpx.Response for SEC EDGAR API."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = SAMPLE_SEC_RESPONSE
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("app.connectors.sec_connector.get_config")
async def test_metadata_only_mode(mock_config):
    """Default mode creates metadata-only documents from SEC API."""
    mock_config.return_value = _make_config(full_text=False)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=_mock_sec_response())

    from app.connectors.sec_connector import SECConnector

    connector = SECConnector(client=mock_client)
    result = await connector.fetch_documents()

    assert isinstance(result, ConnectorFetchResult)
    assert len(result.documents) == 3  # 3 forms
    assert all(doc.source == "sec_edgar" for doc in result.documents)
    assert all(doc.ticker == "AAPL" for doc in result.documents)


@pytest.mark.asyncio
@patch("app.connectors.sec_connector.get_config")
async def test_since_ts_filters_old_filings(mock_config):
    """Providing since_ts filters out filings older than the timestamp."""
    mock_config.return_value = _make_config(full_text=False)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=_mock_sec_response())

    from app.connectors.sec_connector import SECConnector
    from datetime import datetime, UTC

    # Set since_ts to after the oldest filing but before the newest
    since_ts = datetime(2024, 10, 1, tzinfo=UTC).timestamp()

    connector = SECConnector(client=mock_client)
    result = await connector.fetch_documents(since_ts=since_ts)

    # Should only include filings after 2024-10-01: 2025-01-15, 2024-10-20
    assert len(result.documents) == 2


@pytest.mark.asyncio
@patch("app.connectors.sec_connector.get_config")
async def test_disabled_connector_returns_empty(mock_config):
    """Disabled connector returns empty result."""
    mock_config.return_value = _make_config(enabled=False)

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    from app.connectors.sec_connector import SECConnector

    connector = SECConnector(client=mock_client)
    connector.enabled = False
    result = await connector.fetch_documents()

    assert len(result.documents) == 0


@pytest.mark.asyncio
@patch("app.connectors.sec_connector.get_config")
async def test_timestamp_preserved_on_metadata_docs(mock_config):
    """Metadata documents carry the correct filing timestamp."""
    mock_config.return_value = _make_config(full_text=False)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=_mock_sec_response())

    from app.connectors.sec_connector import SECConnector

    connector = SECConnector(client=mock_client)
    result = await connector.fetch_documents()

    # First doc is 2025-01-15 10-K
    doc = result.documents[0]
    assert doc.timestamp is not None
    assert doc.timestamp > 0
    assert doc.metadata["filing_date"] == "2025-01-15"


@pytest.mark.asyncio
@patch("app.connectors.sec_connector.get_config")
async def test_full_text_mode_creates_chunks(mock_config):
    """Full-text mode downloads and parses HTML, creating chunked documents."""
    mock_config.return_value = _make_config(full_text=True)

    # Mock SEC API response
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=_mock_sec_response())

    # Mock the filing body HTML download
    body_response = MagicMock()
    body_response.raise_for_status = MagicMock()
    body_response.text = (
        "<html><body>"
        "<h2>Item 1A. Risk Factors</h2>"
        "<p>" + "Risk factor content word. " * 200 + "</p>"
        "<h2>Item 7. Management's Discussion and Analysis</h2>"
        "<p>" + "MDA content word. " * 200 + "</p>"
        "</body></html>"
    )

    from app.connectors.sec_connector import SECConnector

    connector = SECConnector(client=mock_client)
    # Mock the www client for body download
    connector._www_client = AsyncMock(spec=httpx.AsyncClient)
    connector._www_client.get = AsyncMock(return_value=body_response)

    result = await connector.fetch_documents()

    # Should have metadata docs + body chunk docs
    assert len(result.documents) > 3
    body_docs = [d for d in result.documents if d.source == "sec_edgar_body"]
    assert len(body_docs) > 0
    # Verify timestamps on chunks
    for doc in body_docs:
        assert doc.timestamp is not None
        assert doc.timestamp > 0


@pytest.mark.asyncio
@patch("app.connectors.sec_connector.get_config")
async def test_body_download_failure_falls_back(mock_config):
    """If body download fails, only metadata doc is returned."""
    mock_config.return_value = _make_config(full_text=True)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=_mock_sec_response())

    from app.connectors.sec_connector import SECConnector

    connector = SECConnector(client=mock_client)
    connector._www_client = AsyncMock(spec=httpx.AsyncClient)
    connector._www_client.get = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock()
        )
    )

    result = await connector.fetch_documents()

    # Should still have metadata docs (3), just no body chunks
    assert len(result.documents) == 3
    assert all(d.source == "sec_edgar" for d in result.documents)
