"""Finnhub news connector."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import httpx

from app.config import get_config
from app.connectors.base import BaseConnector, ConnectorFetchResult
from app.models import DataTemperature, Document

logger = logging.getLogger(__name__)


class NewsConnector(BaseConnector):
    """Pull company news from Finnhub."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        cfg = get_config().connectors
        super().__init__(
            name="finnhub_news",
            poll_interval_seconds=cfg.finnhub_poll_interval_seconds,
            enabled=cfg.enabled and cfg.finnhub_enabled and bool(cfg.finnhub_api_key),
        )
        self._symbols = cfg.market_symbols
        self._token = cfg.finnhub_api_key
        self._client = client or httpx.AsyncClient(
            base_url=cfg.finnhub_base_url,
            timeout=cfg.request_timeout_seconds,
        )

    async def fetch_documents(
        self,
        *,
        since_ts: float | None = None,
    ) -> ConnectorFetchResult:
        if not self.enabled:
            return ConnectorFetchResult(connector_name=self.name)

        documents: list[Document] = []
        date_from, date_to = _build_date_window(since_ts)

        for symbol in self._symbols:
            response = await self._client.get(
                "/company-news",
                params={
                    "symbol": symbol,
                    "from": date_from,
                    "to": date_to,
                    "token": self._token,
                },
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                continue

            for item in payload:
                identifier = item.get("id")
                if identifier is None:
                    continue
                timestamp = float(item.get("datetime", 0))
                if since_ts is not None and timestamp <= since_ts:
                    continue
                headline = str(item.get("headline", "")).strip()
                summary = str(item.get("summary", "")).strip()
                source = str(item.get("source", "finnhub")).strip()
                url = str(item.get("url", "")).strip()
                content = " ".join(part for part in [headline, summary, url] if part).strip()
                if not content:
                    continue
                documents.append(
                    Document(
                        doc_id=f"finnhub_news_{identifier}",
                        content=content,
                        source="finnhub_news",
                        ticker=symbol,
                        timestamp=timestamp,
                        temperature=DataTemperature.HOT,
                        metadata={
                            "provider": "finnhub",
                            "external_id": identifier,
                            "headline": headline,
                            "summary": summary,
                            "news_source": source,
                            "url": url,
                        },
                    )
                )

        logger.info("NewsConnector fetched %d documents", len(documents))
        return ConnectorFetchResult(
            connector_name=self.name,
            documents=documents,
            metadata={"symbols": list(self._symbols)},
        )

    async def close(self) -> None:
        await self._client.aclose()


def _build_date_window(since_ts: float | None) -> tuple[str, str]:
    now = datetime.now(tz=UTC)
    start = datetime.fromtimestamp(since_ts, tz=UTC) if since_ts is not None else now - timedelta(days=2)
    return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")
