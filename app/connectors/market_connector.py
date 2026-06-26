"""Alpha Vantage market connector."""

from __future__ import annotations

import logging
import time

import httpx

from app.config import get_config
from app.connectors.base import BaseConnector, ConnectorFetchResult
from app.models import DataTemperature, Document

logger = logging.getLogger(__name__)


class MarketConnector(BaseConnector):
    """Pull current market quotes from Alpha Vantage."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        cfg = get_config().connectors
        super().__init__(
            name="alpha_vantage_market",
            poll_interval_seconds=cfg.alpha_poll_interval_seconds,
            enabled=cfg.enabled and cfg.alpha_enabled and bool(cfg.alpha_vantage_api_key),
        )
        self._symbols = cfg.market_symbols
        self._api_key = cfg.alpha_vantage_api_key
        self._base_url = cfg.alpha_vantage_base_url
        self._client = client or httpx.AsyncClient(timeout=cfg.request_timeout_seconds)

    async def fetch_documents(
        self,
        *,
        since_ts: float | None = None,
    ) -> ConnectorFetchResult:
        if not self.enabled:
            return ConnectorFetchResult(connector_name=self.name)

        now_ts = time.time()
        if since_ts is not None and (now_ts - since_ts) < max(self.poll_interval_seconds - 5, 0):
            return ConnectorFetchResult(connector_name=self.name)

        documents: list[Document] = []
        for symbol in self._symbols:
            response = await self._client.get(
                self._base_url,
                params={
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self._api_key,
                },
            )
            response.raise_for_status()
            payload = response.json()
            quote = payload.get("Global Quote", {})
            if not quote:
                continue
            price = quote.get("05. price", "")
            change = quote.get("09. change", "")
            change_pct = quote.get("10. change percent", "")
            volume = quote.get("06. volume", "")
            last_trading_day = quote.get("07. latest trading day", "")
            content = (
                f"{symbol} last traded at {price}. Change {change} ({change_pct}). "
                f"Volume {volume}. Latest trading day {last_trading_day}."
            )
            documents.append(Document(
                doc_id=f"alpha_quote_{symbol}_{int(now_ts // max(self.poll_interval_seconds, 1))}",
                content=content,
                source="alpha_vantage_market",
                ticker=symbol,
                timestamp=now_ts,
                temperature=DataTemperature.HOT,
                metadata={
                    "provider": "alpha_vantage",
                    "price": price,
                    "change": change,
                    "change_percent": change_pct,
                    "volume": volume,
                    "latest_trading_day": last_trading_day,
                },
            ))

        logger.info("MarketConnector fetched %d documents", len(documents))
        return ConnectorFetchResult(
            connector_name=self.name,
            documents=documents,
            metadata={"symbols": list(self._symbols)},
        )

    async def close(self) -> None:
        await self._client.aclose()
