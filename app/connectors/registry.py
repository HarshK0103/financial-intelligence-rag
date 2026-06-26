"""Connector factory helpers."""

from __future__ import annotations

from app.config import get_config
from app.connectors.base import BaseConnector
from app.connectors.market_connector import MarketConnector
from app.connectors.news_connector import NewsConnector
from app.connectors.sec_connector import SECConnector


def build_connectors() -> list[BaseConnector]:
    """Build enabled connectors from configuration."""
    cfg = get_config().connectors
    if not cfg.enabled:
        return []

    connectors: list[BaseConnector] = [
        SECConnector(),
        NewsConnector(),
        MarketConnector(),
    ]
    return [connector for connector in connectors if connector.enabled]
