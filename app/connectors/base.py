"""Connector base types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.models import Document


@dataclass
class ConnectorFetchResult:
    """Documents fetched from an external source."""

    connector_name: str
    documents: list[Document] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseConnector(ABC):
    """Abstract base class for all data connectors."""

    def __init__(self, *, name: str, poll_interval_seconds: int, enabled: bool) -> None:
        self.name = name
        self.poll_interval_seconds = poll_interval_seconds
        self.enabled = enabled

    @abstractmethod
    async def fetch_documents(
        self,
        *,
        since_ts: float | None = None,
    ) -> ConnectorFetchResult:
        """Fetch and normalize upstream data into internal documents."""

    async def close(self) -> None:
        """Release network resources when needed."""

