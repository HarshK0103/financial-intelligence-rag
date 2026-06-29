"""Background connector sync orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import deque

from app.connectors.base import BaseConnector
from app.models import IngestionEvent
from app.observability.metrics import observe_connector_sync

logger = logging.getLogger(__name__)


class ConnectorSyncService:
    """Poll connectors and enqueue normalized documents for ingestion."""

    def __init__(
        self,
        connectors: list[BaseConnector],
        submit_event,
        *,
        startup_sync: bool = True,
        seen_capacity: int = 50_000,
    ) -> None:
        self._connectors = connectors
        self._submit_event = submit_event
        self._startup_sync = startup_sync
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._last_sync_ts: dict[str, float] = {}
        self._last_error: dict[str, str] = {}
        self._documents_synced: dict[str, int] = {connector.name: 0 for connector in connectors}
        self._seen_capacity = seen_capacity
        self._seen_ids: set[str] = set()
        self._seen_queue: deque[str] = deque()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for connector in self._connectors:
            task = asyncio.create_task(
                self._run_connector_loop(connector),
                name=f"connector-sync-{connector.name}",
            )
            self._tasks.append(task)

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        for connector in self._connectors:
            await connector.close()

    async def sync_once(self, connector_name: str | None = None) -> dict[str, int]:
        """Trigger one immediate sync for one or all connectors."""
        counts: dict[str, int] = {}
        targets = [
            connector for connector in self._connectors if connector_name is None or connector.name == connector_name
        ]
        for connector in targets:
            counts[connector.name] = await self._sync_connector(connector)
        return counts

    def status_snapshot(self) -> dict[str, dict[str, object]]:
        """Return runtime status for health endpoints."""
        return {
            connector.name: {
                "enabled": connector.enabled,
                "poll_interval_seconds": connector.poll_interval_seconds,
                "last_sync_ts": self._last_sync_ts.get(connector.name),
                "last_error": self._last_error.get(connector.name),
                "documents_synced": self._documents_synced.get(connector.name, 0),
            }
            for connector in self._connectors
        }

    async def _run_connector_loop(self, connector: BaseConnector) -> None:
        if self._startup_sync:
            await self._sync_connector(connector)

        while self._running:
            try:
                await asyncio.sleep(connector.poll_interval_seconds)
                await self._sync_connector(connector)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error[connector.name] = str(exc)
                logger.exception("Connector sync failed for %s", connector.name)
                await asyncio.sleep(5)

    async def _sync_connector(self, connector: BaseConnector) -> int:
        started = time.perf_counter()
        since_ts = self._last_sync_ts.get(connector.name)
        try:
            result = await connector.fetch_documents(since_ts=since_ts)
            filtered_docs = [doc for doc in result.documents if self._remember_doc(doc.doc_id)]
            if filtered_docs:
                event = IngestionEvent(
                    event_id=_build_event_id(connector.name, filtered_docs),
                    documents=filtered_docs,
                    source=connector.name,
                )
                await self._submit_event(event)

            now_ts = time.time()
            self._last_sync_ts[connector.name] = now_ts
            self._documents_synced[connector.name] = self._documents_synced.get(connector.name, 0) + len(filtered_docs)
            self._last_error.pop(connector.name, None)
            duration_seconds = time.perf_counter() - started
            observe_connector_sync(
                connector.name,
                duration_seconds=duration_seconds,
                document_count=len(filtered_docs),
                error=False,
            )
            logger.info(
                "Connector %s synced %d documents",
                connector.name,
                len(filtered_docs),
            )
            return len(filtered_docs)
        except Exception:
            duration_seconds = time.perf_counter() - started
            observe_connector_sync(
                connector.name,
                duration_seconds=duration_seconds,
                document_count=0,
                error=True,
            )
            raise

    def _remember_doc(self, doc_id: str) -> bool:
        if doc_id in self._seen_ids:
            return False
        self._seen_ids.add(doc_id)
        self._seen_queue.append(doc_id)
        while len(self._seen_queue) > self._seen_capacity:
            evicted = self._seen_queue.popleft()
            self._seen_ids.discard(evicted)
        return True


def _build_event_id(connector_name: str, documents) -> str:
    seed = f"{connector_name}:{documents[0].doc_id}:{len(documents)}:{time.time()}"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]
