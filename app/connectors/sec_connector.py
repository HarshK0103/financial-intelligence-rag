"""SEC EDGAR connector.

Supports two modes:
- **Metadata-only** (default): Ingests filing metadata (date, form type,
  accession number, URL) for fast, lightweight document creation.
- **Full-text** (``CONNECTORS_SEC_FULL_TEXT=true``): Additionally downloads
  the filing HTML from EDGAR, parses it with :mod:`app.connectors.sec_parser`,
  extracts key sections (Risk Factors, MD&A, etc.), and creates chunked
  documents suitable for high-quality retrieval.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx

from app.config import get_config
from app.connectors.base import BaseConnector, ConnectorFetchResult
from app.connectors.sec_parser import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    parse_filing,
)
from app.models import DataTemperature, Document

logger = logging.getLogger(__name__)


class SECConnector(BaseConnector):
    """Pull filing metadata and optionally full document bodies from SEC EDGAR."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        cfg = get_config().connectors
        super().__init__(
            name="sec_edgar",
            poll_interval_seconds=cfg.sec_poll_interval_seconds,
            enabled=cfg.enabled and cfg.sec_enabled,
        )
        self._cik_map = cfg.sec_cik_map
        self._forms = {form.upper() for form in cfg.sec_forms}
        self._full_text: bool = getattr(cfg, "sec_full_text", False)
        self._client = client or httpx.AsyncClient(
            base_url="https://data.sec.gov",
            timeout=cfg.request_timeout_seconds,
            headers={"User-Agent": cfg.sec_user_agent},
        )
        # Separate client for downloading filing HTML from www.sec.gov
        self._www_client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": cfg.sec_user_agent},
        )

    async def fetch_documents(
        self,
        *,
        since_ts: float | None = None,
    ) -> ConnectorFetchResult:
        if not self.enabled or not self._cik_map:
            return ConnectorFetchResult(connector_name=self.name)

        documents: list[Document] = []
        for ticker, cik in self._cik_map.items():
            padded_cik = cik.zfill(10)
            response = await self._client.get(f"/submissions/CIK{padded_cik}.json")
            response.raise_for_status()
            payload = response.json()
            recent = payload.get("filings", {}).get("recent", {})

            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])
            descriptions = recent.get("primaryDocDescription", [])

            for form, filing_date, accession, primary_doc, description in zip(
                forms,
                dates,
                accessions,
                primary_docs,
                descriptions if descriptions else [""] * len(forms),
                strict=False,
            ):
                if str(form).upper() not in self._forms:
                    continue
                timestamp = _parse_date_to_ts(str(filing_date))
                if since_ts is not None and timestamp <= since_ts:
                    continue
                accession_no_dash = str(accession).replace("-", "")
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(padded_cik)}/{accession_no_dash}/{primary_doc}"
                )
                description_text = str(description or "").strip()

                # ── Metadata document (always created) ────────────
                content = (
                    f"{ticker} filed {form} with the SEC on {filing_date}. "
                    f"{description_text or 'Recent filing metadata from EDGAR.'} "
                    f"Primary document: {primary_doc}. Filing URL: {filing_url}"
                ).strip()
                meta_doc = Document(
                    doc_id=f"sec_{ticker}_{accession_no_dash}",
                    content=content,
                    source="sec_edgar",
                    ticker=ticker,
                    timestamp=timestamp,
                    temperature=DataTemperature.COLD,
                    metadata={
                        "provider": "sec",
                        "cik": padded_cik,
                        "form": form,
                        "filing_date": filing_date,
                        "accession": accession,
                        "primary_document": primary_doc,
                        "filing_url": filing_url,
                    },
                )
                documents.append(meta_doc)

                # ── Full-text body parsing (opt-in) ───────────────
                if self._full_text:
                    body_docs = await self._fetch_filing_body(
                        filing_url=filing_url,
                        ticker=ticker,
                        form_type=str(form),
                        accession_no_dash=accession_no_dash,
                        filing_date=str(filing_date),
                        timestamp=timestamp,
                    )
                    documents.extend(body_docs)

        logger.info("SECConnector fetched %d documents", len(documents))
        return ConnectorFetchResult(
            connector_name=self.name,
            documents=documents,
            metadata={"tickers": sorted(self._cik_map), "full_text": self._full_text},
        )

    async def _fetch_filing_body(
        self,
        *,
        filing_url: str,
        ticker: str,
        form_type: str,
        accession_no_dash: str,
        filing_date: str,
        timestamp: float,
    ) -> list[Document]:
        """Download and parse the full filing HTML into chunked documents."""
        try:
            response = await self._www_client.get(filing_url)
            response.raise_for_status()
            html = response.text

            chunks = parse_filing(
                html,
                ticker=ticker,
                form_type=form_type,
                chunk_words=DEFAULT_CHUNK_WORDS,
                overlap_words=DEFAULT_OVERLAP_WORDS,
            )

            documents: list[Document] = []
            for chunk in chunks:
                doc_id = (
                    f"sec_{ticker}_{accession_no_dash}"
                    f"_{chunk['section']}_c{chunk['chunk_index']}"
                )
                documents.append(Document(
                    doc_id=doc_id,
                    content=chunk["content"],
                    source="sec_edgar_body",
                    ticker=ticker,
                    timestamp=timestamp,
                    temperature=DataTemperature.COLD,
                    metadata={
                        "provider": "sec",
                        "form": form_type,
                        "filing_date": filing_date,
                        "section": chunk["section"],
                        "chunk_index": chunk["chunk_index"],
                        "filing_url": filing_url,
                    },
                ))

            logger.info(
                "SECConnector: parsed %s %s body into %d chunks",
                ticker,
                form_type,
                len(documents),
            )
            return documents

        except Exception as exc:
            logger.warning(
                "SECConnector: failed to fetch/parse body for %s %s: %s "
                "(falling back to metadata-only)",
                ticker,
                form_type,
                exc,
            )
            return []

    async def close(self) -> None:
        await self._client.aclose()
        await self._www_client.aclose()


def _parse_date_to_ts(value: str) -> float:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return datetime.now(tz=UTC).timestamp()
