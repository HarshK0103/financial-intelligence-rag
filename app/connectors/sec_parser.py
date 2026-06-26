"""SEC EDGAR filing body parser.

Downloads and parses the full HTML body of SEC filings (10-K, 10-Q, 8-K)
from EDGAR, extracts key sections, and chunks them into retrieval-ready
segments.

Uses only the standard library ``html.parser`` — no external dependencies.
"""

from __future__ import annotations

import logging
import re
import textwrap
from html.parser import HTMLParser
from typing import Final

logger = logging.getLogger(__name__)

# ── Configurable defaults ─────────────────────────────────────────

DEFAULT_CHUNK_WORDS: Final[int] = 700
DEFAULT_OVERLAP_WORDS: Final[int] = 70
MAX_FILING_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB safety limit

# ── Section detection patterns ────────────────────────────────────
# SEC filings have standard sections identified by "Item N" headers.

_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("risk_factors", re.compile(
        r"item\s+1a[\.\:\s\u2014\u2013\-]+risk\s+factors",
        re.IGNORECASE,
    )),
    ("mda", re.compile(
        r"item\s+7[\.\:\s\u2014\u2013\-]+"
        r"management.{0,10}s?\s+discussion\s+and\s+analysis",
        re.IGNORECASE,
    )),
    ("financial_statements", re.compile(
        r"item\s+8[\.\:\s\u2014\u2013\-]+financial\s+statements",
        re.IGNORECASE,
    )),
    ("business", re.compile(
        r"item\s+1[\.\:\s\u2014\u2013\-]+business(?!\s+address)",
        re.IGNORECASE,
    )),
    ("legal_proceedings", re.compile(
        r"item\s+3[\.\:\s\u2014\u2013\-]+legal\s+proceedings",
        re.IGNORECASE,
    )),
    ("revenue_recognition", re.compile(
        r"revenue\s+recognition",
        re.IGNORECASE,
    )),
    ("segment_information", re.compile(
        r"segment\s+(?:information|reporting|results)",
        re.IGNORECASE,
    )),
]


# ── HTML → text converter ────────────────────────────────────────


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-plain-text converter for SEC filings.

    Strips all tags but preserves paragraph and heading breaks as
    newlines.  Ignores ``<script>``, ``<style>``, and ``<ix:…>``
    (iXBRL) content entirely.
    """

    _BLOCK_TAGS: Final[frozenset[str]] = frozenset({
        "p", "div", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "td", "th", "table", "section", "article",
    })
    _SKIP_TAGS: Final[frozenset[str]] = frozenset({
        "script", "style", "head", "meta", "link",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower().split(":")[-1]  # strip namespace
        if lower_tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif lower_tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower().split(":")[-1]
        if lower_tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif lower_tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of whitespace while keeping paragraph breaks
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    """Convert an SEC filing HTML body to clean plain text."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# ── Section extraction ────────────────────────────────────────────


def extract_sections(text: str) -> dict[str, str]:
    """Extract named sections from a plain-text SEC filing.

    Returns a dict mapping section names (e.g. ``"risk_factors"``,
    ``"mda"``) to their text content.  If no sections are detected,
    returns ``{"full_text": text}`` so the caller always gets content.
    """
    # Find all section start positions
    found: list[tuple[str, int]] = []
    for name, pattern in _SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            found.append((name, match.start()))

    if not found:
        return {"full_text": text}

    # Sort by position and extract text between consecutive headers
    found.sort(key=lambda x: x[1])
    sections: dict[str, str] = {}

    for i, (name, start) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(text)
        section_text = text[start:end].strip()
        # Remove the header line itself
        first_newline = section_text.find("\n")
        if first_newline > 0:
            section_text = section_text[first_newline:].strip()
        if len(section_text) > 50:  # Skip trivially empty sections
            sections[name] = section_text

    return sections or {"full_text": text}


# ── Chunking ──────────────────────────────────────────────────────


def chunk_text(
    text: str,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """Split *text* into overlapping word-based chunks.

    Parameters
    ----------
    text
        The plain text to chunk.
    chunk_words
        Target number of words per chunk.
    overlap_words
        Number of overlapping words between consecutive chunks.

    Returns
    -------
    list[str]
        Chunks of approximately *chunk_words* words each.
    """
    words = text.split()
    if len(words) <= chunk_words:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    step = max(1, chunk_words - overlap_words)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_words])
        if chunk.strip():
            chunks.append(chunk.strip())
        if i + chunk_words >= len(words):
            break

    return chunks


# ── High-level API ────────────────────────────────────────────────


def parse_filing(
    html: str,
    *,
    ticker: str,
    form_type: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict[str, str]]:
    """Parse an SEC filing HTML into chunked, section-tagged segments.

    Parameters
    ----------
    html
        Raw HTML of the SEC filing.
    ticker
        The ticker symbol (e.g. ``"AAPL"``).
    form_type
        The SEC form type (e.g. ``"10-K"``).
    chunk_words
        Words per chunk.
    overlap_words
        Overlapping words between chunks.

    Returns
    -------
    list[dict[str, str]]
        Each dict has keys ``"content"``, ``"section"``, ``"chunk_index"``.
    """
    if len(html) > MAX_FILING_BYTES:
        logger.warning(
            "Filing for %s exceeds %d bytes, truncating",
            ticker,
            MAX_FILING_BYTES,
        )
        html = html[:MAX_FILING_BYTES]

    text = html_to_text(html)
    if not text:
        logger.warning("Empty text after parsing filing for %s", ticker)
        return []

    sections = extract_sections(text)
    result: list[dict[str, str]] = []

    for section_name, section_text in sections.items():
        chunks = chunk_text(
            section_text,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
        )
        for idx, chunk in enumerate(chunks):
            prefix = (
                f"[{ticker} {form_type} — {section_name.replace('_', ' ').title()}] "
            )
            result.append({
                "content": prefix + chunk,
                "section": section_name,
                "chunk_index": str(idx),
            })

    logger.info(
        "Parsed %s %s filing: %d sections, %d chunks",
        ticker,
        form_type,
        len(sections),
        len(result),
    )
    return result
