"""Tests for the SEC filing HTML parser and chunker."""

from app.connectors.sec_parser import (
    chunk_text,
    extract_sections,
    html_to_text,
    parse_filing,
)

# ── Sample HTML fragments ─────────────────────────────────────────

SIMPLE_HTML = """
<html><body>
<h1>APPLE INC.</h1>
<p>10-K Annual Report</p>
<p>Item 1A. Risk Factors</p>
<p>The Company faces significant competition in all areas.
   Apple competes with companies that have more resources.
   The market for smartphones is highly competitive.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Revenue increased 8% year over year to $94.8 billion.
   Services revenue grew 14% driven by App Store and iCloud.
   Gross margin expanded to 46.2%.</p>
</body></html>
"""

IXBRL_HTML = """
<html><body>
<ix:header></ix:header>
<div>
  <p>Financial data in iXBRL format</p>
  <script>var x = 1;</script>
  <style>.hidden { display: none; }</style>
  <p>Revenue was $50 billion.</p>
</div>
</body></html>
"""

EMPTY_HTML = "<html><body></body></html>"


# ══════════════════════════════════════════════════════════════════
# html_to_text
# ══════════════════════════════════════════════════════════════════


def test_html_to_text_extracts_content() -> None:
    text = html_to_text(SIMPLE_HTML)
    assert "APPLE INC." in text
    assert "Risk Factors" in text
    assert "Revenue increased" in text


def test_html_to_text_strips_scripts() -> None:
    text = html_to_text(IXBRL_HTML)
    assert "var x" not in text
    assert ".hidden" not in text
    assert "Revenue was $50 billion" in text


def test_html_to_text_empty() -> None:
    text = html_to_text(EMPTY_HTML)
    assert text == "" or text.strip() == ""


def test_html_to_text_preserves_paragraph_breaks() -> None:
    html = "<p>First paragraph</p><p>Second paragraph</p>"
    text = html_to_text(html)
    assert "\n" in text


# ══════════════════════════════════════════════════════════════════
# extract_sections
# ══════════════════════════════════════════════════════════════════


def test_extract_sections_finds_risk_factors() -> None:
    text = html_to_text(SIMPLE_HTML)
    sections = extract_sections(text)
    assert "risk_factors" in sections


def test_extract_sections_finds_mda() -> None:
    text = html_to_text(SIMPLE_HTML)
    sections = extract_sections(text)
    assert "mda" in sections


def test_extract_sections_no_headers_returns_full_text() -> None:
    sections = extract_sections("Just a plain document with no section headers.")
    assert "full_text" in sections


# ══════════════════════════════════════════════════════════════════
# chunk_text
# ══════════════════════════════════════════════════════════════════


def test_chunk_text_short_text_single_chunk() -> None:
    text = "This is a short document."
    chunks = chunk_text(text, chunk_words=100, overlap_words=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_creates_multiple_chunks() -> None:
    words = ["word"] * 200
    text = " ".join(words)
    chunks = chunk_text(text, chunk_words=50, overlap_words=10)
    assert len(chunks) >= 4


def test_chunk_text_overlap() -> None:
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_words=30, overlap_words=10)
    # Consecutive chunks should share some words
    assert len(chunks) >= 2
    first_words = set(chunks[0].split())
    second_words = set(chunks[1].split())
    overlap = first_words & second_words
    assert len(overlap) > 0


def test_chunk_text_empty() -> None:
    assert chunk_text("", chunk_words=50, overlap_words=10) == []


# ══════════════════════════════════════════════════════════════════
# parse_filing (integration)
# ══════════════════════════════════════════════════════════════════


def test_parse_filing_produces_chunks() -> None:
    chunks = parse_filing(SIMPLE_HTML, ticker="AAPL", form_type="10-K")
    assert len(chunks) > 0
    for chunk in chunks:
        assert "content" in chunk
        assert "section" in chunk
        assert "chunk_index" in chunk


def test_parse_filing_content_has_ticker_prefix() -> None:
    chunks = parse_filing(SIMPLE_HTML, ticker="AAPL", form_type="10-K")
    assert any("[AAPL 10-K" in c["content"] for c in chunks)


def test_parse_filing_empty_html() -> None:
    chunks = parse_filing(EMPTY_HTML, ticker="TEST", form_type="10-Q")
    assert chunks == []


def test_parse_filing_respects_size_limit() -> None:
    """Very large HTML is truncated before parsing."""
    huge_html = "<html><body><p>" + ("x " * 5_000_000) + "</p></body></html>"
    # Should not raise — truncated to MAX_FILING_BYTES
    chunks = parse_filing(huge_html, ticker="BIG", form_type="10-K")
    # May produce chunks or may be empty depending on truncation point
    assert isinstance(chunks, list)
