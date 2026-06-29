"""Tests for the template-based inference engine."""

import pytest

from app.inference.inference_engine import InferenceEngine
from app.models import Document, QueryType, ScoredDocument


@pytest.fixture
def engine() -> InferenceEngine:
    return InferenceEngine(max_output_tokens=200)


def _scored_doc(content: str, ticker: str = "AAPL") -> ScoredDocument:
    return ScoredDocument(
        document=Document(
            doc_id="test",
            content=content,
            source="test",
            ticker=ticker,
        ),
    )


# ── Query classification ──────────────────────────────────────────


class TestClassifyQuery:
    def test_price_keywords(self, engine: InferenceEngine) -> None:
        assert engine.classify_query("What is the stock price of AAPL?") == QueryType.PRICE

    def test_earnings_keywords(self, engine: InferenceEngine) -> None:
        assert engine.classify_query("What was NVDA revenue in Q2 2025?") == QueryType.EARNINGS

    def test_news_keywords(self, engine: InferenceEngine) -> None:
        assert engine.classify_query("What is the latest news on Tesla?") == QueryType.NEWS

    def test_comparison_keywords(self, engine: InferenceEngine) -> None:
        assert engine.classify_query("Compare AAPL vs GOOGL market cap") == QueryType.COMPARISON

    def test_analysis_keywords(self, engine: InferenceEngine) -> None:
        assert engine.classify_query("What is the analyst outlook for MSFT?") == QueryType.ANALYSIS

    def test_general_fallback(self, engine: InferenceEngine) -> None:
        assert engine.classify_query("Tell me about blue widgets") == QueryType.GENERAL


# ── Generate ──────────────────────────────────────────────────────


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generates_non_empty_output(self, engine: InferenceEngine) -> None:
        docs = [_scored_doc("AAPL revenue was $94.8 billion in Q3 2024.")]
        result = await engine.generate("What was AAPL revenue?", docs, QueryType.EARNINGS)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_empty_docs_returns_no_documents_message(self, engine: InferenceEngine) -> None:
        result = await engine.generate("random query", [], QueryType.GENERAL)
        assert "no relevant documents" in result.lower()

    @pytest.mark.asyncio
    async def test_price_template_includes_disclaimer(self, engine: InferenceEngine) -> None:
        docs = [_scored_doc("AAPL last traded at $195.50")]
        result = await engine.generate("AAPL stock price", docs, QueryType.PRICE)
        assert "verify" in result.lower() or "delayed" in result.lower()

    @pytest.mark.asyncio
    async def test_each_query_type_produces_output(self, engine: InferenceEngine) -> None:
        docs = [_scored_doc("Sample financial data for testing")]
        for qt in QueryType:
            result = await engine.generate("test query", docs, qt)
            assert len(result) > 0, f"Empty output for {qt.value}"


# ── Budget enforcement ────────────────────────────────────────────


class TestBudgetEnforcement:
    @pytest.mark.asyncio
    async def test_long_output_is_truncated(self) -> None:
        engine = InferenceEngine(max_output_tokens=20)
        max_chars = 20 * 4  # _CHARS_PER_TOKEN = 4
        docs = [_scored_doc("x " * 500)]
        result = await engine.generate("test", docs, QueryType.GENERAL)
        assert len(result) <= max_chars + 10  # small buffer for " …"

    @pytest.mark.asyncio
    async def test_short_output_not_truncated(self, engine: InferenceEngine) -> None:
        docs = [_scored_doc("Short.")]
        result = await engine.generate("test", docs, QueryType.GENERAL)
        assert "…" not in result


# ── Prompt compression ────────────────────────────────────────────


class TestCompression:
    def test_compress_strips_whitespace(self, engine: InferenceEngine) -> None:
        text = "hello\n\n\n\n\nworld    spacing"
        result = engine._compress(text)
        assert "\n\n\n" not in result
        assert "    " not in result
