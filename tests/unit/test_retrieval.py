import pytest

from app.models import Document
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker
from app.retrieval.retrieval_engine import RetrievalEngine
from app.retrieval.vector_retriever import VectorRetriever


@pytest.mark.asyncio
async def test_hybrid_retrieval_returns_indexed_document() -> None:
    embedding = [1.0] + [0.0] * 383
    document = Document(
        doc_id="filing_test",
        content="AAPL revenue increased to 10 billion dollars.",
        ticker="AAPL",
        embedding=embedding,
    )
    bm25 = BM25Retriever()
    vector = VectorRetriever()

    await bm25.add_documents([document])
    await vector.add_documents([(document, embedding)])
    results = await RetrievalEngine(bm25, vector, Reranker()).retrieve(
        "AAPL revenue", embedding, top_k=5
    )

    assert results
    assert results[0].document.doc_id == "filing_test"
