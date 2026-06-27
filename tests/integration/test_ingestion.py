import time


def test_ingestion_updates_store_and_retrieval(client) -> None:
    before = client.get("/api/health").json()["hot_store_docs"]
    payload = {
        "source": "test",
        "documents": [
            {
                "doc_id": "news_amd_integration",
                "content": "AMD accelerator revenue growth reached 75 percent.",
                "source": "news",
                "ticker": "AMD",
                "temperature": "hot",
            }
        ],
    }

    response = client.post("/api/ingest", json=payload)
    assert response.status_code == 200

    for _ in range(100):
        time.sleep(0.02)
        if client.get("/api/health").json()["hot_store_docs"] > before:
            break

    query = client.post(
        "/api/query",
        json={
            "query": "What is AMD accelerator revenue growth?",
            "tickers": ["AMD"],
            "max_results": 20,
        },
    )

    assert client.get("/api/health").json()["hot_store_docs"] > before
    assert query.status_code == 200
    assert any(
        source["document"]["doc_id"] == "news_amd_integration"
        for source in query.json()["sources"]
    )
