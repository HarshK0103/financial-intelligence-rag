def test_startup_serves_dashboard(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Financial Intelligence RAG" in response.text


def test_health_endpoint_reports_loaded_stores(client) -> None:
    response = client.get("/api/health")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "healthy"
    assert data["hot_store_docs"] >= 26
    assert data["cold_store_docs"] >= 15
    assert len(data["active_tickers"]) == 14
