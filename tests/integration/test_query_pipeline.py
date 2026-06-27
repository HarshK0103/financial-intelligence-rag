import time


def test_query_pipeline_and_cache_hit(client) -> None:
    payload = {"query": "What was revenue in Q2 fiscal 2025?", "max_results": 5}

    first = client.post("/api/query", json=payload)
    time.sleep(0.05)
    second = client.post("/api/query", json=payload)

    assert first.status_code == 200
    assert first.json()["sources"]
    assert first.json()["is_degraded"] is False
    assert second.status_code == 200
    assert second.json()["cache_layer"] == "l1_exact"
