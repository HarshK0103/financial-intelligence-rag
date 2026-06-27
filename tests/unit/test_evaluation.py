from evaluation.latency_benchmark import summarize_latencies
from evaluation.retrieval_metrics import evaluate_rankings


def test_retrieval_metrics() -> None:
    benchmark = [{"id": "q1", "relevant_doc_ids": ["d1"]}]
    metrics = evaluate_rankings(benchmark, {"q1": ["d2", "d1"]})

    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr"] == 0.5
    assert 0.0 < metrics["ndcg@10"] < 1.0


def test_latency_summary() -> None:
    metrics = summarize_latencies([1.0, 2.0, 3.0, 4.0, 5.0])

    assert metrics == {"p50_ms": 3.0, "p95_ms": 5.0, "p99_ms": 5.0}
