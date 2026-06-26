"""Compute retrieval quality metrics from labeled benchmark queries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
        if doc_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_rankings(
    benchmark: list[dict], rankings: dict[str, list[str]]
) -> dict[str, float]:
    rows = []
    for item in benchmark:
        ranked_ids = rankings.get(item["id"], [])
        relevant_ids = set(item["relevant_doc_ids"])
        rows.append({
            "recall@5": recall_at_k(ranked_ids, relevant_ids, 5),
            "recall@10": recall_at_k(ranked_ids, relevant_ids, 10),
            "mrr": reciprocal_rank(ranked_ids, relevant_ids),
            "ndcg@10": ndcg_at_k(ranked_ids, relevant_ids, 10),
        })
    return {
        key: round(sum(row[key] for row in rows) / len(rows), 6)
        for key in rows[0]
    } if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(__file__).with_name("benchmark_queries.json"),
    )
    parser.add_argument("--rankings", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    rankings = json.loads(args.rankings.read_text(encoding="utf-8"))
    print(json.dumps(evaluate_rankings(benchmark, rankings), indent=2))


if __name__ == "__main__":
    main()
