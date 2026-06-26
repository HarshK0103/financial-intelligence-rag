"""Measure freshness classification accuracy."""

from __future__ import annotations

import json
import time

from app.consistency.freshness_scorer import FreshnessScorer
from app.models import Document


def freshness_accuracy(
    scorer: FreshnessScorer,
    cases: list[tuple[Document, bool]],
    query_time: float,
) -> float:
    if not cases:
        return 0.0
    correct = sum(
        scorer.is_stale(document, query_time) == expected_stale
        for document, expected_stale in cases
    )
    return correct / len(cases)


def main() -> None:
    now = time.time()
    cases = [
        (Document(doc_id="fresh_price", content="fresh", timestamp=now - 10), False),
        (Document(doc_id="fresh_news", content="fresh", timestamp=now - 120), False),
        (Document(doc_id="stale_price", content="stale", timestamp=now - 301), True),
        (Document(doc_id="stale_filing", content="stale", timestamp=now - 3600), True),
    ]
    accuracy = freshness_accuracy(FreshnessScorer(), cases, now)
    print(json.dumps({"freshness_accuracy": round(accuracy, 6)}, indent=2))


if __name__ == "__main__":
    main()
