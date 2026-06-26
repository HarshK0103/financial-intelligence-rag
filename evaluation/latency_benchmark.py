"""Measure query endpoint P50, P95, and P99 latency."""

from __future__ import annotations

import argparse
import json
import math
import numbers
import time
from pathlib import Path
from urllib.request import Request, urlopen


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((p / 100) * len(ordered)) - 1)
    return ordered[index]


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(percentile(latencies_ms, 50), 3),
        "p95_ms": round(percentile(latencies_ms, 95), 3),
        "p99_ms": round(percentile(latencies_ms, 99), 3),
    }


def summarize_stage_metrics(stage_samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not stage_samples:
        return {}

    keys = sorted({
        key
        for sample in stage_samples
        for key, value in sample.items()
        if isinstance(value, numbers.Real) and not isinstance(value, bool)
    })
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(sample.get(key, 0.0)) for sample in stage_samples]
        summary[key] = {
            "avg_ms": round(sum(values) / len(values), 3),
            "p95_ms": round(percentile(values, 95), 3),
            "max_ms": round(max(values), 3),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(__file__).with_name("benchmark_queries.json"),
    )
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))

    latencies = []
    stage_samples: list[dict[str, float]] = []
    for index in range(args.iterations):
        item = benchmark[index % len(benchmark)]
        body = json.dumps({
            "query": item["query"],
            "tickers": item.get("tickers", []),
            "max_results": 10,
        }).encode("utf-8")
        request = Request(
            f"{args.base_url}/api/query",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        latencies.append((time.perf_counter() - start) * 1000)
        stage_samples.append(payload.get("metrics", {}))

    print(json.dumps({
        **summarize_latencies(latencies),
        "stage_metrics": summarize_stage_metrics(stage_samples),
    }, indent=2))


if __name__ == "__main__":
    main()
