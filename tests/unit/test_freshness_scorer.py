"""Tests for the freshness scoring module."""

import time

import pytest

from app.consistency.freshness_scorer import FreshnessScorer
from app.models import Document, ScoredDocument


def _make_doc(age_seconds: float) -> Document:
    """Create a Document with a timestamp *age_seconds* in the past."""
    return Document(
        doc_id=f"test_{age_seconds}",
        content="test content",
        source="test",
        timestamp=time.time() - age_seconds,
    )


@pytest.fixture
def scorer() -> FreshnessScorer:
    return FreshnessScorer(
        halflife_seconds=60.0,
        stale_threshold_seconds=300.0,
    )


# ── Basic scoring ─────────────────────────────────────────────────


def test_fresh_document_scores_near_one(scorer: FreshnessScorer) -> None:
    doc = _make_doc(0.0)
    score = scorer.score(doc, time.time())
    assert score >= 0.99


def test_document_at_halflife_scores_near_half(scorer: FreshnessScorer) -> None:
    doc = _make_doc(60.0)  # exactly one halflife
    score = scorer.score(doc, time.time())
    assert abs(score - 0.5) < 0.05


def test_old_document_near_stale_threshold(scorer: FreshnessScorer) -> None:
    doc = _make_doc(280.0)  # close to 300s stale threshold
    score = scorer.score(doc, time.time())
    assert score < 0.1


def test_document_beyond_stale_threshold_scores_zero(
    scorer: FreshnessScorer,
) -> None:
    doc = _make_doc(300.0)
    score = scorer.score(doc, time.time())
    assert score == 0.0

    doc_very_old = _make_doc(10000.0)
    score_old = scorer.score(doc_very_old, time.time())
    assert score_old == 0.0


# ── Edge cases ────────────────────────────────────────────────────


def test_future_document_scores_one(scorer: FreshnessScorer) -> None:
    """Documents with timestamps in the future are treated as perfectly fresh."""
    doc = Document(
        doc_id="future",
        content="test",
        source="test",
        timestamp=time.time() + 100.0,
    )
    score = scorer.score(doc, time.time())
    assert score == 1.0


def test_score_always_in_valid_range(scorer: FreshnessScorer) -> None:
    for age in [0, 1, 10, 30, 60, 120, 240, 300, 600]:
        doc = _make_doc(float(age))
        score = scorer.score(doc, time.time())
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for age={age}"


def test_score_monotonically_decreases(scorer: FreshnessScorer) -> None:
    """Older documents should always have lower scores."""
    now = time.time()
    ages = [0, 10, 30, 60, 120, 240]
    scores = [scorer.score(_make_doc(a), now) for a in ages]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Score at age {ages[i]} ({scores[i]}) should be >= " f"score at age {ages[i+1]} ({scores[i+1]})"
        )


# ── Apply method ──────────────────────────────────────────────────


def test_apply_populates_freshness_scores(scorer: FreshnessScorer) -> None:
    docs = [
        ScoredDocument(document=_make_doc(0.0)),
        ScoredDocument(document=_make_doc(120.0)),
    ]
    result = scorer.apply(docs, time.time())
    assert len(result) == 2
    assert result[0].freshness_score > result[1].freshness_score
    assert result[0].freshness_score > 0.9
    assert result[1].freshness_score < 0.5
