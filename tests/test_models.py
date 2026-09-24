from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from signal_brief.models import Candidate, ScoredCandidate, SourceType


def _candidate(**overrides) -> Candidate:
    defaults = dict(
        source=SourceType.ARXIV,
        external_id="2605.21384",
        title="SpecBench",
        summary="A benchmark.",
        url="https://arxiv.org/abs/2605.21384",
        published_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
        authors=["Bingchen Zhao"],
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def test_dedupe_key_combines_source_and_id():
    c = _candidate()
    assert c.dedupe_key == "arxiv:2605.21384"


def test_dedupe_key_distinguishes_sources_with_same_external_id():
    # a GitHub repo literally named "2605.21384" should not collide with the
    # arXiv paper of the same id -- this is exactly why dedupe_key is
    # source-qualified rather than just the raw external_id.
    a = _candidate(source=SourceType.ARXIV, external_id="2605.21384")
    b = _candidate(source=SourceType.GITHUB, external_id="2605.21384")
    assert a.dedupe_key != b.dedupe_key


def test_candidate_requires_published_at():
    with pytest.raises(ValidationError):
        Candidate(
            source=SourceType.ARXIV,
            external_id="x",
            title="x",
            url="https://example.com",
        )


def test_scored_candidate_rejects_out_of_range_score():
    c = _candidate()
    with pytest.raises(ValidationError):
        ScoredCandidate(candidate=c, relevance_score=10.1, rationale="too high")
    with pytest.raises(ValidationError):
        ScoredCandidate(candidate=c, relevance_score=-0.1, rationale="too low")


def test_scored_candidate_accepts_boundary_scores():
    c = _candidate()
    ScoredCandidate(candidate=c, relevance_score=0.0, rationale="min ok")
    ScoredCandidate(candidate=c, relevance_score=10.0, rationale="max ok")
