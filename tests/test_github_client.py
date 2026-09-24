from __future__ import annotations

import httpx
import pytest

from signal_brief.clients.github_client import GitHubClient
from signal_brief.models import SourceType


def test_build_query_without_pushed_after():
    assert GitHubClient.build_query(["reward-hacking", "GRPO"]) == "reward-hacking GRPO"


def test_build_query_with_pushed_after():
    q = GitHubClient.build_query(["GRPO"], pushed_after="2026-06-01")
    assert q == "GRPO pushed:>2026-06-01"


@pytest.mark.live
def test_live_search_against_real_github_api():
    """Hits the real api.github.com. No mocking -- this is deliberately a
    live integration test, matching the standard of validating against
    real data rather than a synthetic fixture wherever the environment
    allows it (arXiv's API blocks non-browser fetches from this sandbox;
    GitHub's does not, so there's no excuse not to test it for real)."""
    with GitHubClient() as client:
        candidates = client.search(["reward-hacking", "GRPO"], max_results=5)

    assert 1 <= len(candidates) <= 5
    for c in candidates:
        assert c.source is SourceType.GITHUB
        assert "/" in c.external_id  # owner/repo
        assert c.url.startswith("https://github.com/")
        assert c.title == c.external_id


def test_parses_response_with_missing_description(monkeypatch):
    """Real repos frequently have a null description -- make sure that
    doesn't crash the parser (caught this by reading actual API output)."""

    def fake_get(self, url, params=None):
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "items": [
                    {
                        "full_name": "someone/no-description-repo",
                        "description": None,
                        "html_url": "https://github.com/someone/no-description-repo",
                        "pushed_at": "2026-07-01T12:00:00Z",
                        "owner": {"login": "someone"},
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    with GitHubClient() as client:
        candidates = client.search(["anything"])
    assert len(candidates) == 1
    assert candidates[0].summary == ""


def test_parses_response_with_missing_owner(monkeypatch):
    """Some search results (e.g. orgs mid-transfer) can omit `owner` entirely."""

    def fake_get(self, url, params=None):
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "items": [
                    {
                        "full_name": "orphaned/repo",
                        "description": "no owner field at all",
                        "html_url": "https://github.com/orphaned/repo",
                        "pushed_at": "2026-07-01T12:00:00Z",
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    with GitHubClient() as client:
        candidates = client.search(["anything"])
    assert candidates[0].authors == []
