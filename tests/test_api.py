from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SIGNAL_USE_FAKE_LLM", "1")

from signal_brief import api as api_module  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    # api.py resolves DATA_DIR as a module-level global; each endpoint looks
    # it up at call time (not captured in a closure), so monkeypatching the
    # module attribute is enough to fully isolate each test's SQLite state
    # without touching the source or relying on process-wide env vars.
    monkeypatch.setattr(api_module, "DATA_DIR", tmp_path)
    monkeypatch.setenv("SIGNAL_USE_FAKE_LLM", "1")
    return TestClient(api_module.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_arc_status_roundtrip(client):
    r = client.post(
        "/arc-status",
        json={"project": "Assay", "status": "GPU run not yet executed", "next_action": "Run GRPO training"},
    )
    assert r.status_code == 200

    r = client.get("/arc-status")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["project"] == "Assay"


def test_brief_with_no_candidates(client):
    r = client.post("/brief", json={"candidates": [], "calendar_events": [], "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert "Nothing new cleared the relevance bar today" in body["brief_text"]
    assert body["top_items"] == []


def test_brief_includes_previously_set_arc_status(client):
    client.post(
        "/arc-status",
        json={"project": "Assay", "status": "GPU run not yet executed", "next_action": "Run GRPO training"},
    )
    r = client.post("/brief", json={"candidates": [], "top_k": 5})
    body = r.json()
    assert body["arc_status"][0]["project"] == "Assay"
    assert "Assay" in body["brief_text"]


def test_brief_with_real_candidate_shape(client):
    payload = {
        "candidates": [
            {
                "source": "arxiv",
                "external_id": "2605.21384",
                "title": "SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents",
                "summary": "Reward hacking benchmark for coding agents using held-out tests.",
                "url": "https://arxiv.org/abs/2605.21384",
                "published_at": "2026-05-20T00:00:00Z",
                "authors": ["Bingchen Zhao"],
            }
        ],
        "top_k": 5,
    }
    r = client.post("/brief", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["top_items"]) == 1
    assert body["top_items"][0]["candidate"]["external_id"] == "2605.21384"
    assert body["top_items"][0]["digest_summary"] is not None


def test_brief_rejects_malformed_candidate(client):
    r = client.post("/brief", json={"candidates": [{"source": "arxiv"}]})  # missing required fields
    assert r.status_code == 422


def test_second_brief_call_dedupes_against_first(client):
    payload = {
        "candidates": [
            {
                "source": "arxiv",
                "external_id": "2605.21384",
                "title": "SpecBench",
                "summary": "x",
                "url": "https://arxiv.org/abs/2605.21384",
                "published_at": "2026-05-20T00:00:00Z",
            }
        ],
        "top_k": 5,
    }
    r1 = client.post("/brief", json=payload)
    assert len(r1.json()["top_items"]) == 1

    r2 = client.post("/brief", json=payload)
    assert len(r2.json()["top_items"]) == 0
    assert "Nothing new" in r2.json()["brief_text"]


def test_invalid_top_k_type_returns_422(client):
    r = client.post("/brief", json={"candidates": [], "top_k": "not-a-number"})
    assert r.status_code == 422


@pytest.mark.live
def test_brief_auto_uses_defaults_when_body_is_empty(client):
    """/brief/auto is meant to be callable with just {} from n8n's HTTP
    Request node -- every field must have a sane default. Hits real
    arXiv (expected to fail 403 from this sandbox, caught gracefully)
    and real GitHub (expected to succeed)."""
    r = client.post("/brief/auto", json={})
    assert r.status_code == 200
    body = r.json()
    assert "brief_text" in body


@pytest.mark.live
def test_brief_auto_includes_calendar_events_passed_through(client):
    r = client.post(
        "/brief/auto",
        json={
            "calendar_events": [{"title": "Deep work block", "start": "2026-07-14T09:00:00Z"}],
            "max_results_per_source": 1,
        },
    )
    assert r.status_code == 200
    assert "Deep work block" in r.json()["brief_text"]


def test_brief_auto_survives_arxiv_being_unreachable(client, monkeypatch):
    """This sandbox's own network returns 403 from export.arxiv.org, which
    is exactly the failure mode this test simulates deliberately (rather
    than relying on environment flakiness): /brief/auto must degrade to
    GitHub-only instead of 500ing. GitHub is also mocked here so this test
    is fully deterministic and doesn't depend on network at all -- the
    live-network version of this path is covered by the two tests above."""
    import signal_brief.api as api_module

    class ExplodingArxivClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def search(self, *a, **kw):
            raise ConnectionError("simulated 403 from export.arxiv.org")

    class EmptyGitHubClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def search(self, *a, **kw):
            return []

    class EmptyHuggingFaceClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def search(self, *a, **kw):
            return []

    monkeypatch.setattr(api_module, "ArxivClient", lambda: ExplodingArxivClient())
    monkeypatch.setattr(api_module, "GitHubClient", lambda token=None: EmptyGitHubClient())
    monkeypatch.setattr(api_module, "HuggingFaceClient", lambda: EmptyHuggingFaceClient())
    r = client.post("/brief/auto", json={"max_results_per_source": 1})
    assert r.status_code == 200
    assert "Nothing new cleared the relevance bar today" in r.json()["brief_text"]


class TestArcStatusExtract:
    def test_extracts_and_persists_a_real_project_mention(self, client):
        r = client.post(
            "/arc-status/extract",
            json={"text": "Finally kicked off the Assay GPU run tonight, first seed is training."},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["updates"]) == 1
        assert body["updates"][0]["project"] == "Assay"
        assert "Updated:" in body["confirmation_text"]

        # actually persisted, not just returned -- confirm via GET
        stored = client.get("/arc-status").json()
        assert any(s["project"] == "Assay" for s in stored)

    def test_unrelated_text_extracts_nothing_and_says_so(self, client):
        r = client.post("/arc-status/extract", json={"text": "just checking if the bot is alive"})
        assert r.status_code == 200
        body = r.json()
        assert body["updates"] == []
        assert "Didn't find" in body["confirmation_text"]

    def test_falls_back_to_default_known_projects_when_store_is_empty(self, client):
        # No prior /arc-status posts in this test's isolated data dir, so
        # extraction must fall back to DEFAULT_KNOWN_PROJECTS rather than
        # matching against an empty list (which would silently extract
        # nothing, forever, for a brand new install).
        r = client.post("/arc-status/extract", json={"text": "Started work on Polygraph's fuzzer today."})
        assert len(r.json()["updates"]) == 1

    def test_prefers_existing_stored_projects_over_defaults(self, client):
        client.post(
            "/arc-status", json={"project": "SideQuest", "status": "s", "next_action": "n"}
        )
        r = client.post("/arc-status/extract", json={"text": "Made progress on SideQuest today."})
        assert len(r.json()["updates"]) == 1
        assert r.json()["updates"][0]["project"] == "SideQuest"

    def test_second_extraction_updates_rather_than_duplicates(self, client):
        client.post("/arc-status/extract", json={"text": "Started the Assay GPU run."})
        client.post("/arc-status/extract", json={"text": "Assay run finished, results look clean."})
        stored = client.get("/arc-status").json()
        assay_rows = [s for s in stored if s["project"] == "Assay"]
        assert len(assay_rows) == 1
        assert "finished" in assay_rows[0]["status"]

    def test_missing_text_field_degrades_gracefully_rather_than_422(self, client):
        # text is Optional -- a completely missing field is treated the
        # same as an empty/null one (e.g. a voice message with nothing
        # transcribed), not rejected outright.
        r = client.post("/arc-status/extract", json={})
        assert r.status_code == 200
        assert r.json()["updates"] == []
        assert "No text to work with" in r.json()["confirmation_text"]

    def test_explicit_null_text_degrades_gracefully(self, client):
        r = client.post("/arc-status/extract", json={"text": None})
        assert r.status_code == 200
        assert "No text to work with" in r.json()["confirmation_text"]

    def test_whitespace_only_text_degrades_gracefully(self, client):
        r = client.post("/arc-status/extract", json={"text": "   "})
        assert r.status_code == 200
        assert "No text to work with" in r.json()["confirmation_text"]


def test_delete_arc_status(client):
    client.post("/arc-status", json={"project": "TempProject", "status": "wip", "next_action": "done"})
    r = client.delete("/arc-status/TempProject")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

    r_404 = client.delete("/arc-status/TempProject")
    assert r_404.status_code == 404


def test_get_brief_history(client):
    client.post("/brief", json={"candidates": [], "top_k": 5})
    r = client.get("/brief/history")
    assert r.status_code == 200
    history = r.json()
    assert len(history) == 1


def test_taxonomy_endpoints(client):
    r_get = client.get("/taxonomy")
    assert r_get.status_code == 200
    assert "taxonomy" in r_get.json()

    new_tax = ["RLVR", "GRPO", "Agentic Coding"]
    r_post = client.post("/taxonomy", json={"taxonomy": new_tax})
    assert r_post.status_code == 200
    assert r_post.json()["taxonomy"] == new_tax

    r_get_updated = client.get("/taxonomy")
    assert r_get_updated.json()["taxonomy"] == new_tax

