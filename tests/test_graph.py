from __future__ import annotations

from datetime import datetime, timezone

import pytest

from signal_brief.graph import DEFAULT_TAXONOMY, build_graph
from signal_brief.llm.fake_client import FakeLLMClient
from signal_brief.models import ArcStatus, CalendarEvent, Candidate, SourceType
from signal_brief.stores import SeenStore


def _candidate(idx: int, title: str, summary: str = "") -> Candidate:
    return Candidate(
        source=SourceType.ARXIV,
        external_id=f"id-{idx}",
        title=title,
        summary=summary,
        url=f"https://arxiv.org/abs/id-{idx}",
        published_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )


class TestDedupe:
    def test_without_seen_store_nothing_is_filtered(self):
        graph = build_graph(llm=FakeLLMClient(), seen_store=None)
        candidates = [_candidate(1, "GRPO paper")]
        result = graph.invoke({"raw_candidates": candidates, "taxonomy": DEFAULT_TAXONOMY})
        assert len(result["deduped_candidates"]) == 1

    def test_second_run_with_same_seen_store_sees_nothing_new(self, tmp_db_path):
        """Regression test for the bug where mark_seen was an external,
        easy-to-forget post-invoke step: this exercises the graph exactly
        as a real caller would (two independent build_graph + invoke calls,
        NOT calling mark_seen manually), and asserts dedup still holds."""
        candidates = [_candidate(1, "GRPO paper"), _candidate(2, "Unrelated paper")]

        result_1 = build_graph(llm=FakeLLMClient(), seen_store=SeenStore(tmp_db_path)).invoke(
            {"raw_candidates": candidates, "taxonomy": DEFAULT_TAXONOMY}
        )
        assert len(result_1["deduped_candidates"]) == 2

        result_2 = build_graph(llm=FakeLLMClient(), seen_store=SeenStore(tmp_db_path)).invoke(
            {"raw_candidates": candidates, "taxonomy": DEFAULT_TAXONOMY}
        )
        assert result_2["deduped_candidates"] == []
        assert "Nothing new cleared the relevance bar today" in result_2["brief_text"]

    def test_partial_overlap_only_new_items_survive(self, tmp_db_path):
        first_batch = [_candidate(1, "GRPO paper")]
        build_graph(llm=FakeLLMClient(), seen_store=SeenStore(tmp_db_path)).invoke(
            {"raw_candidates": first_batch, "taxonomy": DEFAULT_TAXONOMY}
        )

        second_batch = [_candidate(1, "GRPO paper"), _candidate(2, "New paper")]
        result = build_graph(llm=FakeLLMClient(), seen_store=SeenStore(tmp_db_path)).invoke(
            {"raw_candidates": second_batch, "taxonomy": DEFAULT_TAXONOMY}
        )
        assert [c.external_id for c in result["deduped_candidates"]] == ["id-2"]

    def test_partially_failed_run_does_not_mark_seen(self, tmp_db_path, monkeypatch):
        """If score_node blows up mid-run, persist_seen never executes (it's
        downstream of score/rank/summarize/compose), so the candidates
        involved must still be eligible on the next attempt -- verifies the
        ordering choice in build_graph, not just the happy path."""

        class ExplodingLLM(FakeLLMClient):
            def score_relevance(self, title, summary, taxonomy):
                raise RuntimeError("simulated LLM outage")

        candidates = [_candidate(1, "GRPO paper")]
        graph = build_graph(llm=ExplodingLLM(), seen_store=SeenStore(tmp_db_path))
        with pytest.raises(RuntimeError, match="simulated LLM outage"):
            graph.invoke({"raw_candidates": candidates, "taxonomy": DEFAULT_TAXONOMY})

        # nothing should have been marked seen -- the failed run's
        # candidates must still be eligible next time.
        retry = build_graph(llm=FakeLLMClient(), seen_store=SeenStore(tmp_db_path)).invoke(
            {"raw_candidates": candidates, "taxonomy": DEFAULT_TAXONOMY}
        )
        assert len(retry["deduped_candidates"]) == 1


class TestRankAndSelect:
    def test_top_k_respects_limit(self):
        candidates = [_candidate(i, f"GRPO paper {i}") for i in range(10)]
        graph = build_graph(llm=FakeLLMClient())
        result = graph.invoke({"raw_candidates": candidates, "taxonomy": DEFAULT_TAXONOMY, "top_k": 3})
        assert len(result["top_candidates"]) == 3

    def test_ranking_is_descending_by_score(self):
        low = _candidate(1, "Unrelated cooking content")
        high = _candidate(2, "GRPO reward hacking RLVR paper")
        graph = build_graph(llm=FakeLLMClient())
        result = graph.invoke(
            {"raw_candidates": [low, high], "taxonomy": DEFAULT_TAXONOMY, "top_k": 5}
        )
        scores = [sc.relevance_score for sc in result["top_candidates"]]
        assert scores == sorted(scores, reverse=True)
        assert result["top_candidates"][0].candidate.external_id == "id-2"

    def test_summarize_only_runs_on_top_k_not_all_scored(self):
        """Cost/latency matters here: summarizing every candidate defeats
        the point of filtering first. Confirms scored_candidates can be
        larger than top_candidates and only the latter gets a digest_summary."""
        candidates = [_candidate(i, f"GRPO paper {i}", summary="x " * 40) for i in range(5)]
        graph = build_graph(llm=FakeLLMClient())
        result = graph.invoke({"raw_candidates": candidates, "taxonomy": DEFAULT_TAXONOMY, "top_k": 2})
        assert len(result["scored_candidates"]) == 5
        assert len(result["top_candidates"]) == 2
        assert all(sc.digest_summary is not None for sc in result["top_candidates"])


class TestCompose:
    def test_empty_candidates_produces_explicit_empty_message(self):
        graph = build_graph(llm=FakeLLMClient())
        result = graph.invoke({"raw_candidates": [], "taxonomy": DEFAULT_TAXONOMY})
        assert "Nothing new cleared the relevance bar today" in result["brief_text"]

    def test_calendar_events_are_sorted_and_included(self):
        graph = build_graph(llm=FakeLLMClient())
        events = [
            CalendarEvent(title="Later", start=datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc)),
            CalendarEvent(title="Earlier", start=datetime(2026, 7, 13, 9, 0, tzinfo=timezone.utc)),
        ]
        result = graph.invoke(
            {"raw_candidates": [], "calendar_events": events, "taxonomy": DEFAULT_TAXONOMY}
        )
        earlier_pos = result["brief_text"].index("Earlier")
        later_pos = result["brief_text"].index("Later")
        assert earlier_pos < later_pos

    def test_arc_status_appears_in_brief(self):
        graph = build_graph(llm=FakeLLMClient())
        arc = [ArcStatus(project="Assay", status="GPU run not yet executed", next_action="Run GRPO training")]
        result = graph.invoke({"raw_candidates": [], "arc_status": arc, "taxonomy": DEFAULT_TAXONOMY})
        assert "Assay" in result["brief_text"]
        assert "GPU run not yet executed" in result["brief_text"]
        assert "Run GRPO training" in result["brief_text"]

    def test_missing_optional_state_keys_do_not_raise(self):
        """A caller (e.g. a minimal n8n payload) might omit calendar_events/
        arc_status/taxonomy entirely -- the graph must not KeyError."""
        graph = build_graph(llm=FakeLLMClient())
        result = graph.invoke({"raw_candidates": [_candidate(1, "GRPO paper")]})
        assert result["brief_text"]  # non-empty, didn't raise


class TestTaxonomyOverride:
    def test_custom_taxonomy_replaces_default(self):
        graph = build_graph(llm=FakeLLMClient())
        candidate = _candidate(1, "A paper about pastry lamination technique")
        result = graph.invoke(
            {
                "raw_candidates": [candidate],
                "taxonomy": ["pastry", "lamination"],
                "top_k": 5,
            }
        )
        assert result["scored_candidates"][0].relevance_score == 4.0  # 2 keyword hits

    def test_no_taxonomy_falls_back_to_default(self):
        graph = build_graph(llm=FakeLLMClient())
        candidate = _candidate(1, "A paper about GRPO reward hacking")
        result = graph.invoke({"raw_candidates": [candidate]})  # taxonomy omitted
        assert result["scored_candidates"][0].relevance_score > 0
