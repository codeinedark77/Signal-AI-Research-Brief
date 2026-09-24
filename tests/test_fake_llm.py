from __future__ import annotations

from signal_brief.llm.fake_client import FakeLLMClient


class TestScoreRelevance:
    def test_no_overlap_scores_zero(self):
        client = FakeLLMClient()
        score, rationale = client.score_relevance("Cooking pasta", "A recipe.", ["GRPO", "reward hacking"])
        assert score == 0.0
        assert "no" in rationale.lower()

    def test_single_keyword_hit(self):
        client = FakeLLMClient()
        score, rationale = client.score_relevance(
            "A paper about GRPO", "details", ["GRPO", "reward hacking"]
        )
        assert score == 2.0
        assert "GRPO" in rationale

    def test_multiple_keyword_hits_sum_and_cap_at_ten(self):
        client = FakeLLMClient()
        taxonomy = ["GRPO", "reward hacking", "RLVR", "verifier", "agentic coding", "reward model"]
        score, _ = client.score_relevance(
            "GRPO reward hacking RLVR verifier agentic coding reward model paper", "x", taxonomy
        )
        # 6 keywords * 2.0 = 12.0, must be capped at 10.0 (matches the
        # ScoredCandidate validator's le=10 bound -- this is exactly the
        # kind of boundary a fake client needs to respect to be a useful
        # stand-in for the real one in graph tests).
        assert score == 10.0

    def test_matching_is_case_insensitive(self):
        client = FakeLLMClient()
        score, _ = client.score_relevance("grpo study", "x", ["GRPO"])
        assert score == 2.0

    def test_matches_against_title_and_summary_both(self):
        client = FakeLLMClient()
        score, _ = client.score_relevance("Title", "mentions GRPO here", ["GRPO"])
        assert score == 2.0


class TestSummarize:
    def test_short_summary_returned_unmodified(self):
        client = FakeLLMClient()
        assert client.summarize("T", "A short summary.") == "A short summary."

    def test_long_summary_is_truncated_with_ellipsis(self):
        client = FakeLLMClient()
        long_summary = " ".join(f"word{i}" for i in range(50))
        result = client.summarize("T", long_summary)
        assert result.endswith("...")
        # "..." is appended directly to the 30th word, not added as its own
        # token, so split() still yields exactly 30 words -- the last one
        # just carries the ellipsis suffix.
        words = result.split()
        assert len(words) == 30
        assert words[-1] == "word29..."

    def test_empty_summary_falls_back_to_title(self):
        client = FakeLLMClient()
        assert client.summarize("Fallback Title", "") == "Fallback Title"


class TestExtractArcUpdates:
    KNOWN = ["Hermes", "Orchestrator", "Polygraph", "Crucible", "Assay", "Signal"]

    def test_no_known_project_mentioned_returns_empty(self):
        client = FakeLLMClient()
        updates = client.extract_arc_updates("just checking if the bot is alive", self.KNOWN)
        assert updates == []

    def test_single_project_mention_is_extracted(self):
        client = FakeLLMClient()
        text = "Finally kicked off the Assay GPU run tonight, first seed is training."
        updates = client.extract_arc_updates(text, self.KNOWN)
        assert len(updates) == 1
        assert updates[0].project == "Assay"
        assert "kicked off the Assay GPU run" in updates[0].status

    def test_multiple_project_mentions_each_extracted(self):
        client = FakeLLMClient()
        text = "Fixed a fuzzer edge case in Polygraph. Also started the Assay GPU run."
        updates = client.extract_arc_updates(text, self.KNOWN)
        projects = {u.project for u in updates}
        assert projects == {"Polygraph", "Assay"}

    def test_next_action_sentence_is_shared_across_updates_in_one_message(self):
        client = FakeLLMClient()
        text = (
            "Started the Assay GPU run. Fixed a bug in Polygraph too. "
            "Next, check the loss curve in the morning."
        )
        updates = client.extract_arc_updates(text, self.KNOWN)
        assert all(u.next_action == "Next, check the loss curve in the morning." for u in updates)

    def test_no_next_sentence_falls_back_to_placeholder(self):
        client = FakeLLMClient()
        updates = client.extract_arc_updates("Started the Assay GPU run.", self.KNOWN)
        assert updates[0].next_action == "(not specified)"

    def test_matching_is_case_insensitive(self):
        client = FakeLLMClient()
        updates = client.extract_arc_updates("started the assay run today", self.KNOWN)
        assert len(updates) == 1
        assert updates[0].project == "Assay"

    def test_first_matching_sentence_wins_when_project_mentioned_twice(self):
        client = FakeLLMClient()
        text = "Assay is still queued. Assay actually started an hour ago."
        updates = client.extract_arc_updates(text, self.KNOWN)
        assert len(updates) == 1
        assert updates[0].status == "Assay is still queued."

    def test_empty_known_projects_list_returns_empty(self):
        client = FakeLLMClient()
        updates = client.extract_arc_updates("Started the Assay GPU run.", [])
        assert updates == []
