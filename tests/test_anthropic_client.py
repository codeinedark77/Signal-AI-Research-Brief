from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from signal_brief.llm.anthropic_client import AnthropicLLMClient


def _fake_response(text: str):
    """Mimics the shape of an anthropic.types.Message enough for our
    _extract_text() to work: .content is a list of objects with .type
    and .text."""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


@pytest.fixture
def client():
    c = AnthropicLLMClient(api_key="sk-ant-fake-key-for-tests")
    c._client = MagicMock()
    return c


class TestInitialization:
    def test_raises_clearly_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set"):
            AnthropicLLMClient()

    def test_accepts_explicit_api_key_even_without_env_var(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        AnthropicLLMClient(api_key="sk-ant-explicit")  # must not raise

    def test_default_model_is_current_sonnet(self):
        c = AnthropicLLMClient(api_key="sk-ant-fake")
        assert c._model == "claude-sonnet-5"

    def test_model_override_is_respected(self):
        c = AnthropicLLMClient(model="claude-haiku-4-5-20251001", api_key="sk-ant-fake")
        assert c._model == "claude-haiku-4-5-20251001"


class TestScoreRelevance:
    def test_parses_well_formed_response(self, client):
        client._client.messages.create.return_value = _fake_response("7.5|Directly about GRPO reward hacking")
        score, rationale = client.score_relevance("T", "S", ["GRPO"])
        assert score == 7.5
        assert rationale == "Directly about GRPO reward hacking"

    def test_clamps_out_of_range_score_from_model(self, client):
        client._client.messages.create.return_value = _fake_response("15|way too high")
        score, _ = client.score_relevance("T", "S", ["GRPO"])
        assert score == 10.0

    def test_clamps_negative_score_from_model(self, client):
        client._client.messages.create.return_value = _fake_response("-3|negative somehow")
        score, _ = client.score_relevance("T", "S", ["GRPO"])
        assert score == 0.0

    def test_unparseable_output_degrades_to_zero_not_a_crash(self, client):
        client._client.messages.create.return_value = _fake_response("I'm not sure how to answer that.")
        score, rationale = client.score_relevance("T", "S", ["GRPO"])
        assert score == 0.0
        assert "Unparseable model output" in rationale

    def test_handles_rationale_containing_pipe_character(self, client):
        # DOTALL + greedy group(2) means a rationale with its own "|" must
        # still come through whole, not truncated at the first pipe.
        client._client.messages.create.return_value = _fake_response("6|Relevant to GRPO | reward hacking both")
        score, rationale = client.score_relevance("T", "S", ["GRPO"])
        assert score == 6.0
        assert rationale == "Relevant to GRPO | reward hacking both"


class TestSummarize:
    def test_returns_stripped_text(self, client):
        client._client.messages.create.return_value = _fake_response("  A clean two-sentence summary.  ")
        assert client.summarize("T", "S") == "A clean two-sentence summary."


class TestExtractArcUpdates:
    KNOWN = ["Hermes", "Orchestrator", "Polygraph", "Crucible", "Assay", "Signal"]

    def test_parses_well_formed_json_array(self, client):
        client._client.messages.create.return_value = _fake_response(
            '[{"project": "Assay", "status": "GPU run started", "next_action": "check loss curve"}]'
        )
        updates = client.extract_arc_updates("started assay", self.KNOWN)
        assert len(updates) == 1
        assert updates[0].project == "Assay"
        assert updates[0].status == "GPU run started"
        assert updates[0].next_action == "check loss curve"

    def test_strips_markdown_code_fence_around_json(self, client):
        client._client.messages.create.return_value = _fake_response(
            '```json\n[{"project": "Assay", "status": "started", "next_action": "wait"}]\n```'
        )
        updates = client.extract_arc_updates("x", self.KNOWN)
        assert len(updates) == 1
        assert updates[0].project == "Assay"

    def test_empty_array_for_unrelated_text(self, client):
        client._client.messages.create.return_value = _fake_response("[]")
        assert client.extract_arc_updates("hey is this thing on", self.KNOWN) == []

    def test_malformed_json_degrades_to_empty_list_not_a_crash(self, client):
        client._client.messages.create.return_value = _fake_response("not json at all, sorry")
        assert client.extract_arc_updates("x", self.KNOWN) == []

    def test_json_object_instead_of_array_degrades_to_empty_list(self, client):
        client._client.messages.create.return_value = _fake_response(
            '{"project": "Assay", "status": "started", "next_action": "wait"}'
        )
        assert client.extract_arc_updates("x", self.KNOWN) == []

    def test_item_missing_required_key_is_skipped_not_fatal(self, client):
        client._client.messages.create.return_value = _fake_response(
            '[{"project": "Assay"}, '
            '{"project": "Polygraph", "status": "fixed a bug", "next_action": "ship it"}]'
        )
        updates = client.extract_arc_updates("x", self.KNOWN)
        assert len(updates) == 1
        assert updates[0].project == "Polygraph"

    def test_missing_next_action_key_defaults_to_placeholder(self, client):
        client._client.messages.create.return_value = _fake_response(
            '[{"project": "Assay", "status": "started"}]'
        )
        updates = client.extract_arc_updates("x", self.KNOWN)
        assert updates[0].next_action == "(not specified)"

    def test_non_dict_items_in_array_are_skipped(self, client):
        client._client.messages.create.return_value = _fake_response('["just a string", 42, null]')
        assert client.extract_arc_updates("x", self.KNOWN) == []
