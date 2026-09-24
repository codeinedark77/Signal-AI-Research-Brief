"""Anthropic-backed LLM client. Reads ANTHROPIC_API_KEY from the environment.

Default model is claude-sonnet-5 -- relevance judgments here are nuanced
enough (is this GRPO paper actually adjacent to Crucible's reward design,
or just GRPO-adjacent noise?) to be worth a stronger model. Swap the
`model=` argument to "claude-haiku-4-5-20251001" if you want this to cost
closer to nothing and are fine with coarser judgments.
"""
from __future__ import annotations

import json
import os
import re

import anthropic

from ..models import ArcUpdateExtraction

_SCORE_LINE_RE = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*(.+)", re.DOTALL)


class AnthropicLLMClient:
    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it, pass api_key=, or use "
                "FakeLLMClient / SIGNAL_USE_FAKE_LLM=1 for a dry run."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def _extract_text(self, resp) -> str:
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()

    def score_relevance(self, title: str, summary: str, taxonomy: list[str]) -> tuple[float, str]:
        prompt = (
            "You score research/code items for relevance to a specific research taxonomy.\n"
            f"Taxonomy: {', '.join(taxonomy)}\n\n"
            f"Title: {title}\nSummary: {summary}\n\n"
            "Respond with EXACTLY one line, no preamble, in this format:\n"
            "<score 0-10>|<one-line rationale tied to the taxonomy>"
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        text = self._extract_text(resp)
        match = _SCORE_LINE_RE.match(text)
        if not match:
            return 0.0, f"Unparseable model output: {text[:120]}"
        score = max(0.0, min(10.0, float(match.group(1))))
        return score, match.group(2).strip()

    def summarize(self, title: str, summary: str) -> str:
        prompt = (
            "Summarize this in 2-3 plain sentences for a daily research digest. "
            "No preamble, just the summary.\n\n"
            f"Title: {title}\nAbstract: {summary}"
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_text(resp)

    def extract_arc_updates(self, text: str, known_projects: list[str]) -> list[ArcUpdateExtraction]:
        prompt = (
            "You extract project status updates from a freeform evening check-in "
            "message written by a solo developer. Known projects: "
            f"{', '.join(known_projects)}.\n\n"
            f"Message: {text}\n\n"
            "Respond with ONLY a JSON array (no markdown fences, no preamble), one "
            'object per project actually mentioned as having a status change: '
            '[{"project": "...", "status": "...", "next_action": "..."}]. '
            "If a project's next action isn't stated, use \"(not specified)\". "
            "If the message doesn't describe progress on any known project "
            "(e.g. small talk, an unrelated question), respond with []. "
            "Do not invent a project that isn't in the known list unless the "
            "message clearly names a new one."
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text_out = self._extract_text(resp)
        # Models occasionally wrap JSON in a fenced code block despite
        # instructions not to -- strip that before parsing rather than
        # failing the whole extraction over a formatting quirk.
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text_out.strip())
        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            return []
        if not isinstance(raw, list):
            return []

        updates = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            if not {"project", "status"} <= item.keys():
                continue
            updates.append(
                ArcUpdateExtraction(
                    project=str(item["project"]),
                    status=str(item["status"]),
                    next_action=str(item.get("next_action") or "(not specified)"),
                )
            )
        return updates

    def transcribe_voice(self, file_content: bytes, filename: str = "audio.ogg") -> str:
        raise NotImplementedError("Anthropic API does not support native voice transcription.")
