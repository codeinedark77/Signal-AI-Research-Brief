"""Deterministic fake LLM client -- no network calls, used by the test suite
and available as SIGNAL_USE_FAKE_LLM=1 for a dry run without an API key.
"""
from __future__ import annotations

import re

from ..models import ArcUpdateExtraction


class FakeLLMClient:
    """Scores by keyword overlap against the taxonomy; summarizes by truncation.

    This is NOT meant to produce good relevance judgments -- it's meant to
    make the graph's control flow (dedupe -> score -> rank -> summarize ->
    compose) deterministically testable without hitting a real model.
    """

    def score_relevance(self, title: str, summary: str, taxonomy: list[str]) -> tuple[float, str]:
        text = f"{title} {summary}".lower()
        hits = [kw for kw in taxonomy if kw.lower() in text]
        score = min(10.0, 2.0 * len(hits))
        rationale = f"Matches: {', '.join(hits)}" if hits else "No taxonomy keyword overlap"
        return score, rationale

    def summarize(self, title: str, summary: str) -> str:
        words = summary.split()
        if not words:
            return title
        short = " ".join(words[:30])
        return f"{short}..." if len(words) > 30 else short

    def extract_arc_updates(self, text: str, known_projects: list[str]) -> list[ArcUpdateExtraction]:
        """One update per known project name found in the text, first
        matching sentence wins. `next_action` is pulled from a sentence
        starting with "next" if one exists, else a fixed placeholder --
        deterministic, not smart, which is exactly what a test double
        for this needs to be."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        next_action_sentence = next(
            (s for s in sentences if re.match(r"(?i)^next[,:]?\s", s)), None
        )
        next_action = next_action_sentence or "(not specified)"

        updates: list[ArcUpdateExtraction] = []
        for project in known_projects:
            match = next((s for s in sentences if project.lower() in s.lower()), None)
            if match is not None:
                updates.append(
                    ArcUpdateExtraction(project=project, status=match, next_action=next_action)
                )
        return updates

    def transcribe_voice(self, file_content: bytes, filename: str = "audio.ogg") -> str:
        return "This is a fake transcription of the voice note."
