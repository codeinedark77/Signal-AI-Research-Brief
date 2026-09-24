"""Pluggable LLM client interface for the reasoning steps (score + summarize
+ extract). Anything satisfying this protocol can be dropped into
build_graph() or the /arc-status/extract endpoint. That's what makes both
testable without a network call or an API key: swap in FakeLLMClient for
tests, AnthropicLLMClient (or an Ollama-backed client you write later) for
the real run.
"""
from __future__ import annotations

from typing import Protocol

from ..models import ArcUpdateExtraction


class LLMClient(Protocol):
    def score_relevance(self, title: str, summary: str, taxonomy: list[str]) -> tuple[float, str]:
        """Return (score in [0, 10], one-line rationale)."""
        ...

    def summarize(self, title: str, summary: str) -> str:
        """Return a 2-3 sentence digest summary."""
        ...

    def extract_arc_updates(self, text: str, known_projects: list[str]) -> list[ArcUpdateExtraction]:
        """Pull zero or more (project, status, next_action) updates out of a
        freeform evening check-in message. Empty list if the text doesn't
        describe progress on anything -- not everything sent to the bot is
        a status update, and forcing one out of small talk would just
        pollute ArcStatusStore with noise.
        """
        ...

    def transcribe_voice(self, file_content: bytes, filename: str = "audio.ogg") -> str:
        """Transcribe an audio file into text."""
        ...
