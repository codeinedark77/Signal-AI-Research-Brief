"""Ollama-backed LLM client mapped through the OpenAI SDK."""
from __future__ import annotations

import openai
from pydantic import BaseModel, Field

from ..models import ArcUpdateExtraction

class RelevanceScore(BaseModel):
    score: float = Field(description="Relevance score from 0.0 to 10.0")
    rationale: str = Field(description="One-line rationale tied to the taxonomy")

class ArcExtractionResponse(BaseModel):
    updates: list[ArcUpdateExtraction] = Field(description="List of project updates extracted from the message")

class OllamaLLMClient:
    def __init__(self, model: str = "qwen2.5-coder:7b"):
        import os
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        # Point the OpenAI SDK at the local Ollama server
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key="ollama"  # Required by SDK, ignored by Ollama
        )
        self._model = model

    def score_relevance(self, title: str, summary: str, taxonomy: list[str]) -> tuple[float, str]:
        prompt = (
            "You score research/code items for relevance to a specific research taxonomy.\n"
            f"Taxonomy: {', '.join(taxonomy)}\n\n"
            f"Title: {title}\nSummary: {summary}\n\n"
        )
        try:
            resp = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format=RelevanceScore,
            )
            parsed = resp.choices[0].message.parsed
            if not parsed:
                return 0.0, "Unparseable model output"
            score = max(0.0, min(10.0, float(parsed.score)))
            return score, parsed.rationale
        except Exception as e:
            return 0.0, f"Error: {e}"

    def summarize(self, title: str, summary: str) -> str:
        prompt = (
            "Summarize this in 2-3 plain sentences for a daily research digest. "
            "No preamble, just the summary.\n\n"
            f"Title: {title}\nAbstract: {summary}"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    def extract_arc_updates(self, text: str, known_projects: list[str]) -> list[ArcUpdateExtraction]:
        prompt = (
            "You extract project status updates from a freeform evening check-in "
            "message written by a solo developer. Known projects: "
            f"{', '.join(known_projects)}.\n\n"
            f"Message: {text}\n\n"
            "If a project's next action isn't stated, use \"(not specified)\". "
            "If the message doesn't describe progress on any known project "
            "(e.g. small talk, an unrelated question), return an empty list. "
            "Do not invent a project that isn't in the known list unless the "
            "message clearly names a new one."
        )
        resp = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format=ArcExtractionResponse,
        )
        parsed = resp.choices[0].message.parsed
        if not parsed:
            return []
        
        # Enforce defaults for next_action if model returned empty
        for update in parsed.updates:
            if not update.next_action:
                update.next_action = "(not specified)"
        
        return parsed.updates

    def transcribe_voice(self, file_content: bytes, filename: str = "audio.ogg") -> str:
        # Ollama doesn't natively do whisper through the OpenAI endpoint yet
        return "Voice transcription not currently supported by local Ollama endpoint."
