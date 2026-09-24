"""Core data models shared across clients, stores, the graph, and the API."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    ARXIV = "arxiv"
    GITHUB = "github"
    HUGGINGFACE = "huggingface"


class Candidate(BaseModel):
    """A single raw item pulled from an ingestion source, pre-scoring."""

    source: SourceType
    external_id: str  # e.g. arXiv id "2605.21384" or "owner/repo"
    title: str
    summary: str = ""
    url: str
    published_at: datetime
    authors: list[str] = Field(default_factory=list)

    @property
    def dedupe_key(self) -> str:
        return f"{self.source.value}:{self.external_id}"


class ScoredCandidate(BaseModel):
    candidate: Candidate
    relevance_score: float = Field(ge=0, le=10)
    rationale: str
    digest_summary: Optional[str] = None  # filled by summarize_node for top-K only


class ArcStatus(BaseModel):
    project: str
    status: str
    next_action: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ArcUpdateExtraction(BaseModel):
    """One project's status update as extracted from a freeform evening
    check-in message. Distinct from ArcStatus (no updated_at) because this
    is the LLM's raw output before it's been upserted anywhere."""

    project: str
    status: str
    next_action: str


class CalendarEvent(BaseModel):
    title: str
    start: datetime
    end: Optional[datetime] = None
    location: Optional[str] = None


class BriefRequest(BaseModel):
    """Input payload n8n (or the CLI) posts to /brief."""

    candidates: list[Candidate] = Field(default_factory=list)
    calendar_events: list[CalendarEvent] = Field(default_factory=list)
    top_k: int = 5
    taxonomy: Optional[list[str]] = None  # override the default relevance taxonomy


class BriefResponse(BaseModel):
    generated_at: datetime
    calendar_events: list[CalendarEvent]
    top_items: list[ScoredCandidate]
    arc_status: list[ArcStatus]
    brief_text: str  # final human-readable digest, ready to send to Telegram/email
