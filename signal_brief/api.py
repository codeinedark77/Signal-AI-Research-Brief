"""FastAPI wrapper exposing POST /brief. This is the one HTTP call n8n's
plumbing needs to make to get the "smart" half of the pipeline done.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .clients.arxiv_client import ArxivClient
from .clients.github_client import GitHubClient
from .clients.huggingface_client import HuggingFaceClient
from .graph import DEFAULT_KNOWN_PROJECTS, DEFAULT_TAXONOMY, build_graph
from .llm.base import LLMClient
from .llm.fake_client import FakeLLMClient
from .models import ArcStatus, BriefRequest, BriefResponse, CalendarEvent
from .stores import ArcStatusStore, SeenStore, BriefStore

DATA_DIR = Path(os.environ.get("SIGNAL_DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Signal Brief Service", version="0.1.0")


def _get_llm_client() -> LLMClient:
    if os.environ.get("SIGNAL_USE_FAKE_LLM") == "1":
        from .llm.fake_client import FakeLLMClient
        return FakeLLMClient()
    if os.environ.get("OPENAI_API_KEY"):
        from .llm.openai_client import OpenAILLMClient
        return OpenAILLMClient()
    if os.environ.get("ANTHROPIC_API_KEY"):
        from .llm.anthropic_client import AnthropicLLMClient
        return AnthropicLLMClient()
    
    # Fallback to local Ollama
    from .llm.ollama_client import OllamaLLMClient
    return OllamaLLMClient(model="qwen2.5-coder:7b")


def _run_brief(candidates, calendar_events, top_k, taxonomy) -> BriefResponse:
    seen_store = SeenStore(DATA_DIR / "seen.db")
    arc_store = ArcStatusStore(DATA_DIR / "arc_status.db")

    llm = _get_llm_client()
    graph = build_graph(llm=llm, seen_store=seen_store)
    arc_status = arc_store.all()

    result = graph.invoke(
        {
            "raw_candidates": candidates,
            "calendar_events": calendar_events,
            "arc_status": arc_status,
            "taxonomy": taxonomy,
            "top_k": top_k,
        }
    )

    brief = BriefResponse(
        generated_at=datetime.now(timezone.utc),
        calendar_events=calendar_events,
        top_items=result.get("top_candidates", []),
        arc_status=arc_status,
        brief_text=result.get("brief_text", ""),
    )
    BriefStore(DATA_DIR / "briefs.db").save(brief)
    return brief


@app.post("/brief", response_model=BriefResponse)
def create_brief(payload: BriefRequest) -> BriefResponse:
    """Reasoning-only endpoint: n8n (or anything else) has already fetched
    the candidates and just wants them deduped/scored/ranked/summarized."""
    return _run_brief(payload.candidates, payload.calendar_events, payload.top_k, payload.taxonomy)


class AutoBriefRequest(BaseModel):
    """n8n only needs to supply calendar_events (which requires an OAuth
    credential n8n's Google Calendar node already handles) -- arXiv and
    GitHub ingestion happen here, using the same clients covered by
    test_arxiv_client.py / test_github_client.py, instead of being
    re-implemented as an untested n8n Code node."""

    calendar_events: list[CalendarEvent] = Field(default_factory=list)
    top_k: int = 5
    taxonomy: Optional[list[str]] = None
    arxiv_keywords: list[str] = Field(
        default_factory=lambda: ["reward hacking", "GRPO", "RLVR", "agentic coding", "coding agent"]
    )
    arxiv_categories: list[str] = Field(default_factory=lambda: ["cs.AI", "cs.LG", "cs.CL", "cs.SE"])
    github_keywords: list[str] = Field(
        default_factory=lambda: ["reward-hacking", "GRPO", "agentic-coding"]
    )
    huggingface_keywords: list[str] = Field(
        default_factory=lambda: ["GRPO", "RLVR", "coding agent"]
    )
    max_results_per_source: int = 15


@app.post("/brief/auto", response_model=BriefResponse)
def create_brief_auto(payload: AutoBriefRequest) -> BriefResponse:
    candidates = []

    with ArxivClient() as arxiv:
        try:
            candidates += arxiv.search(
                payload.arxiv_keywords, payload.arxiv_categories, payload.max_results_per_source
            )
        except Exception as exc:
            print(f"[signal] arXiv fetch failed: {exc}", file=sys.stderr)

    with GitHubClient(token=os.environ.get("GITHUB_TOKEN")) as github:
        try:
            pushed_after = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
            candidates += github.search(
                payload.github_keywords, 
                pushed_after=pushed_after, 
                max_results=payload.max_results_per_source
            )
        except Exception as exc:
            print(f"[signal] GitHub fetch failed: {exc}", file=sys.stderr)

    with HuggingFaceClient() as hf:
        try:
            candidates += hf.search(payload.huggingface_keywords, max_results=payload.max_results_per_source)
        except Exception as exc:
            print(f"[signal] Hugging Face fetch failed: {exc}", file=sys.stderr)

    return _run_brief(candidates, payload.calendar_events, payload.top_k, payload.taxonomy)


@app.post("/arc-status")
def update_arc_status(status: ArcStatus) -> dict:
    arc_store = ArcStatusStore(DATA_DIR / "arc_status.db")
    arc_store.upsert(status)
    return {"ok": True}


@app.get("/arc-status")
def list_arc_status() -> list[ArcStatus]:
    arc_store = ArcStatusStore(DATA_DIR / "arc_status.db")
    return arc_store.all()


class ArcExtractRequest(BaseModel):
    text: Optional[str] = None


class ArcExtractResponse(BaseModel):
    updates: list[ArcStatus]
    confirmation_text: str


def _format_confirmation(updates: list[ArcStatus]) -> str:
    if not updates:
        return "Didn't find a status update for any known project in that message -- nothing changed."
    lines = ["Updated:"]
    for u in updates:
        lines.append(f"  - {u.project}: {u.status}")
        lines.append(f"    next: {u.next_action}")
    return "\n".join(lines)


@app.post("/arc-status/extract", response_model=ArcExtractResponse)
def extract_arc_status(payload: ArcExtractRequest) -> ArcExtractResponse:
    """The evening capture loop's one call: n8n forwards whatever text came
    in on Telegram (already transcribed, if it was a voice note -- see
    README for why transcription itself isn't wired up here), this pulls
    out zero or more project updates and upserts them.
    """
    if not payload.text or not payload.text.strip():
        # A voice message with no transcription step wired up (or just an
        # empty message) lands here. Handling it as a normal, testable
        # response -- rather than making n8n branch on message type before
        # calling this endpoint -- keeps the workflow linear.
        return ArcExtractResponse(
            updates=[],
            confirmation_text=(
                "No text to work with. If that was a voice note, transcription isn't "
                "wired up yet -- type your update instead for now (see README)."
            ),
        )

    arc_store = ArcStatusStore(DATA_DIR / "arc_status.db")
    existing_projects = [s.project for s in arc_store.all()]
    known_projects = existing_projects or DEFAULT_KNOWN_PROJECTS

    llm = _get_llm_client()
    extracted = llm.extract_arc_updates(payload.text, known_projects)

    updates: list[ArcStatus] = []
    for e in extracted:
        status = ArcStatus(project=e.project, status=e.status, next_action=e.next_action)
        arc_store.upsert(status)
        updates.append(status)

    return ArcExtractResponse(updates=updates, confirmation_text=_format_confirmation(updates))

@app.post("/arc-status/voice", response_model=ArcExtractResponse)
async def extract_arc_status_voice(file: UploadFile = File(...)) -> ArcExtractResponse:
    """The evening capture loop's voice call: natively transcibes using Whisper
    then extracts project updates.
    """
    file_content = await file.read()
    if not file_content:
        return ArcExtractResponse(
            updates=[],
            confirmation_text="Uploaded file was empty."
        )

    llm = _get_llm_client()
    try:
        transcription = llm.transcribe_voice(file_content, file.filename or "audio.ogg")
    except Exception as e:
        return ArcExtractResponse(
            updates=[],
            confirmation_text=f"Failed to transcribe audio: {e}"
        )

    # Pass transcribed text to the standard extraction logic
    return extract_arc_status(ArcExtractRequest(text=transcription))

@app.delete("/arc-status/{project}")
def delete_arc_status(project: str) -> dict:
    arc_store = ArcStatusStore(DATA_DIR / "arc_status.db")
    deleted = arc_store.delete(project)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True}

@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}

@app.get("/brief/latest", response_model=BriefResponse)
def get_latest_brief() -> BriefResponse:
    store = BriefStore(DATA_DIR / "briefs.db")
    latest = store.get_latest()
    if not latest:
        raise HTTPException(status_code=404, detail="No brief found")
    return latest

@app.get("/brief/history", response_model=list[BriefResponse])
def get_brief_history(limit: int = 10) -> list[BriefResponse]:
    store = BriefStore(DATA_DIR / "briefs.db")
    return store.get_history(limit=limit)

@app.get("/taxonomy")
def get_taxonomy() -> dict:
    taxonomy_file = DATA_DIR / "taxonomy.json"
    if taxonomy_file.exists():
        import json
        try:
            return {"taxonomy": json.loads(taxonomy_file.read_text())}
        except Exception:
            pass
    return {"taxonomy": DEFAULT_TAXONOMY}

class TaxonomyUpdate(BaseModel):
    taxonomy: list[str]

@app.post("/taxonomy")
def update_taxonomy(payload: TaxonomyUpdate) -> dict:
    taxonomy_file = DATA_DIR / "taxonomy.json"
    import json
    taxonomy_file.write_text(json.dumps(payload.taxonomy, indent=2))
    return {"ok": True, "taxonomy": payload.taxonomy}

app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")

