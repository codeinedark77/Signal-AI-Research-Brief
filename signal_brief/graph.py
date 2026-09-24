"""The reasoning graph: dedupe -> score -> rank/select -> summarize -> compose.

This is the one piece worth being an actual LangGraph graph rather than a
plain function -- each stage is independently testable, the state is
explicit (no hidden prompt history), and it's easy to slot in more nodes
later (e.g. a "fetch full text if borderline" loop) without restructuring.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .llm.base import LLMClient
from .models import ArcStatus, CalendarEvent, Candidate, ScoredCandidate
from .stores import SeenStore

DEFAULT_TAXONOMY = [
    "RLVR",
    "reinforcement learning from verifiable rewards",
    "GRPO",
    "group relative policy optimization",
    "reward hacking",
    "reward model",
    "specification gaming",
    "agentic coding",
    "autonomous coding agent",
    "coding agent benchmark",
    "verifier",
    "LLM agent",
    "RL environment",
]

# Seeds ArcStatusStore project-name matching for the evening capture loop
# before you've ever posted a status update (empty store -> nothing to
# match against otherwise). Once you've posted real updates, /arc-status/
# extract prefers whatever's already in the store over this list.
DEFAULT_KNOWN_PROJECTS = [
    "Hermes",
    "Orchestrator",
    "Polygraph",
    "Crucible",
    "Assay",
    "Signal",
]


class BriefState(TypedDict, total=False):
    raw_candidates: list[Candidate]
    deduped_candidates: list[Candidate]
    scored_candidates: list[ScoredCandidate]
    top_candidates: list[ScoredCandidate]
    calendar_events: list[CalendarEvent]
    arc_status: list[ArcStatus]
    taxonomy: list[str]
    top_k: int
    brief_text: str


def build_graph(llm: LLMClient, seen_store: SeenStore | None = None):
    """Compile the graph. `seen_store` is optional so the graph can be unit
    tested without touching disk; the API/CLI entrypoints always pass one.
    """

    def dedupe_node(state: BriefState) -> dict:
        candidates = state.get("raw_candidates") or []
        if seen_store is None:
            deduped = candidates
        else:
            keys = [c.dedupe_key for c in candidates]
            unseen_keys = seen_store.filter_unseen(keys)
            deduped = [c for c in candidates if c.dedupe_key in unseen_keys]
        return {"deduped_candidates": deduped}

    def score_node(state: BriefState) -> dict:
        taxonomy = state.get("taxonomy") or DEFAULT_TAXONOMY
        scored = []
        for c in state.get("deduped_candidates") or []:
            score, rationale = llm.score_relevance(c.title, c.summary, taxonomy)
            scored.append(ScoredCandidate(candidate=c, relevance_score=score, rationale=rationale))
        return {"scored_candidates": scored}

    def rank_select_node(state: BriefState) -> dict:
        top_k = state.get("top_k", 5)
        ranked = sorted(
            state.get("scored_candidates") or [], key=lambda sc: sc.relevance_score, reverse=True
        )
        return {"top_candidates": ranked[:top_k]}

    def summarize_node(state: BriefState) -> dict:
        summarized = []
        for sc in state.get("top_candidates") or []:
            digest = llm.summarize(sc.candidate.title, sc.candidate.summary)
            summarized.append(sc.model_copy(update={"digest_summary": digest}))
        return {"top_candidates": summarized}

    def compose_node(state: BriefState) -> dict:
        now = datetime.now(timezone.utc)
        lines = [f"Signal -- {now:%A, %d %B %Y}", ""]

        events = state.get("calendar_events") or []
        if events:
            lines.append("Today:")
            for e in sorted(events, key=lambda ev: ev.start):
                lines.append(f"  - {e.start:%H:%M} {e.title}")
            lines.append("")

        arc = state.get("arc_status") or []
        if arc:
            lines.append("Where you left off:")
            for a in arc:
                lines.append(f"  - {a.project}: {a.status} -> next: {a.next_action}")
            lines.append("")

        top = state.get("top_candidates") or []
        if top:
            lines.append("Worth a look (last run):")
            for sc in top:
                lines.append(f"  - [{sc.relevance_score:.1f}] {sc.candidate.title}")
                lines.append(f"    {sc.digest_summary or sc.rationale}")
                lines.append(f"    {sc.candidate.url}")
        else:
            lines.append("Nothing new cleared the relevance bar today.")

        return {"brief_text": "\n".join(lines)}

    def persist_seen_node(state: BriefState) -> dict:
        # Owned by the graph, not the caller: if dedupe checked seen_store,
        # this node is what keeps that check honest on the next run. This
        # runs last (after compose), so a mid-pipeline exception -- e.g. the
        # LLM call in score_node failing -- means nothing gets marked seen
        # and today's candidates are still eligible on the next attempt.
        if seen_store is not None:
            keys = [c.dedupe_key for c in state.get("deduped_candidates") or []]
            seen_store.mark_seen(keys)
            seen_store.prune(days_old=30)
        return {}

    graph = StateGraph(BriefState)
    graph.add_node("dedupe", dedupe_node)
    graph.add_node("score", score_node)
    graph.add_node("rank_select", rank_select_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("compose", compose_node)
    graph.add_node("persist_seen", persist_seen_node)

    graph.set_entry_point("dedupe")
    graph.add_edge("dedupe", "score")
    graph.add_edge("score", "rank_select")
    graph.add_edge("rank_select", "summarize")
    graph.add_edge("summarize", "compose")
    graph.add_edge("compose", "persist_seen")
    graph.add_edge("persist_seen", END)

    return graph.compile()
