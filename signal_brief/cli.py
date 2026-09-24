"""CLI entrypoint: fetch real sources, run the graph, print the brief.

This exists as an escape hatch: if you don't want to stand up n8n right
away, `python -m signal_brief.cli` alone, on a cron job, gets you the
morning brief with no plumbing framework at all.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .clients.arxiv_client import ArxivClient
from .clients.github_client import GitHubClient
from .clients.huggingface_client import HuggingFaceClient
from .graph import DEFAULT_TAXONOMY, build_graph
from .stores import ArcStatusStore, SeenStore


def _build_llm():
    if os.environ.get("SIGNAL_USE_FAKE_LLM") == "1":
        from .llm.fake_client import FakeLLMClient
        return FakeLLMClient()
    if os.environ.get("OPENAI_API_KEY"):
        from .llm.openai_client import OpenAILLMClient
        return OpenAILLMClient()
    if os.environ.get("ANTHROPIC_API_KEY"):
        from .llm.anthropic_client import AnthropicLLMClient
        return AnthropicLLMClient()
    
    # Fallback to local Ollama if no keys are provided
    from .llm.ollama_client import OllamaLLMClient
    return OllamaLLMClient(model="qwen2.5-coder:7b")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Signal daily brief once.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--categories", nargs="*", default=["cs.AI", "cs.LG", "cs.CL", "cs.SE"])
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=["reward hacking", "GRPO", "RLVR", "agentic coding", "coding agent"],
    )
    parser.add_argument(
        "--github-keywords", nargs="*", default=["reward-hacking", "GRPO", "agentic-coding"]
    )
    parser.add_argument(
        "--huggingface-keywords", nargs="*", default=["GRPO", "RLVR", "coding agent"]
    )
    parser.add_argument("--max-results", type=int, default=15)
    parser.add_argument("--skip-arxiv", action="store_true")
    parser.add_argument("--skip-github", action="store_true")
    parser.add_argument("--skip-huggingface", action="store_true")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    candidates = []

    if not args.skip_arxiv:
        with ArxivClient() as arxiv:
            try:
                candidates += arxiv.search(args.keywords, args.categories, args.max_results)
            except Exception as exc:  # network/parse failure shouldn't kill the whole run
                print(f"[signal] arXiv fetch failed: {exc}", file=sys.stderr)

    if not args.skip_github:
        with GitHubClient(token=os.environ.get("GITHUB_TOKEN")) as github:
            try:
                pushed_after = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
                candidates += github.search(
                    args.github_keywords, 
                    pushed_after=pushed_after, 
                    max_results=args.max_results
                )
            except Exception as exc:
                print(f"[signal] GitHub fetch failed: {exc}", file=sys.stderr)

    if not args.skip_huggingface:
        with HuggingFaceClient() as hf:
            try:
                candidates += hf.search(args.huggingface_keywords, max_results=args.max_results)
            except Exception as exc:
                print(f"[signal] Hugging Face fetch failed: {exc}", file=sys.stderr)

    seen_store = SeenStore(data_dir / "seen.db")
    arc_store = ArcStatusStore(data_dir / "arc_status.db")

    graph = build_graph(llm=_build_llm(), seen_store=seen_store)
    result = graph.invoke(
        {
            "raw_candidates": candidates,
            "calendar_events": [],
            "arc_status": arc_store.all(),
            "taxonomy": DEFAULT_TAXONOMY,
            "top_k": args.top_k,
        }
    )
    print(result.get("brief_text", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
