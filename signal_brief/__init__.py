"""Signal — a daily research + status brief for Yash's RLVR/agentic-coding work.

Plumbing (fetch, schedule, deliver) lives in n8n. This package is the
"reasoning" half: dedupe -> score -> rank -> summarize -> compose, built
as a small LangGraph graph and exposed over a FastAPI endpoint that n8n
calls with an HTTP Request node.
"""

__version__ = "0.1.0"
