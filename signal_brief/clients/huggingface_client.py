"""Hugging Face API client -- recent models matching keywords."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from ..models import Candidate, SourceType

BASE_URL = "https://huggingface.co/api/models"

class HuggingFaceClient:
    def __init__(self, http_client: httpx.Client | None = None, timeout: float = 15.0):
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    def search(self, keywords: list[str], max_results: int = 15) -> list[Candidate]:
        candidates: list[Candidate] = []
        for kw in keywords:
            params = {
                "search": kw,
                "sort": "lastModified",
                "direction": "-1",
                "limit": max_results
            }
            resp = self._client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            for item in data:
                last_modified = item.get("lastModified")
                published = (
                    datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
                    if last_modified
                    else datetime.now(timezone.utc)
                )
                model_id = item.get("id") or ""
                author = item.get("author") or ""
                
                candidates.append(
                    Candidate(
                        source=SourceType.HUGGINGFACE,
                        external_id=model_id,
                        title=model_id,
                        summary=f"Hugging Face Model: {model_id} (Pipeline: {item.get('pipeline_tag', 'N/A')})",
                        url=f"https://huggingface.co/{model_id}",
                        published_at=published,
                        authors=[author] if author else [],
                    )
                )
        # Dedupe across keyword searches
        unique_cands = {c.external_id: c for c in candidates}
        sorted_cands = sorted(unique_cands.values(), key=lambda c: c.published_at, reverse=True)
        return sorted_cands[:max_results]

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "HuggingFaceClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
