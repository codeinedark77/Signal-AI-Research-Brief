"""GitHub Search API client -- repos with recent activity matching keywords."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from ..models import Candidate, SourceType

BASE_URL = "https://api.github.com/search/repositories"


class GitHubClient:
    def __init__(
        self,
        http_client: httpx.Client | None = None,
        token: str | None = None,
        timeout: float = 15.0,
    ):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "signal-brief/0.1",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = http_client or httpx.Client(timeout=timeout, headers=headers)
        self._owns_client = http_client is None

    @staticmethod
    def build_query(keywords: list[str], pushed_after: str | None = None) -> str:
        q = " ".join(keywords)
        if pushed_after:
            q += f" pushed:>{pushed_after}"
        return q

    def search(
        self, keywords: list[str], pushed_after: str | None = None, max_results: int = 15
    ) -> list[Candidate]:
        q = self.build_query(keywords, pushed_after)
        params = {"q": q, "sort": "updated", "order": "desc", "per_page": max_results}
        resp = self._client.get(BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        candidates: list[Candidate] = []
        for item in data.get("items", []):
            pushed_at = item.get("pushed_at")
            published = (
                datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
                if pushed_at
                else datetime.now(timezone.utc)
            )
            owner_login = (item.get("owner") or {}).get("login", "")
            candidates.append(
                Candidate(
                    source=SourceType.GITHUB,
                    external_id=item["full_name"],
                    title=item["full_name"],
                    summary=item.get("description") or "",
                    url=item["html_url"],
                    published_at=published,
                    authors=[owner_login] if owner_login else [],
                )
            )
        return candidates

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
