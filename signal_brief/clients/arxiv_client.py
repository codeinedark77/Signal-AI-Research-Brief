"""arXiv Atom API client and parser.

The parser is deliberately stdlib-only (xml.etree.ElementTree) -- it's
a well-understood, well-specified format and pulling in a feed-parsing
dependency for this would be overkill.

Schema reference: https://info.arxiv.org/help/api/user-manual.html
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlencode

import httpx

from ..models import Candidate, SourceType

ATOM_NS = "http://www.w3.org/2005/Atom"
NS = {"atom": ATOM_NS}

BASE_URL = "http://export.arxiv.org/api/query"

_VERSION_SUFFIX_RE = re.compile(r"v\d+$")


class ArxivClient:
    def __init__(self, http_client: httpx.Client | None = None, timeout: float = 15.0):
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    def build_query_url(self, keywords: list[str], categories: list[str], max_results: int = 25) -> str:
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        kw_query = " OR ".join(f'abs:"{k}"' for k in keywords)
        if cat_query and kw_query:
            search_query = f"({cat_query}) AND ({kw_query})"
        else:
            search_query = cat_query or kw_query
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        return f"{BASE_URL}?{urlencode(params)}"

    def fetch_raw(self, keywords: list[str], categories: list[str], max_results: int = 25) -> str:
        url = self.build_query_url(keywords, categories, max_results)
        resp = self._client.get(url, headers={"User-Agent": "signal-brief/0.1 (personal research digest)"})
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def parse(atom_xml: str) -> list[Candidate]:
        """Parse an arXiv Atom response into Candidates.

        Malformed individual entries are skipped rather than aborting the
        whole batch -- a single bad entry shouldn't zero out today's brief.
        """
        root = ET.fromstring(atom_xml)
        candidates: list[Candidate] = []
        for entry in root.findall("atom:entry", NS):
            id_el = entry.find("atom:id", NS)
            title_el = entry.find("atom:title", NS)
            summary_el = entry.find("atom:summary", NS)
            published_el = entry.find("atom:published", NS)
            if id_el is None or title_el is None or published_el is None:
                continue
            if not (id_el.text and title_el.text and published_el.text):
                continue

            raw_id = id_el.text.strip()
            last_segment = raw_id.rstrip("/").split("/")[-1]
            external_id = _VERSION_SUFFIX_RE.sub("", last_segment)

            authors = [
                name_el.text.strip()
                for author_el in entry.findall("atom:author", NS)
                if (name_el := author_el.find("atom:name", NS)) is not None and name_el.text
            ]

            title = " ".join(title_el.text.split())
            summary = " ".join(summary_el.text.split()) if summary_el is not None and summary_el.text else ""
            published = datetime.fromisoformat(published_el.text.strip().replace("Z", "+00:00"))

            candidates.append(
                Candidate(
                    source=SourceType.ARXIV,
                    external_id=external_id,
                    title=title,
                    summary=summary,
                    url=f"https://arxiv.org/abs/{external_id}",
                    published_at=published,
                    authors=authors,
                )
            )
        return candidates

    def search(self, keywords: list[str], categories: list[str], max_results: int = 25) -> list[Candidate]:
        raw = self.fetch_raw(keywords, categories, max_results)
        return self.parse(raw)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "ArxivClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
