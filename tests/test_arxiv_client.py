from __future__ import annotations

from signal_brief.clients.arxiv_client import ArxivClient
from signal_brief.models import SourceType

EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="html">ArXiv Query</title>
  <id>http://arxiv.org/api/empty</id>
  <updated>2026-07-13T00:00:00-04:00</updated>
</feed>
"""

MALFORMED_ENTRY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="html">ArXiv Query</title>
  <id>http://arxiv.org/api/malformed</id>
  <updated>2026-07-13T00:00:00-04:00</updated>
  <entry>
    <!-- missing id, title, and published entirely -->
    <summary>An entry with no identifying fields at all.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2601.00001v1</id>
    <updated>2026-01-01T00:00:00Z</updated>
    <published>2026-01-01T00:00:00Z</published>
    <title>  A title  with
       odd whitespace  </title>
    <author><name>Someone</name></author>
  </entry>
</feed>
"""


class TestParseRealFixture:
    """These assertions are checked against a real captured arXiv response
    (two genuine, currently-indexed papers), not invented data."""

    def test_parses_expected_count(self, arxiv_sample_xml):
        candidates = ArxivClient.parse(arxiv_sample_xml)
        assert len(candidates) == 2

    def test_ids_are_stripped_of_version_suffix(self, arxiv_sample_xml):
        candidates = ArxivClient.parse(arxiv_sample_xml)
        ids = {c.external_id for c in candidates}
        assert ids == {"2605.21384", "2509.22047"}
        # the raw <id> values in the fixture carry v1/v2 suffixes; if this
        # regressed we'd see "2605.21384v1" leak through here instead.
        assert not any("v" in cid and cid.split("v")[-1].isdigit() for cid in ids)

    def test_all_candidates_are_arxiv_sourced(self, arxiv_sample_xml):
        candidates = ArxivClient.parse(arxiv_sample_xml)
        assert all(c.source is SourceType.ARXIV for c in candidates)

    def test_urls_point_to_abs_page_without_version_suffix(self, arxiv_sample_xml):
        candidates = ArxivClient.parse(arxiv_sample_xml)
        urls = {c.url for c in candidates}
        assert urls == {
            "https://arxiv.org/abs/2605.21384",
            "https://arxiv.org/abs/2509.22047",
        }

    def test_authors_are_extracted(self, arxiv_sample_xml):
        candidates = ArxivClient.parse(arxiv_sample_xml)
        by_id = {c.external_id: c for c in candidates}
        assert by_id["2605.21384"].authors == [
            "Bingchen Zhao",
            "Dhruv Srikanth",
            "Yuxiang Wu",
            "Zhengyao Jiang",
        ]
        assert by_id["2509.22047"].authors == ["Yuki Ichihara"]

    def test_summary_whitespace_is_normalized(self, arxiv_sample_xml):
        candidates = ArxivClient.parse(arxiv_sample_xml)
        for c in candidates:
            assert "\n" not in c.summary
            assert "  " not in c.summary  # no double-spaces from XML indentation


class TestParseEdgeCases:
    def test_empty_feed_returns_empty_list(self):
        assert ArxivClient.parse(EMPTY_FEED) == []

    def test_entry_missing_required_fields_is_skipped_not_fatal(self):
        # the first entry has none of id/title/published; the parser must
        # skip it rather than raising, and still return the valid second entry.
        candidates = ArxivClient.parse(MALFORMED_ENTRY_FEED)
        assert len(candidates) == 1
        assert candidates[0].external_id == "2601.00001"

    def test_title_whitespace_is_collapsed(self):
        candidates = ArxivClient.parse(MALFORMED_ENTRY_FEED)
        assert candidates[0].title == "A title with odd whitespace"


def _decode_search_query(url: str) -> str:
    from urllib.parse import parse_qs, urlparse

    qs = parse_qs(urlparse(url).query)
    return qs["search_query"][0]


class TestBuildQueryUrl:
    def test_combines_categories_and_keywords_with_and(self):
        with ArxivClient() as client:
            url = client.build_query_url(["reward hacking", "GRPO"], ["cs.AI", "cs.LG"])
            query = _decode_search_query(url)
            assert query == '(cat:cs.AI OR cat:cs.LG) AND (abs:"reward hacking" OR abs:"GRPO")'

    def test_keywords_only_when_no_categories(self):
        with ArxivClient() as client:
            url = client.build_query_url(["GRPO", "RLVR"], [])
            query = _decode_search_query(url)
            # no categories given -> no "cat:" clause and no dangling "AND ()"
            assert "cat:" not in query
            assert query == 'abs:"GRPO" OR abs:"RLVR"'

    def test_categories_only_when_no_keywords(self):
        with ArxivClient() as client:
            url = client.build_query_url([], ["cs.SE"])
            query = _decode_search_query(url)
            assert query == "cat:cs.SE"

    def test_max_results_and_sort_params_present(self):
        with ArxivClient() as client:
            url = client.build_query_url(["GRPO"], ["cs.LG"], max_results=7)
            from urllib.parse import parse_qs, urlparse

            qs = parse_qs(urlparse(url).query)
            assert qs["max_results"] == ["7"]
            assert qs["sortBy"] == ["submittedDate"]
            assert qs["sortOrder"] == ["descending"]
