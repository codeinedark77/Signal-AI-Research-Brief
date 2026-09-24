from __future__ import annotations

import os

import pytest

from signal_brief.cli import main
from signal_brief.models import ArcStatus
from signal_brief.stores import ArcStatusStore


@pytest.fixture(autouse=True)
def fake_llm_env(monkeypatch):
    monkeypatch.setenv("SIGNAL_USE_FAKE_LLM", "1")


def test_main_with_both_sources_skipped_exits_cleanly(tmp_path, capsys):
    rc = main(["--data-dir", str(tmp_path), "--skip-arxiv", "--skip-github", "--skip-huggingface"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Nothing new cleared the relevance bar today" in out


def test_main_creates_data_dir_if_missing(tmp_path):
    data_dir = tmp_path / "nested" / "does" / "not" / "exist"
    assert not data_dir.exists()
    main(["--data-dir", str(data_dir), "--skip-arxiv", "--skip-github"])
    assert data_dir.exists()
    assert (data_dir / "seen.db").exists()
    assert (data_dir / "arc_status.db").exists()


def test_main_surfaces_previously_recorded_arc_status(tmp_path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ArcStatusStore(data_dir / "arc_status.db").upsert(
        ArcStatus(project="Assay", status="GPU run not yet executed", next_action="Run GRPO training on RTX 3050")
    )
    main(["--data-dir", str(data_dir), "--skip-arxiv", "--skip-github"])
    out = capsys.readouterr().out
    assert "Assay" in out
    assert "GPU run not yet executed" in out


def test_main_respects_top_k(tmp_path):
    # top_k only matters once there are candidates; with both sources
    # skipped there's nothing to rank, so this just confirms the flag
    # parses and doesn't blow up argparse (the ranking behavior itself is
    # covered thoroughly in test_graph.py against the graph directly).
    rc = main(["--data-dir", str(tmp_path), "--skip-arxiv", "--skip-github", "--top-k", "2"])
    assert rc == 0


def test_arxiv_fetch_failure_does_not_abort_the_run(tmp_path, monkeypatch, capsys):
    """If arXiv is unreachable (e.g. this sandbox's egress rules), the CLI
    must degrade to GitHub-only rather than crash. Simulates that by
    forcing the arXiv client to fail while GitHub is skipped too, so the
    only way this test passes is if the try/except in main() actually
    swallows the exception and continues."""
    import signal_brief.cli as cli_module

    class ExplodingArxivClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def search(self, *a, **kw):
            raise ConnectionError("simulated: export.arxiv.org unreachable from this network")

    monkeypatch.setattr(cli_module, "ArxivClient", lambda: ExplodingArxivClient())
    rc = main(["--data-dir", str(tmp_path), "--skip-github"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "arXiv fetch failed" in err
