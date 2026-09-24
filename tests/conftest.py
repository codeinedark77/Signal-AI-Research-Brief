from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def arxiv_sample_xml() -> str:
    return (FIXTURES_DIR / "arxiv_sample_response.xml").read_text()


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"
