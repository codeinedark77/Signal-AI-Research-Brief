"""SQLite-backed persistence: seen-item dedupe store and arc status tracker.

Two small, independent stores. Deliberately not an ORM -- this is the
same "boring, inspectable, stdlib-first" bias used elsewhere: the data
is a handful of rows, so a query builder would be pure overhead.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .models import ArcStatus, BriefResponse


class SeenStore:
    """Tracks which candidate dedupe_keys have already been surfaced."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_items (
                    dedupe_key TEXT PRIMARY KEY,
                    first_seen_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def filter_unseen(self, dedupe_keys: list[str]) -> set[str]:
        """Return the subset of dedupe_keys NOT already recorded as seen."""
        if not dedupe_keys:
            return set()
        with self._conn() as conn:
            placeholders = ",".join("?" for _ in dedupe_keys)
            rows = conn.execute(
                f"SELECT dedupe_key FROM seen_items WHERE dedupe_key IN ({placeholders})",
                dedupe_keys,
            ).fetchall()
        seen = {r[0] for r in rows}
        return set(dedupe_keys) - seen

    def mark_seen(self, dedupe_keys: list[str]) -> None:
        if not dedupe_keys:
            return
        now = datetime.now().astimezone().isoformat()
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO seen_items (dedupe_key, first_seen_at) VALUES (?, ?)",
                [(k, now) for k in dedupe_keys],
            )

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM seen_items").fetchone()
        return int(row[0])

    def prune(self, days_old: int = 30) -> None:
        """Delete seen items older than days_old."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM seen_items WHERE first_seen_at < datetime('now', ?)",
                (f"-{days_old} days",),
            )


class ArcStatusStore:
    """Tracks per-project status/next-action -- the 'where did I leave off' state."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS arc_status (
                    project TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    next_action TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert(self, status: ArcStatus) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO arc_status (project, status, next_action, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(project) DO UPDATE SET
                    status=excluded.status,
                    next_action=excluded.next_action,
                    updated_at=excluded.updated_at
                """,
                (status.project, status.status, status.next_action, status.updated_at.isoformat()),
            )

    def all(self) -> list[ArcStatus]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT project, status, next_action, updated_at FROM arc_status ORDER BY project"
            ).fetchall()
        return [
            ArcStatus(
                project=r[0],
                status=r[1],
                next_action=r[2],
                updated_at=datetime.fromisoformat(r[3]),
            )
            for r in rows
        ]

    def delete(self, project: str) -> bool:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM arc_status WHERE project = ?", (project,))
            return cursor.rowcount > 0

class BriefStore:
    """Tracks generated briefs to serve to the dashboard."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS briefs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, brief: BriefResponse) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO briefs (generated_at, payload_json) VALUES (?, ?)",
                (brief.generated_at.isoformat(), brief.model_dump_json()),
            )
            conn.execute(
                "DELETE FROM briefs WHERE id NOT IN (SELECT id FROM briefs ORDER BY generated_at DESC LIMIT 20)"
            )

    def get_latest(self) -> BriefResponse | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload_json FROM briefs ORDER BY generated_at DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return BriefResponse.model_validate_json(row[0])

    def get_history(self, limit: int = 10) -> list[BriefResponse]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM briefs ORDER BY generated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [BriefResponse.model_validate_json(r[0]) for r in rows]

