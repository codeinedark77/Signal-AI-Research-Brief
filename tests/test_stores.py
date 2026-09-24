from __future__ import annotations

from datetime import datetime, timezone

from signal_brief.models import ArcStatus
from signal_brief.stores import ArcStatusStore, SeenStore


class TestSeenStore:
    def test_new_keys_are_unseen(self, tmp_db_path):
        store = SeenStore(tmp_db_path)
        assert store.filter_unseen(["a", "b"]) == {"a", "b"}

    def test_marked_keys_are_filtered_out(self, tmp_db_path):
        store = SeenStore(tmp_db_path)
        store.mark_seen(["a"])
        assert store.filter_unseen(["a", "b"]) == {"b"}

    def test_empty_input_returns_empty_set(self, tmp_db_path):
        store = SeenStore(tmp_db_path)
        assert store.filter_unseen([]) == set()

    def test_mark_seen_is_idempotent(self, tmp_db_path):
        store = SeenStore(tmp_db_path)
        store.mark_seen(["a"])
        store.mark_seen(["a"])  # must not raise (INSERT OR IGNORE, PK collision)
        assert store.count() == 1

    def test_mark_seen_empty_list_is_a_noop(self, tmp_db_path):
        store = SeenStore(tmp_db_path)
        store.mark_seen([])
        assert store.count() == 0

    def test_persists_across_new_instances_same_path(self, tmp_db_path):
        SeenStore(tmp_db_path).mark_seen(["a", "b"])
        second = SeenStore(tmp_db_path)
        assert second.filter_unseen(["a", "b", "c"]) == {"c"}

    def test_dedupe_key_with_special_characters_does_not_break_sql(self, tmp_db_path):
        # dedupe keys can contain colons, slashes (github "owner/repo"), etc.
        store = SeenStore(tmp_db_path)
        tricky = ["github:owner/repo-name", "arxiv:2605.21384", "x:'; DROP TABLE seen_items; --"]
        store.mark_seen(tricky)
        assert store.filter_unseen(tricky) == set()
        # table must still exist and be queryable -- proves parameterized
        # queries actually protected us above, not just luck.
        assert store.count() == 3


class TestArcStatusStore:
    def test_upsert_then_read_back(self, tmp_db_path):
        store = ArcStatusStore(tmp_db_path)
        store.upsert(
            ArcStatus(project="Assay", status="GPU run not yet executed", next_action="Run GRPO training")
        )
        results = store.all()
        assert len(results) == 1
        assert results[0].project == "Assay"
        assert results[0].status == "GPU run not yet executed"

    def test_upsert_same_project_updates_not_duplicates(self, tmp_db_path):
        store = ArcStatusStore(tmp_db_path)
        store.upsert(ArcStatus(project="Assay", status="v1", next_action="a"))
        store.upsert(ArcStatus(project="Assay", status="v2", next_action="b"))
        results = store.all()
        assert len(results) == 1
        assert results[0].status == "v2"

    def test_multiple_projects_are_independent(self, tmp_db_path):
        store = ArcStatusStore(tmp_db_path)
        store.upsert(ArcStatus(project="Assay", status="s1", next_action="n1"))
        store.upsert(ArcStatus(project="Crucible", status="s2", next_action="n2"))
        results = {r.project: r for r in store.all()}
        assert set(results) == {"Assay", "Crucible"}

    def test_updated_at_round_trips_with_timezone(self, tmp_db_path):
        store = ArcStatusStore(tmp_db_path)
        ts = datetime(2026, 7, 13, 9, 30, tzinfo=timezone.utc)
        store.upsert(ArcStatus(project="Assay", status="s", next_action="n", updated_at=ts))
        results = store.all()
        assert results[0].updated_at == ts

    def test_empty_store_returns_empty_list(self, tmp_db_path):
        store = ArcStatusStore(tmp_db_path)
        assert store.all() == []

    def test_delete(self, tmp_db_path):
        store = ArcStatusStore(tmp_db_path)
        store.upsert(ArcStatus(project="Assay", status="s", next_action="n"))
        assert store.delete("Assay") is True
        assert store.delete("Assay") is False
        assert store.all() == []


class TestBriefStore:
    def test_save_get_latest_get_history(self, tmp_db_path):
        from signal_brief.models import BriefResponse
        from signal_brief.stores import BriefStore

        store = BriefStore(tmp_db_path)
        assert store.get_latest() is None
        assert store.get_history() == []

        now = datetime.now(timezone.utc)
        brief1 = BriefResponse(
            generated_at=now,
            calendar_events=[],
            top_items=[],
            arc_status=[],
            brief_text="Brief 1"
        )
        store.save(brief1)

        latest = store.get_latest()
        assert latest is not None
        assert latest.brief_text == "Brief 1"
        assert len(store.get_history()) == 1

