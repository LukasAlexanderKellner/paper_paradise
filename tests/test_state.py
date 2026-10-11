import json
from datetime import datetime, timezone

from pipeline.state import filter_unsent, load_state, mark_sent, save_state
from tests.conftest import make_item

NOW = datetime(2026, 10, 12, 5, 0, tzinfo=timezone.utc)


def test_filter_unsent_removes_already_sent(sample_items):
    state = {"arxiv:2610.00002": "2026-10-05T05:00:00+00:00", "https://github.com/a/b": "2026-10-05T05:00:00+00:00"}
    result = filter_unsent(sample_items, state)
    assert [i.key for i in result["arxiv"]] == ["arxiv:2610.00001"]
    assert result["github"] == []
    assert len(result["blogs"]) == 1


def test_mark_sent_adds_new_and_keeps_first_timestamp():
    state = {"a": "2026-10-01T05:00:00+00:00"}
    new = mark_sent(state, [make_item("blogs", "a"), make_item("blogs", "b")], retention_days=180, now=NOW)
    assert new["a"] == "2026-10-01T05:00:00+00:00"
    assert new["b"] == NOW.isoformat(timespec="seconds")
    assert state == {"a": "2026-10-01T05:00:00+00:00"}  # original untouched


def test_mark_sent_prunes_old_entries():
    state = {"old": "2026-01-01T05:00:00+00:00", "recent": "2026-10-01T05:00:00+00:00"}
    new = mark_sent(state, [], retention_days=180, now=NOW)
    assert set(new) == {"recent"}


def test_load_missing_file_returns_empty(tmp_path):
    assert load_state(tmp_path / "missing.json") == {}


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "sub" / "state.json"
    save_state(path, {"b": "2026-10-01T05:00:00+00:00", "a": "2026-10-02T05:00:00+00:00"})
    assert load_state(path) == {"a": "2026-10-02T05:00:00+00:00", "b": "2026-10-01T05:00:00+00:00"}
    assert list(json.loads(path.read_text())) == ["a", "b"]  # sorted for stable git diffs
    assert not list(path.parent.glob("*.tmp"))
