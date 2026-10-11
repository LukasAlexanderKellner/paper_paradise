import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.models import Item

# The state maps Item.key -> ISO timestamp of the first send.


def load_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def filter_unsent(items_by_source: dict[str, list[Item]], state: dict[str, str]) -> dict[str, list[Item]]:
    """Drop items that were already sent in an earlier newsletter."""
    return {
        source: [item for item in items if item.key not in state]
        for source, items in items_by_source.items()
    }


def mark_sent(
    state: dict[str, str],
    items: list[Item],
    retention_days: int,
    now: datetime | None = None,
) -> dict[str, str]:
    """Return a new state with the given items added and entries older than retention_days removed."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=retention_days)
    new_state = {
        key: sent_at for key, sent_at in state.items()
        if datetime.fromisoformat(sent_at) >= cutoff
    }
    for item in items:
        new_state.setdefault(item.key, now.isoformat(timespec="seconds"))
    return new_state


def save_state(path: Path, state: dict[str, str]) -> None:
    """Write the state atomically, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(state.items())), f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
