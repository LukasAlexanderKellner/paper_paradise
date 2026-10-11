import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Item  # noqa: E402


def make_item(source: str, key: str, title: str | None = None, **kwargs) -> Item:
    return Item(
        source=source,
        title=title or f"{source} {key}",
        url=kwargs.pop("url", f"https://example.com/{key}"),
        key=key,
        **kwargs,
    )


@pytest.fixture
def sample_items() -> dict[str, list[Item]]:
    """Example data for all four sources, including one paper listed on both HF and arXiv."""
    date = datetime(2026, 10, 8, tzinfo=timezone.utc)
    return {
        "arxiv": [
            make_item("arxiv", "arxiv:2610.00001", title="Shared Paper", url="https://arxiv.org/pdf/2610.00001v1",
                      summary="Abstract A", date=date, score=0.61, meta={"categories": ["cs.LG"]}),
            make_item("arxiv", "arxiv:2610.00002", title="Only on arXiv", summary="Abstract B <script>",
                      date=date, score=0.55, meta={"categories": ["cs.DC"]}),
        ],
        "huggingface": [
            make_item("huggingface", "arxiv:2610.00001", title="Shared Paper", url="https://arxiv.org/abs/2610.00001",
                      summary="Abstract A", date=date, meta={"upvotes": 42}),
        ],
        "github": [
            make_item("github", "https://github.com/a/b", title="a/b", url="https://github.com/a/b",
                      summary="Repo", date=date, score=0.44, meta={"stars": 120}),
        ],
        "blogs": [
            make_item("blogs", "https://pytorch.org/blog/x", title="Blog X", url="https://pytorch.org/blog/x",
                      date=datetime(2026, 10, 7), meta={"feed": "Blog - PyTorch"}),
        ],
    }
