"""fetch_arxiv and fetch_huggingface with mocked clients (no network)."""
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core import sources
from core.config import load_settings


@pytest.fixture(autouse=True)
def fake_scores(monkeypatch):
    monkeypatch.setattr(sources, "similarity_scores", lambda profile, texts: [0.5] * len(texts))


def fake_arxiv_client(monkeypatch, n):
    papers = [
        SimpleNamespace(title=f"Paper {i}", summary="Abstract", pdf_url=f"https://arxiv.org/pdf/2610.{i:05d}v1",
                        entry_id=f"https://arxiv.org/abs/2610.{i:05d}v1", published=datetime(2026, 10, 8, tzinfo=timezone.utc),
                        categories=["cs.LG"])
        for i in range(n)
    ]

    class Client:
        def __init__(self, **kwargs):
            pass

        def results(self, search):
            return papers[: search.max_results]

    monkeypatch.setattr(sources.arxiv, "Client", Client)


@pytest.mark.parametrize("available, warned", [(10, True), (9, False)])
def test_reaching_arxiv_max_fetch_is_logged(monkeypatch, caplog, available, warned):
    fake_arxiv_client(monkeypatch, available)
    since = datetime(2026, 10, 5, tzinfo=timezone.utc)
    with caplog.at_level(logging.WARNING, logger="core.sources"):
        items = sources.fetch_arxiv(load_settings(), max_fetch=10, top_k=3, since=since)
    assert len(items) == 3
    assert ("max_fetch (10) reached" in caplog.text) is warned

