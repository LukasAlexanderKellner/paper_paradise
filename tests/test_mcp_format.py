"""Pins the MCP text format so the refactoring keeps Claude Desktop output unchanged."""
import importlib
import sys

from tests.conftest import make_item


def load_server(monkeypatch):
    # Avoid loading the embedding model on import
    monkeypatch.setattr("core.ranking.get_embedder", lambda: None)
    sys.modules.pop("paper_paradise_mcp", None)
    return importlib.import_module("paper_paradise_mcp")


def test_github_format(monkeypatch):
    server = load_server(monkeypatch)
    item = make_item("github", "https://github.com/a/b", title="a/b", url="https://github.com/a/b", score=0.442,
                     meta={"stars": 64, "created": "2026-10-07", "description": None, "since": "2026-10-04"})
    assert server.format_github([item]) == (
        "=== Top 1 new ML repositories on GitHub (since 2026-10-04) ===\n\n"
        "Repo: a/b\nRelevance score: 0.442\nStars: ⭐ 64\nCreated: 2026-10-07\n"
        "URL: https://github.com/a/b\nDescription: None\n" + "-" * 40
    )


def test_empty_messages(monkeypatch):
    server = load_server(monkeypatch)
    assert server.format_arxiv([]) == "No matching arXiv papers found after filtering."
    assert server.format_huggingface([]) == "No current Hugging Face papers found."
    assert server.format_github([]) == "No matching GitHub repositories found."
    assert server.format_blogs([]) == "No recent blog posts found."


def test_tool_error_messages(monkeypatch):
    server = load_server(monkeypatch)
    from core.models import SourceError, SourceStatusError

    def raise_(exc):
        def f(*a, **k):
            raise exc
        return f

    monkeypatch.setattr(server.sources, "fetch_huggingface", raise_(SourceStatusError(503)))
    assert server.fetch_huggingface_daily_papers() == "Error fetching Hugging Face: status code 503"
    monkeypatch.setattr(server.sources, "fetch_github", raise_(SourceError("pytorch: Status 403")))
    assert server.fetch_github_trending_ml_repos() == "GitHub API query failed: pytorch: Status 403"
    monkeypatch.setattr(server.sources, "fetch_arxiv", raise_(TimeoutError("read timed out")))
    assert server.fetch_filtered_arxiv_papers() == "arXiv query failed: TimeoutError: read timed out"


def test_excerpt_marks_only_truncated_text(monkeypatch):
    server = load_server(monkeypatch)
    short = make_item("arxiv", "arxiv:1", summary="Short abstract.", score=0.5, meta={"categories": ["cs.LG"]})
    long = make_item("arxiv", "arxiv:2", summary="x" * 500, score=0.5, meta={"categories": ["cs.LG"]})
    assert "Abstract (excerpt): Short abstract.\n" in server.format_arxiv([short])
    assert "Abstract (excerpt): " + "x" * 400 + "...\n" in server.format_arxiv([long])
