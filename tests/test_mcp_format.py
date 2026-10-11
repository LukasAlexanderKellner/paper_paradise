"""Pins the MCP text format so the refactoring keeps Claude Desktop output unchanged."""
import importlib
import sys

from tests.conftest import make_item


def load_server(monkeypatch):
    # Avoid loading the embedding model on import
    monkeypatch.setattr("core.ranking.get_embedder", lambda: None)
    sys.modules.pop("paperporn_mcp", None)
    return importlib.import_module("paperporn_mcp")


def test_github_format(monkeypatch):
    server = load_server(monkeypatch)
    item = make_item("github", "https://github.com/a/b", title="a/b", url="https://github.com/a/b", score=0.442,
                     meta={"stars": 64, "created": "2026-10-07", "description": None, "since": "2026-10-04"})
    assert server.format_github([item]) == (
        "=== Top 1 neue ML Repositories auf GitHub (seit 2026-10-04) ===\n\n"
        "Repo: a/b\nRelevanz-Score: 0.442\nStars: ⭐ 64\nErstellt: 2026-10-07\n"
        "URL: https://github.com/a/b\nBeschreibung: None\n" + "-" * 40
    )


def test_empty_messages(monkeypatch):
    server = load_server(monkeypatch)
    assert server.format_arxiv([]) == "Keine passenden arXiv Paper nach der Filterung gefunden."
    assert server.format_huggingface([]) == "Keine aktuellen Hugging Face Papers gefunden."
    assert server.format_github([]) == "Keine passenden GitHub Repositories gefunden."
    assert server.format_blogs([]) == "Keine aktuellen Blog-Beiträge gefunden."


def test_tool_error_messages(monkeypatch):
    server = load_server(monkeypatch)
    from core.models import SourceError, SourceStatusError

    def raise_(exc):
        def f(*a, **k):
            raise exc
        return f

    monkeypatch.setattr(server.sources, "fetch_huggingface", raise_(SourceStatusError(503)))
    assert server.fetch_huggingface_daily_papers() == "Fehler beim Abrufen von Hugging Face: Status Code 503"
    monkeypatch.setattr(server.sources, "fetch_github", raise_(SourceError("pytorch: Status 403")))
    assert server.fetch_github_trending_ml_repos() == "Fehler bei GitHub API Abfrage: pytorch: Status 403"
