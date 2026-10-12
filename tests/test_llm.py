import json
from types import SimpleNamespace

from core.config import load_settings
from pipeline.llm import annotate
from tests.conftest import make_item

ITEMS = [make_item("arxiv", "arxiv:1", summary="A"), make_item("blogs", "https://x/1")]

DIGEST = {
    "headline": "This week's highlight: arXiv 1.",
    "summary": "Sentence one. Sentence two.",
    "highlights": [{"key": "arxiv:1", "takeaway": "A new finding."}, {"key": "unknown", "takeaway": "x"}],
    "notes": [{"key": "arxiv:1", "why": "Fits."}, {"key": "unknown", "why": "x"}],
}


class FakeGemini:
    """Mimics client.models.generate_content of the google-genai SDK."""

    def __init__(self, text=None, error=None):
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)
        self.text, self.error = text, error

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text, prompt_feedback=None)


def test_without_api_key_returns_nothing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = annotate(load_settings(), ITEMS)
    assert result.headline is None and result.summary is None and result.highlights == [] and result.why == {}


def test_parses_digest_and_ignores_unknown_keys():
    client = FakeGemini(json.dumps(DIGEST))
    result = annotate(load_settings(), ITEMS, client=client)
    assert result.headline == "This week's highlight: arXiv 1."
    assert result.summary == "Sentence one. Sentence two."
    assert [(item.key, takeaway) for item, takeaway in result.highlights] == [("arxiv:1", "A new finding.")]
    assert result.why == {"arxiv:1": "Fits."}
    assert len(client.calls) == 1  # exactly one API call
    config = client.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert "highlights" in config.response_json_schema["properties"]
    assert client.calls[0]["model"] == load_settings().llm.model


def test_api_error_is_swallowed():
    result = annotate(load_settings(), ITEMS, client=FakeGemini(error=RuntimeError("boom")))
    assert result.headline is None and result.highlights == []


def test_empty_or_invalid_response_is_swallowed():
    assert annotate(load_settings(), ITEMS, client=FakeGemini(text=None)).summary is None
    assert annotate(load_settings(), ITEMS, client=FakeGemini(text="{nope")).summary is None
    assert annotate(load_settings(), ITEMS, client=FakeGemini(text=json.dumps({"headline": "x"}))).summary is None
