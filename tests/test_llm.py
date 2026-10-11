import json
from types import SimpleNamespace

from core.config import load_settings
from pipeline.llm import annotate
from tests.conftest import make_item

ITEMS = [make_item("arxiv", "arxiv:1", summary="A"), make_item("blogs", "https://x/1")]


class FakeClient:
    def __init__(self, response=None, error=None):
        self.calls = []
        create = self._create
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))
        self.response, self.error = response, error

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def response(payload, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason, stop_details=None,
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
    )


def test_without_api_key_returns_nothing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = annotate(load_settings(), ITEMS)
    assert result.editorial is None and result.why == {}


def test_parses_editorial_and_ignores_unknown_keys():
    client = FakeClient(response({"editorial": "The week.", "items": [
        {"key": "arxiv:1", "why": "Fits."}, {"key": "unknown", "why": "x"}]}))
    result = annotate(load_settings(), ITEMS, client=client)
    assert result.editorial == "The week."
    assert result.why == {"arxiv:1": "Fits."}
    assert len(client.calls) == 1  # exactly one API call
    assert client.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_api_error_is_swallowed():
    result = annotate(load_settings(), ITEMS, client=FakeClient(error=RuntimeError("boom")))
    assert result.editorial is None and result.why == {}


def test_refusal_and_invalid_json_are_swallowed():
    assert annotate(load_settings(), ITEMS, client=FakeClient(response({}, stop_reason="refusal"))).editorial is None
    bad = SimpleNamespace(stop_reason="end_turn", stop_details=None, content=[SimpleNamespace(type="text", text="{nope")])
    assert annotate(load_settings(), ITEMS, client=FakeClient(bad)).editorial is None
