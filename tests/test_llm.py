import json
from types import SimpleNamespace

from core.config import load_settings
from pipeline.llm import annotate
from tests.conftest import make_item

ITEMS = [
    make_item("arxiv", "arxiv:1", title="On-Policy Distillation with RLVR", summary="Uses KV cache quantization."),
    make_item("blogs", "https://x/1", summary="The team rebuilt the GPU scheduler from scratch and cut average queue wait times by 87 percent across the cluster."),
    make_item("blogs", "https://x/2", summary="Short teaser about distributed training."),
]

DIGEST = {
    "headline": "This week's highlight: arXiv 1.",
    "summary": "Sentence one. Sentence two.",
    "highlights": [{"key": "arxiv:1", "takeaway": "A new finding."}, {"key": "unknown", "takeaway": "x"}],
    "notes": [{"key": "arxiv:1", "why": "Fits."}, {"key": "unknown", "why": "x"}],
    "rewrites": [
        {"key": "https://x/1", "summary": "A rebuilt scheduler makes jobs wait far less on shared GPUs."},
        {"key": "https://x/2", "summary": "The team rebuilt the GPU scheduler from scratch and cut average queue wait times by 87 percent across the cluster."},
        {"key": "arxiv:1", "summary": "Not a blog item, must be ignored."},
    ],
    "glossary": [
        {"term": "On-Policy Distillation", "translation": "On-Policy-Destillation",
         "explanation": "Ein Schüler-Modell lernt aus eigenen Ausgaben."},
        {"term": "RLVR", "translation": "RLVR", "explanation": "Verstärkendes Lernen mit prüfbaren Belohnungen."},
        {"term": "rlvr", "translation": "RLVR", "explanation": "Duplikat."},
        {"term": "Hallucinated Method", "translation": "x", "explanation": "Steht nirgends im Text."},
        {"term": "KV cache", "translation": "", "explanation": "Zwischenspeicher für Attention-Werte."},
    ],
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
    assert [(g.term, g.translation) for g in result.glossary] == [
        ("On-Policy Distillation", "On-Policy-Destillation"), ("RLVR", "RLVR"), ("KV cache", "KV cache")]


def test_prompt_lists_known_terms_and_german_glossary():
    client = FakeGemini(json.dumps(DIGEST))
    annotate(load_settings(), ITEMS, client=client)
    prompt = client.calls[0]["contents"]
    for known in load_settings().llm.glossary_known_terms:
        assert known in prompt
    assert "German" in prompt and "glossary" in prompt


def test_glossary_is_capped():
    digest = dict(DIGEST, glossary=[
        {"term": f"term{i}", "translation": "t", "explanation": "e"} for i in range(30)])
    items = [make_item("arxiv", "arxiv:1", summary=" ".join(f"term{i}" for i in range(30)))]
    result = annotate(load_settings(), items, client=FakeGemini(json.dumps(digest)))
    assert len(result.glossary) == 20


def test_api_error_is_swallowed():
    result = annotate(load_settings(), ITEMS, client=FakeGemini(error=RuntimeError("boom")))
    assert result.headline is None and result.highlights == []


def test_empty_or_invalid_response_is_swallowed():
    assert annotate(load_settings(), ITEMS, client=FakeGemini(text=None)).summary is None
    assert annotate(load_settings(), ITEMS, client=FakeGemini(text="{nope")).summary is None
    assert annotate(load_settings(), ITEMS, client=FakeGemini(text=json.dumps({"headline": "x"}))).summary is None


def test_blog_rewrites_are_kept_unless_copied():
    result = annotate(load_settings(), ITEMS, client=FakeGemini(json.dumps(DIGEST)))
    assert result.generated is True
    # x/1 is in own words -> kept; arxiv:1 is not a blog item -> ignored
    assert result.rewrites["https://x/1"] == "A rebuilt scheduler makes jobs wait far less on shared GPUs."
    assert "arxiv:1" not in result.rewrites


def test_rewrite_copying_the_original_is_dropped():
    digest = dict(DIGEST, rewrites=[{"key": "https://x/1", "summary": "In short: " + ITEMS[1].summary}])
    result = annotate(load_settings(), ITEMS, client=FakeGemini(json.dumps(digest)))
    assert "https://x/1" not in result.rewrites


def test_copy_detection_threshold():
    from pipeline.llm import _copies_original
    original = "one two three four five six seven eight nine ten"
    assert _copies_original("Intro: one two three four five six seven eight, done.", original)
    assert not _copies_original("Intro: one two three four five six seven, done.", original)


def test_prompt_asks_for_own_words_blog_summaries():
    client = FakeGemini(json.dumps(DIGEST))
    annotate(load_settings(), ITEMS, client=client)
    prompt = client.calls[0]["contents"]
    assert "own words" in prompt and "Do not copy sentences or phrases" in prompt


def test_no_rewrite_for_blog_without_teaser():
    items = ITEMS + [make_item("blogs", "https://x/empty", summary="")]
    digest = dict(DIGEST, rewrites=DIGEST["rewrites"] + [{"key": "https://x/empty", "summary": "Invented from title."}])
    result = annotate(load_settings(), items, client=FakeGemini(json.dumps(digest)))
    assert "https://x/empty" not in result.rewrites
