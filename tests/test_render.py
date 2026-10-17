from datetime import datetime, timezone

from pipeline.render import NewsletterContext, render_newsletter

START = datetime(2026, 10, 5, tzinfo=timezone.utc)
END = datetime(2026, 10, 12, tzinfo=timezone.utc)


def render(sample_items, **kwargs):
    return render_newsletter(NewsletterContext(sections=sample_items, period_start=START, period_end=END, **kwargs))


def test_renders_all_sections_and_links(sample_items):
    html, text = render(sample_items)
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    for title in ["Hugging Face: community favorites", "arXiv: matching your profile",
                  "GitHub: new repositories", "Engineering blogs"]:
        assert title in html and title in text
    for url in ["https://github.com/a/b", "https://pytorch.org/blog/x", "https://arxiv.org/abs/2610.00001"]:
        assert url in html and url in text
    assert "2026-10-05 – 2026-10-12" in html
    assert "42 upvotes" in text


def test_html_is_escaped_but_text_is_not(sample_items):
    html, text = render(sample_items)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Abstract B <script>" in text


def test_empty_sections_are_skipped(sample_items):
    sample_items["github"] = []
    html, text = render(sample_items)
    assert "GitHub: new repositories" not in html
    assert "GitHub: new repositories" not in text


def test_failed_sources_are_listed(sample_items):
    html, text = render(sample_items, failed_sources=["github"])
    assert "Not available in this issue: GitHub" in html
    assert "Not available in this issue: GitHub" in text


def test_summary_block_and_why_are_optional(sample_items):
    html, text = render(sample_items)
    assert "Why it matters" not in html
    assert "This week's highlight" not in html and "#eff6ff" not in html  # no empty summary box

    sample_items["github"][0].why = "Fits GPU optimization."
    repo = sample_items["github"][0]
    html, text = render(
        sample_items,
        headline="This week's highlight: a/b.",
        summary="Lots of RL this week.",
        highlights=[(repo, "Cuts GPU memory in half.")],
    )
    assert "This week&#39;s highlight: a/b." in html  # apostrophe is HTML-escaped
    assert "This week's highlight: a/b." in text
    for content in (html, text):
        assert "Lots of RL this week." in content
        assert "Cuts GPU memory in half." in content
    assert '<a href="https://github.com/a/b"' in html.split("Lots of RL this week.")[1].split("</ul>")[0]
    assert "- a/b: Cuts GPU memory in half.\n  https://github.com/a/b" in text
    # Summary comes before the first section
    assert html.index("Lots of RL this week.") < html.index("Hugging Face: community favorites")
    assert "Fits GPU optimization." in html and "Why it matters: Fits GPU optimization." in text


def test_glossary_is_rendered_first_in_the_requested_format(sample_items):
    from pipeline.llm import GlossaryEntry

    glossary = [GlossaryEntry(term="KV cache", translation="KV-Cache", explanation="Zwischenspeicher <b>für</b> Attention.")]
    html, text = render(sample_items, glossary=glossary, summary="Lots of RL this week.")
    assert "<strong>KV cache</strong> = KV-Cache: Zwischenspeicher &lt;b&gt;für&lt;/b&gt; Attention." in html
    assert "- KV cache = KV-Cache: Zwischenspeicher <b>für</b> Attention.\n" in text
    # Glossary is the first block, before the summary
    assert html.index("Glossary") < html.index("Lots of RL this week.")
    assert text.index("Glossary") < text.index("Lots of RL this week.")


def test_no_glossary_block_without_entries(sample_items):
    html, text = render(sample_items)
    assert "Glossary" not in html and "Glossary" not in text


def test_summary_text_is_html_escaped(sample_items):
    html, _ = render(sample_items, summary="<script>alert(1)</script>")
    assert "<script>alert" not in html and "&lt;script&gt;" in html


def test_long_summaries_are_truncated(sample_items):
    sample_items["blogs"][0].summary = "word " * 200
    html, _ = render(sample_items)
    assert "word " * 100 not in html and "…" in html


def test_blog_relevance_is_shown_when_scored(sample_items):
    sample_items["blogs"][0].score = 0.512
    html, text = render(sample_items)
    assert "Blog - PyTorch · relevance 0.51" in html
    assert "  Blog - PyTorch, relevance 0.51\n" in text


def test_ai_notice_only_when_ai_text_is_included(sample_items):
    html, text = render(sample_items)
    assert "AI-generated" not in html and "AI-generated" not in text
    html, text = render(sample_items, ai_notice=True)
    assert "AI-generated (Google Gemini)" in html and "AI-generated (Google Gemini)" in text
