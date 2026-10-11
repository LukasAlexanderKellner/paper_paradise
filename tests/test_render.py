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


def test_editorial_and_why_are_optional(sample_items):
    html, _ = render(sample_items)
    assert "Why it matters" not in html

    sample_items["github"][0].why = "Fits GPU optimization."
    html, text = render(sample_items, editorial="This week: lots of RL.")
    assert "This week: lots of RL." in html and "This week: lots of RL." in text
    assert "Fits GPU optimization." in html and "Why it matters: Fits GPU optimization." in text


def test_long_summaries_are_truncated(sample_items):
    sample_items["blogs"][0].summary = "word " * 200
    html, _ = render(sample_items)
    assert "word " * 100 not in html and "…" in html
