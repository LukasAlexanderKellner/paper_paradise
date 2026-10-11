from datetime import datetime, timezone

from pipeline.render import NewsletterContext, render_newsletter

START = datetime(2026, 10, 5, tzinfo=timezone.utc)
END = datetime(2026, 10, 12, tzinfo=timezone.utc)


def render(sample_items, **kwargs):
    return render_newsletter(NewsletterContext(sections=sample_items, period_start=START, period_end=END, **kwargs))


def test_renders_all_sections_and_links(sample_items):
    html, text = render(sample_items)
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    for title in ["Hugging Face: Community-Favoriten", "arXiv: passend zu deinem Profil",
                  "GitHub: neue Repositories", "Engineering-Blogs"]:
        assert title in html and title in text
    for url in ["https://github.com/a/b", "https://pytorch.org/blog/x", "https://arxiv.org/abs/2610.00001"]:
        assert url in html and url in text
    assert "05.10.2026 – 12.10.2026" in html
    assert "42 Upvotes" in text


def test_html_is_escaped_but_text_is_not(sample_items):
    html, text = render(sample_items)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Abstract B <script>" in text


def test_empty_sections_are_skipped(sample_items):
    sample_items["github"] = []
    html, text = render(sample_items)
    assert "GitHub: neue Repositories" not in html
    assert "GitHub: neue Repositories" not in text


def test_failed_sources_are_listed(sample_items):
    html, text = render(sample_items, failed_sources=["github"])
    assert "Nicht verfügbar in dieser Ausgabe: GitHub" in html
    assert "Nicht verfügbar in dieser Ausgabe: GitHub" in text


def test_editorial_and_why_are_optional(sample_items):
    html, _ = render(sample_items)
    assert "Warum relevant" not in html

    sample_items["github"][0].why = "Passt zu GPU-Optimierung."
    html, text = render(sample_items, editorial="Diese Woche: viel RL.")
    assert "Diese Woche: viel RL." in html and "Diese Woche: viel RL." in text
    assert "Passt zu GPU-Optimierung." in html and "Warum relevant: Passt zu GPU-Optimierung." in text


def test_long_summaries_are_truncated(sample_items):
    sample_items["blogs"][0].summary = "wort " * 200
    html, _ = render(sample_items)
    assert "wort " * 100 not in html and "…" in html
