from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from core.models import Item

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

SECTION_TITLES = {
    "huggingface": "Hugging Face: community favorites",
    "arxiv": "arXiv: matching your profile",
    "github": "GitHub: new repositories",
    "blogs": "Engineering blogs",
}
SOURCE_NAMES = {
    "huggingface": "Hugging Face Daily Papers",
    "arxiv": "arXiv",
    "github": "GitHub",
    "blogs": "Engineering blogs",
}


@dataclass
class NewsletterContext:
    sections: dict[str, list[Item]]
    period_start: datetime
    period_end: datetime
    failed_sources: list[str] = field(default_factory=list)
    headline: str | None = None                                     # LLM: "This week's highlight: ..."
    summary: str | None = None                                      # LLM: 5-6 sentences
    highlights: list[tuple[Item, str]] = field(default_factory=list)  # LLM: (item, takeaway)
    glossary: list = field(default_factory=list)                     # LLM: GlossaryEntry(term, translation, explanation)


def _truncate(text: str | None, length: int = 320) -> str:
    text = " ".join((text or "").split())
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + " …"


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(enabled_extensions=("html.j2",), default_for_string=False),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["truncate_words"] = _truncate
    return env


def render_newsletter(ctx: NewsletterContext) -> tuple[str, str]:
    """Render the newsletter as (html, plaintext)."""
    env = _environment()
    variables = {
        "sections": [
            {"source": source, "title": SECTION_TITLES.get(source, source), "items": items}
            for source, items in ctx.sections.items()
            if items
        ],
        "period_start": ctx.period_start,
        "period_end": ctx.period_end,
        "failed_sources": [SOURCE_NAMES.get(s, s) for s in ctx.failed_sources],
        "headline": ctx.headline,
        "glossary": ctx.glossary,
        "summary": ctx.summary,
        "highlights": [{"item": item, "takeaway": takeaway} for item, takeaway in ctx.highlights],
    }
    html = env.get_template("newsletter.html.j2").render(**variables)
    text = env.get_template("newsletter.txt.j2").render(**variables)
    return html, text
