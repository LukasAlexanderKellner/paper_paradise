from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Item:
    """A single newsletter entry, independent of its source."""
    source: str                      # "arxiv" | "huggingface" | "github" | "blogs"
    title: str
    url: str
    key: str                         # normalized identity used for dedupe and state, e.g. "arxiv:2610.12448"
    summary: str = ""
    date: datetime | None = None
    score: float | None = None       # semantic relevance to the profile (arXiv, GitHub)
    meta: dict = field(default_factory=dict)  # source-specific fields (upvotes, stars, categories, ...)
    why: str | None = None           # optional LLM-written "why relevant"


class SourceError(Exception):
    """A source could not be fetched at all."""


class SourceStatusError(SourceError):
    """A source answered with a non-200 HTTP status."""

    def __init__(self, status: int):
        super().__init__(f"Status Code {status}")
        self.status = status
