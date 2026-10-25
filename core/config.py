import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.toml"


@dataclass(frozen=True)
class NewsletterSettings:
    window_days: int
    subject: str
    sender: str
    delivery: str                  # "smtp" or "brevo"
    imprint_url: str
    state_file: Path
    state_retention_days: int
    top_k: dict[str, int]
    arxiv_max_fetch: int
    arxiv_lag_days: int
    github_min_stars: int
    github_max_fetch: int


@dataclass(frozen=True)
class LLMSettings:
    enabled: bool
    model: str
    glossary_known_terms: list[str]


@dataclass(frozen=True)
class BrevoSettings:
    list_ids: list[int]
    sender_email: str
    sender_name: str


@dataclass(frozen=True)
class Settings:
    profile: str
    exclude_keywords: list[str]
    arxiv_categories: list[str]
    github_topics: list[str]
    blog_feeds: list[str]
    newsletter: NewsletterSettings
    llm: LLMSettings
    brevo: BrevoSettings
    raw: dict = field(repr=False, default_factory=dict)


def load_settings(path: str | Path | None = None) -> Settings:
    """Load config.toml (or PAPER_PARADISE_CONFIG) into typed settings."""
    path = Path(path or os.environ.get("PAPER_PARADISE_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    profile = raw["profile"]
    sources = raw["sources"]
    nl = raw.get("newsletter", {})
    llm = raw.get("llm", {})
    brevo = raw.get("brevo", {})

    state_file = Path(nl.get("state_file", "state/sent_items.json"))
    if not state_file.is_absolute():
        state_file = ROOT_DIR / state_file

    return Settings(
        profile=profile["description"].strip(),
        exclude_keywords=list(profile.get("exclude_keywords", [])),
        arxiv_categories=list(sources["arxiv_categories"]),
        github_topics=list(sources["github_topics"]),
        blog_feeds=list(sources["blog_feeds"]),
        newsletter=NewsletterSettings(
            window_days=nl.get("window_days", 7),
            subject=nl.get("subject", "paper_paradise Weekly"),
            sender=nl.get("sender", ""),
            delivery=nl.get("delivery", "smtp"),
            imprint_url=nl.get("imprint_url", ""),
            state_file=state_file,
            state_retention_days=nl.get("state_retention_days", 180),
            top_k={
                "arxiv": 5, "huggingface": 5, "github": 5, "blogs": 5,
                **nl.get("top_k", {}),
            },
            arxiv_max_fetch=nl.get("arxiv", {}).get("max_fetch", 5000),
            arxiv_lag_days=nl.get("arxiv", {}).get("lag_days", 3),
            github_min_stars=nl.get("github", {}).get("min_stars", 20),
            github_max_fetch=nl.get("github", {}).get("max_fetch", 30),
        ),
        llm=LLMSettings(
            enabled=llm.get("enabled", True),
            model=llm.get("model", "gemini-3.8-flash"),
            glossary_known_terms=list(llm.get("glossary_known_terms", [])),
        ),
        brevo=BrevoSettings(
            list_ids=[int(i) for i in brevo.get("list_ids", [])],
            sender_email=brevo.get("sender_email", ""),
            sender_name=brevo.get("sender_name", ""),
        ),
        raw=raw,
    )
