import json
import logging
import os
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from core.config import Settings
from core.models import Item

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You write the opening of a weekly AI/ML newsletter for a single reader. "
    "Write in English, factual and concrete, without marketing phrases. "
    "Use only the provided items and never invent results, numbers or details "
    "that are not in an item's title or summary."
)


class Highlight(BaseModel):
    key: str = Field(description="key of the item, exactly as given")
    takeaway: str = Field(description="1-2 sentences: the finding and why it is exciting")


class ItemNote(BaseModel):
    key: str = Field(description="key of the item, exactly as given")
    why: str = Field(description="1-2 sentences on why the item is relevant to the reader's profile")


class Digest(BaseModel):
    headline: str = Field(description='one sentence starting with "This week\'s highlight:"')
    summary: str = Field(description="5-6 sentences on the most exciting findings, connecting the themes")
    highlights: list[Highlight] = Field(description="the 3-5 most exciting items")
    notes: list[ItemNote] = Field(description="one note for every item")


@dataclass
class Annotations:
    headline: str | None = None
    summary: str | None = None
    highlights: list[tuple[Item, str]] = field(default_factory=list)
    why: dict[str, str] = field(default_factory=dict)


def _build_prompt(profile: str, items: list[Item]) -> str:
    entries = [
        {
            "key": item.key,
            "source": item.source,
            "title": item.title,
            "summary": " ".join((item.summary or "").split())[:600],
        }
        for item in items
    ]
    return (
        f"Reader profile:\n{profile}\n\n"
        f"This week's items (JSON):\n{json.dumps(entries, ensure_ascii=False, indent=1)}\n\n"
        "Tasks:\n"
        '1. headline: one sentence starting with "This week\'s highlight:" that names the single most exciting item.\n'
        "2. summary: 5 to 6 sentences on the most exciting findings across all items, connecting the main themes.\n"
        "3. highlights: the 3 to 5 most exciting items (same key), each with a 1-2 sentence takeaway of its finding.\n"
        "4. notes: for every item (same key) 1 to 2 sentences on why it is relevant to this profile."
    )


def annotate(settings: Settings, items: list[Item], client=None) -> Annotations:
    """
    One Gemini call that writes a headline, a short summary of the week's most exciting findings,
    highlight takeaways and a "why it matters" per item.
    Returns empty Annotations if no GEMINI_API_KEY is set or anything goes wrong.
    """
    if not settings.llm.enabled or not items:
        return Annotations()
    if client is None and not os.environ.get("GEMINI_API_KEY"):
        logger.info("GEMINI_API_KEY not set, sending the newsletter without summary.")
        return Annotations()

    try:
        from google import genai
        from google.genai import types

        client = client or genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=180_000),  # milliseconds
        )
        response = client.models.generate_content(
            model=settings.llm.model,
            contents=_build_prompt(settings.profile, items),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=Digest.model_json_schema(),
            ),
        )
        if not response.text:
            logger.warning("Gemini returned no text (%s), sending the newsletter without summary.",
                           response.prompt_feedback)
            return Annotations()

        digest = Digest.model_validate_json(response.text)
        by_key = {item.key: item for item in items}
        return Annotations(
            headline=digest.headline.strip() or None,
            summary=digest.summary.strip() or None,
            highlights=[(by_key[h.key], h.takeaway.strip()) for h in digest.highlights
                        if h.key in by_key and h.takeaway.strip()],
            why={n.key: n.why.strip() for n in digest.notes if n.key in by_key and n.why.strip()},
        )

    except Exception as e:
        # The newsletter must go out regardless; log the reason and continue without annotations
        logger.warning("LLM step failed (%s: %s), sending the newsletter without summary.", type(e).__name__, e)
        return Annotations()
