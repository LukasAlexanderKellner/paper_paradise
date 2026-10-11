import json
import logging
import os
from dataclasses import dataclass, field

from core.config import Settings
from core.models import Item

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You write a weekly AI/ML newsletter for a single reader. "
    "Write in English, factual and concrete, without marketing phrases. "
    "Refer only to the provided items and do not invent details that are not in the title or summary."
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "editorial": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "why": {"type": "string"}},
                "required": ["key", "why"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["editorial", "items"],
    "additionalProperties": False,
}


@dataclass
class Annotations:
    editorial: str | None = None
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
        "1. editorial: 3 to 5 sentences that put the week in context and connect the main themes.\n"
        "2. items: for every item (same key) 1 to 2 sentences on why it is relevant to this profile."
    )


def annotate(settings: Settings, items: list[Item], client=None) -> Annotations:
    """
    One Claude API call that writes an editorial and a short "why relevant" per item.
    Returns empty Annotations if no API key is set or anything goes wrong.
    """
    if not settings.llm.enabled or not items:
        return Annotations()
    if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("ANTHROPIC_API_KEY not set, sending the newsletter without editorial.")
        return Annotations()

    try:
        import anthropic

        client = client or anthropic.Anthropic(timeout=180.0, max_retries=2)
        response = client.beta.messages.create(
            model=settings.llm.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_prompt(settings.profile, items)}],
            output_config={
                "effort": settings.llm.effort,
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
            # On a safety decline the API retries on Anthropic's recommended fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            logger.warning("LLM step refused (%s), sending the newsletter without editorial.", response.stop_details)
            return Annotations()
        if response.stop_reason == "max_tokens":
            logger.warning("LLM response truncated, sending the newsletter without editorial.")
            return Annotations()

        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        known_keys = {item.key for item in items}
        why = {e["key"]: e["why"].strip() for e in data["items"] if e["key"] in known_keys and e["why"].strip()}
        return Annotations(editorial=data["editorial"].strip() or None, why=why)

    except Exception as e:
        # The newsletter must go out regardless; log the reason and continue without annotations
        logger.warning("LLM step failed (%s: %s), sending the newsletter without editorial.", type(e).__name__, e)
        return Annotations()
