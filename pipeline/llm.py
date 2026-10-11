import json
import logging
import os
from dataclasses import dataclass, field

from core.config import Settings
from core.models import Item

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Du schreibst einen wöchentlichen KI/ML-Newsletter für eine einzelne Leserin bzw. einen einzelnen Leser. "
    "Schreibe auf Deutsch, sachlich und konkret, ohne Marketing-Floskeln. "
    "Beziehe dich nur auf die gelieferten Einträge und erfinde keine Details, die nicht im Titel oder Kurztext stehen."
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
        f"Profil der Leserin bzw. des Lesers:\n{profile}\n\n"
        f"Einträge dieser Woche (JSON):\n{json.dumps(entries, ensure_ascii=False, indent=1)}\n\n"
        "Aufgaben:\n"
        "1. editorial: 3 bis 5 Sätze, die die Woche einordnen und die wichtigsten Themen verbinden.\n"
        "2. items: für jeden Eintrag (gleicher key) 1 bis 2 Sätze, warum er für dieses Profil relevant ist."
    )


def annotate(settings: Settings, items: list[Item], client=None) -> Annotations:
    """
    One Claude API call that writes an editorial and a short "why relevant" per item.
    Returns empty Annotations if no API key is set or anything goes wrong.
    """
    if not settings.llm.enabled or not items:
        return Annotations()
    if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("ANTHROPIC_API_KEY nicht gesetzt, Newsletter geht ohne Editorial raus.")
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
            logger.warning("LLM-Schritt abgelehnt (%s), Newsletter geht ohne Editorial raus.", response.stop_details)
            return Annotations()
        if response.stop_reason == "max_tokens":
            logger.warning("LLM-Antwort abgeschnitten, Newsletter geht ohne Editorial raus.")
            return Annotations()

        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        known_keys = {item.key for item in items}
        why = {e["key"]: e["why"].strip() for e in data["items"] if e["key"] in known_keys and e["why"].strip()}
        return Annotations(editorial=data["editorial"].strip() or None, why=why)

    except Exception as e:
        # The newsletter must go out regardless; log the reason and continue without annotations
        logger.warning("LLM-Schritt fehlgeschlagen (%s: %s), Newsletter geht ohne Editorial raus.", type(e).__name__, e)
        return Annotations()
