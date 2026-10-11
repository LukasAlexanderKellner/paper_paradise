from core.models import Item

# When the same entry appears in several sources, the first source in this order keeps it.
# Hugging Face wins over arXiv because it carries community upvotes.
SOURCE_PRIORITY = ["huggingface", "arxiv", "github", "blogs"]


def dedupe(items_by_source: dict[str, list[Item]]) -> dict[str, list[Item]]:
    """Remove duplicates within and across sources, keyed by Item.key (arXiv ID or URL)."""
    seen: set[str] = set()
    result: dict[str, list[Item]] = {}
    ordered = sorted(
        items_by_source,
        key=lambda s: SOURCE_PRIORITY.index(s) if s in SOURCE_PRIORITY else len(SOURCE_PRIORITY),
    )
    for source in ordered:
        kept = []
        for item in items_by_source[source]:
            if item.key in seen:
                continue
            seen.add(item.key)
            kept.append(item)
        result[source] = kept
    # Preserve the caller's source order
    return {source: result[source] for source in items_by_source}
