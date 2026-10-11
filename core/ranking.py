import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedder():
    """Load the embedding model once, on first use."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading semantic embedding model...")
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=4)
def _profile_embedding(profile: str):
    return get_embedder().encode(profile, convert_to_tensor=True)


def similarity_scores(profile: str, texts: list[str]) -> list[float]:
    """Cosine similarity of each text to the profile, rounded to 3 decimals."""
    if not texts:
        return []
    from sentence_transformers import util

    embeddings = get_embedder().encode(texts, convert_to_tensor=True)
    similarities = util.cos_sim(_profile_embedding(profile), embeddings)[0].tolist()
    return [round(s, 3) for s in similarities]


def is_excluded(keywords: list[str], *texts: str | None) -> bool:
    """Hard negative filter: True if any keyword appears in the given texts."""
    combined = " ".join(t or "" for t in texts).lower()
    return any(neg.lower() in combined for neg in keywords)
