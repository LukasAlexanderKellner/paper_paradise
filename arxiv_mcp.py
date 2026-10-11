from fastmcp import FastMCP
import arxiv
from sentence_transformers import SentenceTransformer, util
import numpy as np
import config_module

# Load configs
cfg = config_module.Config()
STUDENT_PROFILE = cfg.STUDENT_PROFILE
EXCLUDE_KEYWORDS = cfg.EXCLUDE_KEYWORDS
PORN_CATEGORIES = cfg.PORN_CATEGORIES

# initialize FastMCP Server
mcp = FastMCP("Smart arXiv Paper Agent")
arxiv_client = arxiv.Client()


# Load small, fast Embedding-Model 
print("Lade Semantic-Embedding-Modell...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
profile_embedding = embedder.encode(STUDENT_PROFILE, convert_to_tensor=True)


@mcp.tool()
def fetch_filtered_papers(
    categories: list[str] = PORN_CATEGORIES,
    max_fetch: int = 30,
    top_k: int = 5
) -> str:
    """
    Searches for new arXiv papers, filters them semantically against the learning profile, 
    and returns only the most relevant papers to save tokens.
    """
    query_str = " OR ".join([f"cat:{cat}" for cat in categories])
    search = arxiv.Search(
        query=query_str,
        max_results=max_fetch,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    filtered_results = []
    raw_papers = list(arxiv_client.results(search))

    for paper in raw_papers:
        title_abstract = f"{paper.title} {paper.summary}".lower()
        
        # 1. Hard Rule: apply negative filter
        if any(neg in title_abstract for neg in EXCLUDE_KEYWORDS):
            continue
        
        # 2. Calculate Semantic Search Score 
        paper_embedding = embedder.encode(paper.summary, convert_to_tensor=True)
        similarity = util.cos_sim(profile_embedding, paper_embedding).item()
        
        filtered_results.append({
            "title": paper.title,
            "url": paper.pdf_url,
            "categories": paper.categories,
            "score": round(similarity, 3),
            "summary": paper.summary[:400] + "..."  # Shortened for minimal tokens
        })

    # Sort by similarity and select top-K
    filtered_results.sort(key=lambda x: x["score"], reverse=True)
    top_papers = filtered_results[:top_k]

    if not top_papers:
        return "Keine passenden Paper nach der Filterung gefunden."

    # Prompt-freundliches Ergebnis aufbereiten
    output = [f"=== Top {len(top_papers)} relevante Paper für dein Profil ==="]
    for p in top_papers:
        output.append(
            f"Titel: {p['title']}\n"
            f"Relevanz-Score: {p['score']}\n"
            f"Kategorien: {', '.join(p['categories'])}\n"
            f"URL: {p['url']}\n"
            f"Abstract (Auszug): {p['summary']}\n"
            f"{'-'*40}"
        )

    return "\n\n".join(output)


if __name__ == "__main__":
    mcp.run()