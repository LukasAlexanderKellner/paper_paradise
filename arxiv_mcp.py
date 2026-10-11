from fastmcp import FastMCP
import arxiv
import requests
import feedparser
from sentence_transformers import SentenceTransformer, util
import numpy as np
import config_module

# Load configs
cfg = config_module.Config()
STUDENT_PROFILE = cfg.STUDENT_PROFILE
EXCLUDE_KEYWORDS = cfg.EXCLUDE_KEYWORDS
PORN_CATEGORIES = cfg.PORN_CATEGORIES

# Initialize FastMCP Server
mcp = FastMCP("Smart arXiv & AI News Agent")
arxiv_client = arxiv.Client()

# Load small, fast Embedding-Model 
print("Lade Semantic-Embedding-Modell...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
profile_embedding = embedder.encode(STUDENT_PROFILE, convert_to_tensor=True)


@mcp.tool()
def fetch_filtered_arxiv_papers(
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
        return "Keine passenden arXiv Paper nach der Filterung gefunden."

    output = [f"=== Top {len(top_papers)} relevante arXiv Paper für dein Profil ==="]
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


@mcp.tool()
def fetch_huggingface_daily_papers(top_k: int = 5) -> str:
    """
    Fetches today's community-voted top AI/ML papers from Hugging Face Daily Papers.
    """
    url = "https://huggingface.co/api/daily_papers"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"Fehler beim Abrufen von Hugging Face: Status Code {response.status_code}"

        papers = response.json()[:top_k]
        if not papers:
            return "Keine aktuellen Hugging Face Papers gefunden."

        output = [f"=== Top {len(papers)} Hugging Face Community Papers ==="]
        for item in papers:
            paper_data = item.get("paper", {})
            output.append(
                f"Titel: {paper_data.get('title')}\n"
                f"Upvotes: {item.get('upvotes', 0)}\n"
                f"arXiv ID / URL: https://arxiv.org/abs/{paper_data.get('id')}\n"
                f"Summary: {paper_data.get('summary', '')[:350]}...\n"
                f"{'-'*40}"
            )
        return "\n\n".join(output)

    except Exception as e:
        return f"Fehler beim Abrufen der Hugging Face Papers: {str(e)}"


@mcp.tool()
def fetch_github_trending_ml_repos(top_k: int = 5) -> str:
    """
    Searches GitHub for newly created or trending Machine Learning and PyTorch repositories.
    """
    # Ruft kürzlich erstellte ML-Repositories mit vielen Stars ab
    url = "https://api.github.com/search/repositories?q=topic:machine-learning+topic:pytorch&sort=stars&order=desc"
    headers = {"User-Agent": "Smart-AI-Paper-Agent"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Fehler bei GitHub API Abfrage: Status {response.status_code}"

        items = response.json().get("items", [])[:top_k]
        if not items:
            return "Keine passenden GitHub Repositories gefunden."

        output = [f"=== Top {len(items)} Trending ML Repositories auf GitHub ==="]
        for repo in items:
            output.append(
                f"Repo: {repo.get('full_name')}\n"
                f"Stars: ⭐ {repo.get('stargazers_count')}\n"
                f"URL: {repo.get('html_url')}\n"
                f"Beschreibung: {repo.get('description')}\n"
                f"{'-'*40}"
            )
        return "\n\n".join(output)

    except Exception as e:
        return f"Fehler beim Abrufen der GitHub Repositories: {str(e)}"


@mcp.tool()
def fetch_ai_engineering_blogs(top_k: int = 5) -> str:
    """
    Fetches the latest blog posts and tutorials from PyTorch and AI engineering blogs.
    """
    feed_urls = [
        "https://pytorch.org/feed.xml",
        "https://huggingface.co/blog/feed.xml"
    ]
    
    entries = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                entries.append({
                    "title": entry.title,
                    "link": entry.link,
                    "source": feed.feed.get("title", "AI Blog")
                })
        except Exception:
            continue

    if not entries:
        return "Keine aktuellen Blog-Beiträge gefunden."

    entries = entries[:top_k]
    output = ["=== Neueste AI & PyTorch Engineering Blogbeiträge ==="]
    for e in entries:
        output.append(
            f"Titel: {e['title']}\n"
            f"Quelle: {e['source']}\n"
            f"URL: {e['link']}\n"
            f"{'-'*40}"
        )

    return "\n\n".join(output)


if __name__ == "__main__":
    mcp.run()