import os
import sys
from datetime import datetime, timedelta, timezone
from time import mktime

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
GITHUB_TOPICS = cfg.GITHUB_TOPICS
BLOG_FEEDS = cfg.BLOG_FEEDS

# Initialize FastMCP Server
mcp = FastMCP("Smart arXiv & AI News Agent")
arxiv_client = arxiv.Client()

# Load small, fast Embedding-Model 
print("Lade Semantic-Embedding-Modell...", file=sys.stderr)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
profile_embedding = embedder.encode(STUDENT_PROFILE, convert_to_tensor=True)


def is_excluded(*texts: str) -> bool:
    """Hard negative filter: True if any EXCLUDE_KEYWORD appears in the given texts."""
    combined = " ".join(t or "" for t in texts).lower()
    return any(neg.lower() in combined for neg in EXCLUDE_KEYWORDS)


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

    # 1. Hard Rule: apply negative filter
    papers = [
        p for p in raw_papers
        if not is_excluded(p.title, p.summary)
    ]
    if not papers:
        return "Keine passenden arXiv Paper nach der Filterung gefunden."

    # 2. Calculate Semantic Search Score (batched, title + abstract)
    paper_embeddings = embedder.encode(
        [f"{p.title}. {p.summary}" for p in papers], convert_to_tensor=True
    )
    similarities = util.cos_sim(profile_embedding, paper_embeddings)[0].tolist()

    for paper, similarity in zip(papers, similarities):
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

        # Upvotes live in item["paper"], not on the top-level item
        papers = [
            item for item in response.json()
            if not is_excluded(
                item.get("paper", {}).get("title"),
                item.get("paper", {}).get("summary"),
            )
        ]
        papers = sorted(
            papers,
            key=lambda item: item.get("paper", {}).get("upvotes", 0),
            reverse=True,
        )[:top_k]
        if not papers:
            return "Keine aktuellen Hugging Face Papers gefunden."

        output = [f"=== Top {len(papers)} Hugging Face Community Papers ==="]
        for item in papers:
            paper_data = item.get("paper", {})
            output.append(
                f"Titel: {paper_data.get('title')}\n"
                f"Upvotes: {paper_data.get('upvotes', 0)}\n"
                f"arXiv ID / URL: https://arxiv.org/abs/{paper_data.get('id')}\n"
                f"Summary: {paper_data.get('summary', '')[:350]}...\n"
                f"{'-'*40}"
            )
        return "\n\n".join(output)

    except Exception as e:
        return f"Fehler beim Abrufen der Hugging Face Papers: {str(e)}"


@mcp.tool()
def fetch_github_trending_ml_repos(
    top_k: int = 5,
    days: int = 7,
    min_stars: int = 20,
    max_fetch: int = 20
) -> str:
    """
    Searches GitHub for Machine Learning and PyTorch repositories created in the last
    `days` days, takes the most-starred ones as candidates and ranks them semantically
    against the learning profile.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    headers = {"User-Agent": "Smart-AI-Paper-Agent", "Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

    # GitHub ANDs multiple topic qualifiers, so query each topic separately and merge
    repos = {}
    errors = []
    for topic in GITHUB_TOPICS:
        url = "https://api.github.com/search/repositories"
        params = {
            "q": f"topic:{topic} created:>={since} stars:>={min_stars}",
            "sort": "stars",
            "order": "desc",
            "per_page": max_fetch,
        }
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code != 200:
                errors.append(f"{topic}: Status {response.status_code}")
                continue
            for repo in response.json().get("items", []):
                repos[repo["full_name"]] = repo
        except Exception as e:
            errors.append(f"{topic}: {str(e)}")

    # Apply negative filter before ranking
    repos = {
        name: r for name, r in repos.items()
        if not is_excluded(name, r.get("description"), " ".join(r.get("topics", [])))
    }

    if not repos:
        if errors:
            return f"Fehler bei GitHub API Abfrage: {'; '.join(errors)}"
        return "Keine passenden GitHub Repositories gefunden."

    # Rank the most-starred candidates by relevance to the profile
    candidates = list(repos.values())
    repo_embeddings = embedder.encode(
        [
            f"{r['full_name']}. {r.get('description') or ''} {' '.join(r.get('topics', []))}"
            for r in candidates
        ],
        convert_to_tensor=True,
    )
    similarities = util.cos_sim(profile_embedding, repo_embeddings)[0].tolist()
    for repo, similarity in zip(candidates, similarities):
        repo["score"] = round(similarity, 3)

    items = sorted(candidates, key=lambda r: r["score"], reverse=True)[:top_k]

    output = [f"=== Top {len(items)} neue ML Repositories auf GitHub (seit {since}) ==="]
    for repo in items:
        output.append(
            f"Repo: {repo.get('full_name')}\n"
            f"Relevanz-Score: {repo['score']}\n"
            f"Stars: ⭐ {repo.get('stargazers_count')}\n"
            f"Erstellt: {repo.get('created_at', '')[:10]}\n"
            f"URL: {repo.get('html_url')}\n"
            f"Beschreibung: {repo.get('description')}\n"
            f"{'-'*40}"
        )
    return "\n\n".join(output)


@mcp.tool()
def fetch_ai_engineering_blogs(top_k: int = 5) -> str:
    """
    Fetches the latest blog posts and tutorials from PyTorch and AI engineering blogs.
    """
    per_feed = []
    for url in BLOG_FEEDS:
        try:
            feed = feedparser.parse(url)
            # feedparser does not raise on HTTP errors, check status explicitly
            if feed.get("status", 200) >= 400 or not feed.entries:
                print(f"Feed nicht erreichbar: {url} (Status {feed.get('status')})", file=sys.stderr)
                continue
            source = feed.feed.get("title", "AI Blog")
            entries = []
            for entry in feed.entries:
                if is_excluded(entry.get("title"), entry.get("summary")):
                    continue
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                entries.append({
                    "title": entry.title,
                    "link": entry.link,
                    "source": source,
                    "published": datetime.fromtimestamp(mktime(published)) if published else None,
                })
            entries.sort(key=lambda e: e["published"] or datetime.min, reverse=True)
            per_feed.append(entries)
        except Exception as e:
            print(f"Fehler beim Parsen von {url}: {e}", file=sys.stderr)

    # Round-robin across feeds so a single prolific blog cannot fill all slots
    selected = []
    for rank in range(max((len(f) for f in per_feed), default=0)):
        for entries in per_feed:
            if rank < len(entries):
                selected.append(entries[rank])
    selected = selected[:top_k]

    if not selected:
        return "Keine aktuellen Blog-Beiträge gefunden."

    output = ["=== Neueste AI & PyTorch Engineering Blogbeiträge ==="]
    for e in selected:
        output.append(
            f"Titel: {e['title']}\n"
            f"Quelle: {e['source']}\n"
            f"Datum: {e['published'].strftime('%Y-%m-%d') if e['published'] else 'unbekannt'}\n"
            f"URL: {e['link']}\n"
            f"{'-'*40}"
        )

    return "\n\n".join(output)


if __name__ == "__main__":
    mcp.run()