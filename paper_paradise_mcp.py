import logging
import sys

from fastmcp import FastMCP

from core import sources
from core.config import load_settings
from core.models import Item, SourceError, SourceStatusError
from core.ranking import get_embedder

# stdout is the JSON-RPC channel of the stdio transport, so log to stderr only
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
logging.getLogger("core").setLevel(logging.INFO)

# Load configs
settings = load_settings()
ARXIV_CATEGORIES = settings.arxiv_categories

# Initialize FastMCP Server
mcp = FastMCP("Smart arXiv & AI News Agent")

# Load small, fast Embedding-Model at startup so the first tool call is fast
get_embedder()


def format_arxiv(items: list[Item]) -> str:
    if not items:
        return "No matching arXiv papers found after filtering."

    output = [f"=== Top {len(items)} relevant arXiv papers for your profile ==="]
    for p in items:
        output.append(
            f"Title: {p.title}\n"
            f"Relevance score: {p.score}\n"
            f"Categories: {', '.join(p.meta['categories'])}\n"
            f"URL: {p.url}\n"
            f"Abstract (excerpt): {p.summary[:400]}...\n"
            f"{'-'*40}"
        )
    return "\n\n".join(output)


def format_huggingface(items: list[Item]) -> str:
    if not items:
        return "No current Hugging Face papers found."

    output = [f"=== Top {len(items)} Hugging Face Community Papers ==="]
    for p in items:
        output.append(
            f"Title: {p.title}\n"
            f"Upvotes: {p.meta['upvotes']}\n"
            f"arXiv ID / URL: {p.url}\n"
            f"Summary: {p.summary[:350]}...\n"
            f"{'-'*40}"
        )
    return "\n\n".join(output)


def format_github(items: list[Item]) -> str:
    if not items:
        return "No matching GitHub repositories found."

    output = [f"=== Top {len(items)} new ML repositories on GitHub (since {items[0].meta['since']}) ==="]
    for repo in items:
        output.append(
            f"Repo: {repo.title}\n"
            f"Relevance score: {repo.score}\n"
            f"Stars: ⭐ {repo.meta['stars']}\n"
            f"Created: {repo.meta['created']}\n"
            f"URL: {repo.url}\n"
            f"Description: {repo.meta['description']}\n"
            f"{'-'*40}"
        )
    return "\n\n".join(output)


def format_blogs(items: list[Item]) -> str:
    if not items:
        return "No recent blog posts found."

    output = ["=== Relevant AI & PyTorch engineering blog posts ==="]
    for e in items:
        output.append(
            f"Title: {e.title}\n"
            f"Relevance score: {e.score}\n"
            f"Source: {e.meta['feed']}\n"
            f"Date: {e.date.strftime('%Y-%m-%d') if e.date else 'unknown'}\n"
            f"URL: {e.url}\n"
            f"{'-'*40}"
        )
    return "\n\n".join(output)


@mcp.tool()
def fetch_filtered_arxiv_papers(
    categories: list[str] = ARXIV_CATEGORIES,
    max_fetch: int = 30,
    top_k: int = 5
) -> str:
    """
    Searches for new arXiv papers, filters them semantically against the learning profile,
    and returns only the most relevant papers to save tokens.
    """
    items = sources.fetch_arxiv(settings, categories=categories, max_fetch=max_fetch, top_k=top_k)
    return format_arxiv(items)


@mcp.tool()
def fetch_huggingface_daily_papers(top_k: int = 5) -> str:
    """
    Fetches today's community-voted top AI/ML papers from Hugging Face Daily Papers.
    """
    try:
        items = sources.fetch_huggingface(settings, top_k=top_k)
    except SourceStatusError as e:
        return f"Error fetching Hugging Face: status code {e.status}"
    except Exception as e:
        return f"Error fetching Hugging Face papers: {str(e)}"
    return format_huggingface(items)


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
    try:
        items = sources.fetch_github(settings, top_k=top_k, days=days, min_stars=min_stars, max_fetch=max_fetch)
    except SourceError as e:
        return f"GitHub API query failed: {e}"
    return format_github(items)


@mcp.tool()
def fetch_ai_engineering_blogs(top_k: int = 5) -> str:
    """
    Fetches the latest blog posts and tutorials from PyTorch and AI engineering blogs.
    """
    try:
        items = sources.fetch_blogs(settings, top_k=top_k)
    except SourceError:
        return "No recent blog posts found."
    return format_blogs(items)


if __name__ == "__main__":
    mcp.run()
