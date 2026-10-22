import calendar
import html
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone

import arxiv
import feedparser
import requests

from core.config import Settings
from core.models import Item, SourceError, SourceStatusError
from core.ranking import is_excluded, similarity_scores

logger = logging.getLogger(__name__)

HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
FEED_TIMEOUT_SECONDS = 10
# Without a time window, only the newest posts per feed are ranked (some feeds contain years of archive)
FEED_RECENT_ENTRIES = 20

# Sort key for posts without a date
OLDEST = datetime.min.replace(tzinfo=timezone.utc)

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#/]+?)(?:v\d+)?(?:\.pdf)?(?:[?#].*)?$")


def arxiv_key(arxiv_id_or_url: str) -> str:
    """Normalize an arXiv ID or abs/pdf URL to 'arxiv:<id>' without version."""
    match = _ARXIV_ID_RE.search(arxiv_id_or_url)
    arxiv_id = match.group(1) if match else arxiv_id_or_url
    return "arxiv:" + re.sub(r"v\d+$", "", arxiv_id)


def fetch_arxiv(
    settings: Settings,
    categories: list[str] | None = None,
    max_fetch: int = 30,
    top_k: int = 5,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[Item]:
    """
    Fetch the newest arXiv papers of the given categories and rank them against the profile.
    With `since`, restrict to papers submitted in [since, until] (UTC).
    """
    categories = categories or settings.arxiv_categories
    query_str = " OR ".join([f"cat:{cat}" for cat in categories])
    client = arxiv.Client()
    if since:
        until = until or datetime.now(timezone.utc)
        query_str = (
            f"({query_str}) AND submittedDate:"
            f"[{since.strftime('%Y%m%d%H%M')} TO {until.strftime('%Y%m%d%H%M')}]"
        )
        # Bigger pages for weekly fetches, otherwise ~30 requests with a 3s delay each
        client = arxiv.Client(page_size=500, delay_seconds=3, num_retries=5)

    search = arxiv.Search(
        query=query_str,
        max_results=max_fetch,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    raw_papers = list(client.results(search))
    if len(raw_papers) >= max_fetch:
        # Results come newest first, so the cap silently drops the oldest papers of the window
        logger.warning("arXiv: max_fetch (%d) reached, the oldest papers of the window were not fetched; "
                       "raise [newsletter.arxiv] max_fetch.", max_fetch)

    papers = [p for p in raw_papers if not is_excluded(settings.exclude_keywords, p.title, p.summary)]
    if not papers:
        return []

    scores = similarity_scores(settings.profile, [f"{p.title}. {p.summary}" for p in papers])
    items = [
        Item(
            source="arxiv",
            title=p.title,
            url=p.pdf_url,
            key=arxiv_key(p.entry_id),
            summary=p.summary,
            date=p.published,
            score=score,
            meta={"categories": p.categories, "abs_url": p.entry_id},
        )
        for p, score in zip(papers, scores)
    ]
    items.sort(key=lambda i: i.score, reverse=True)
    return items[:top_k]


def _hf_request(params: dict | None = None) -> list[dict]:
    response = requests.get(HF_DAILY_PAPERS_URL, params=params, timeout=10)
    if response.status_code != 200:
        raise SourceStatusError(response.status_code)
    return response.json()


def fetch_huggingface(
    settings: Settings,
    top_k: int = 5,
    since: date | None = None,
    until: date | None = None,
) -> list[Item]:
    """
    Fetch Hugging Face Daily Papers sorted by upvotes.
    Without `since` only the current daily list is fetched; with `since` every day in
    [since, until] is fetched separately, because the API returns one day per request.
    """
    if since is None:
        raw = _hf_request()
    else:
        until = until or datetime.now(timezone.utc).date()
        raw, errors = [], []
        day = since
        while day <= until:
            try:
                raw.extend(_hf_request({"date": day.isoformat(), "limit": 100}))
            except Exception as e:
                errors.append(f"{day}: {e}")
            day += timedelta(days=1)
        if errors:
            logger.warning("Hugging Face: some days failed: %s", "; ".join(errors))
        if not raw and errors:
            raise SourceError("; ".join(errors))

    # The same paper can be listed on several days
    unique = {}
    for item in raw:
        paper = item.get("paper", {})
        if paper.get("id"):  # without an arXiv ID there is no link and no dedupe key
            unique.setdefault(paper["id"], item)

    # Upvotes live in item["paper"], not on the top-level item
    papers = [
        item for item in unique.values()
        if not is_excluded(
            settings.exclude_keywords,
            item.get("paper", {}).get("title"),
            item.get("paper", {}).get("summary"),
        )
    ]
    papers = sorted(
        papers,
        key=lambda item: item.get("paper", {}).get("upvotes", 0),
        reverse=True,
    )[:top_k]

    items = []
    for item in papers:
        paper = item.get("paper", {})
        published = paper.get("submittedOnDailyAt") or item.get("publishedAt")
        items.append(Item(
            source="huggingface",
            title=paper.get("title"),
            url=f"https://arxiv.org/abs/{paper.get('id')}",
            key=arxiv_key(str(paper.get("id"))),
            summary=paper.get("summary", ""),
            date=datetime.fromisoformat(published.replace("Z", "+00:00")) if published else None,
            meta={"upvotes": paper.get("upvotes", 0), "hf_url": f"https://huggingface.co/papers/{paper.get('id')}"},
        ))
    return items


def fetch_github(
    settings: Settings,
    top_k: int = 5,
    days: int = 7,
    min_stars: int = 20,
    max_fetch: int = 20,
) -> list[Item]:
    """
    Search GitHub for repositories created in the last `days` days, take the most-starred
    ones per topic as candidates and rank them semantically against the profile.
    Raises SourceError if nothing was found and at least one topic query failed.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    headers = {"User-Agent": "Smart-AI-Paper-Agent", "Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

    # GitHub ANDs multiple topic qualifiers, so query each topic separately and merge
    repos = {}
    errors = []
    for topic in settings.github_topics:
        params = {
            "q": f"topic:{topic} created:>={since} stars:>={min_stars}",
            "sort": "stars",
            "order": "desc",
            "per_page": max_fetch,
        }
        try:
            response = requests.get(GITHUB_SEARCH_URL, headers=headers, params=params, timeout=10)
            if response.status_code != 200:
                errors.append(f"{topic}: Status {response.status_code}")
                continue
            for repo in response.json().get("items", []):
                repos[repo["full_name"]] = repo
        except Exception as e:
            errors.append(f"{topic}: {str(e)}")

    if errors:
        logger.warning("GitHub: %s", "; ".join(errors))

    # Apply negative filter before ranking
    repos = {
        name: r for name, r in repos.items()
        if not is_excluded(settings.exclude_keywords, name, r.get("description"), " ".join(r.get("topics", [])))
    }

    if not repos:
        if errors:
            raise SourceError("; ".join(errors))
        return []

    # Rank the most-starred candidates by relevance to the profile
    candidates = list(repos.values())
    scores = similarity_scores(
        settings.profile,
        [
            f"{r['full_name']}. {r.get('description') or ''} {' '.join(r.get('topics', []))}"
            for r in candidates
        ],
    )
    items = [
        Item(
            source="github",
            title=r["full_name"],
            url=r.get("html_url"),
            key=r.get("html_url"),
            summary=r.get("description") or "",
            date=datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")) if r.get("created_at") else None,
            score=score,
            meta={
                "stars": r.get("stargazers_count"),
                "created": r.get("created_at", "")[:10],
                "description": r.get("description"),
                "since": since,
            },
        )
        for r, score in zip(candidates, scores)
    ]
    items.sort(key=lambda i: i.score, reverse=True)
    return items[:top_k]


def fetch_blogs(
    settings: Settings,
    top_k: int = 5,
    since: datetime | None = None,
) -> list[Item]:
    """
    Fetch posts of all configured blog feeds, rank them against the profile within each feed
    and pick round-robin across feeds.
    With `since` (timezone-aware), all posts published after it are candidates; without it,
    the newest FEED_RECENT_ENTRIES posts per feed.
    Raises SourceError if no feed could be fetched.
    """
    per_feed = []
    errors = []
    for url in settings.blog_feeds:
        try:
            # Download with requests instead of feedparser.parse(url): same certificate bundle
            # (certifi) as all other sources on every platform, a timeout and a real status code
            response = requests.get(
                url, headers={"User-Agent": "Smart-AI-Paper-Agent"}, timeout=FEED_TIMEOUT_SECONDS
            )
            if response.status_code != 200:
                logger.warning("Feed unreachable: %s (status %s)", url, response.status_code)
                errors.append(f"{url}: Status {response.status_code}")
                continue
            # Pass bytes so feedparser detects the encoding; content-location resolves relative links
            feed = feedparser.parse(
                response.content,
                response_headers={
                    "content-location": response.url,
                    "content-type": response.headers.get("content-type", ""),
                },
            )
            if not feed.entries:
                logger.warning("Feed has no entries: %s", url)
                errors.append(f"{url}: no entries")
                continue
            source = feed.feed.get("title", "AI Blog")
            entries = []
            for entry in feed.entries:
                if is_excluded(settings.exclude_keywords, entry.get("title"), entry.get("summary")):
                    continue
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                # feedparser normalizes feed dates to UTC struct_time
                published = datetime.fromtimestamp(calendar.timegm(published), timezone.utc) if published else None
                if since and (published is None or published < since):
                    continue
                entries.append(Item(
                    source="blogs",
                    title=entry.title,
                    url=entry.link,
                    key=entry.link,
                    summary=_strip_html(entry.get("summary", "")),
                    date=published,
                    meta={"feed": source},
                ))
            entries.sort(key=lambda e: e.date or OLDEST, reverse=True)
            if since is None:
                entries = entries[:FEED_RECENT_ENTRIES]
            per_feed.append(entries)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s: %s", url, type(e).__name__, e)
            errors.append(f"{url}: {e}")

    if not per_feed and errors:
        raise SourceError("; ".join(errors))

    # Score all posts in one batch, then sort each feed by relevance (newest first on ties)
    all_entries = [e for entries in per_feed for e in entries]
    scores = similarity_scores(settings.profile, [f"{e.title}. {e.summary}" for e in all_entries])
    for entry, score in zip(all_entries, scores):
        entry.score = score
    for entries in per_feed:
        entries.sort(key=lambda e: (e.score, e.date or OLDEST), reverse=True)

    # Round-robin across feeds so a single prolific blog cannot fill all slots
    selected = []
    for rank in range(max((len(f) for f in per_feed), default=0)):
        for entries in per_feed:
            if rank < len(entries):
                selected.append(entries[rank])
    return selected[:top_k]


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()  # also decode entities like &#8217;
