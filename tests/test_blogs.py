"""fetch_blogs with mocked HTTP responses (no network)."""
from types import SimpleNamespace

import pytest
import requests

from core import sources
from core.config import load_settings
from core.models import SourceError

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test Blog</title>
<item><title>Newer post</title><link>https://blog.test/newer</link>
  <pubDate>Thu, 08 Oct 2026 10:00:00 GMT</pubDate><description>&lt;p&gt;Some &lt;b&gt;text&lt;/b&gt;&lt;/p&gt;</description></item>
<item><title>Older post</title><link>/older</link>
  <pubDate>Mon, 05 Oct 2026 10:00:00 GMT</pubDate></item>
</channel></rss>"""


def settings_with(feeds):
    s = load_settings()
    object.__setattr__(s, "blog_feeds", feeds)
    return s


def fake_get(responses):
    calls = []

    def get(url, headers=None, timeout=None):
        calls.append({"url": url, "timeout": timeout})
        result = responses[url]
        if isinstance(result, Exception):
            raise result
        status, body = result
        return SimpleNamespace(status_code=status, content=body, url=url, headers={"content-type": "application/rss+xml"})

    get.calls = calls
    return get


def test_parses_feed_with_timeout_and_resolves_relative_links(monkeypatch):
    get = fake_get({"https://blog.test/feed": (200, RSS)})
    monkeypatch.setattr(sources.requests, "get", get)

    items = sources.fetch_blogs(settings_with(["https://blog.test/feed"]), top_k=5)

    assert [i.title for i in items] == ["Newer post", "Older post"]
    assert items[1].url == "https://blog.test/older"
    assert items[0].summary == "Some text"
    assert items[0].meta["feed"] == "Test Blog"
    assert get.calls[0]["timeout"] == sources.FEED_TIMEOUT_SECONDS


def test_one_failing_feed_does_not_break_the_others(monkeypatch):
    monkeypatch.setattr(sources.requests, "get", fake_get({
        "https://ok.test/feed": (200, RSS),
        "https://ssl.test/feed": requests.exceptions.SSLError("CERTIFICATE_VERIFY_FAILED"),
        "https://gone.test/feed": (404, b""),
    }))
    items = sources.fetch_blogs(settings_with(["https://ssl.test/feed", "https://gone.test/feed", "https://ok.test/feed"]))
    assert len(items) == 2


def test_all_feeds_failing_raises_with_reasons(monkeypatch):
    monkeypatch.setattr(sources.requests, "get", fake_get({
        "https://ssl.test/feed": requests.exceptions.SSLError("CERTIFICATE_VERIFY_FAILED"),
        "https://gone.test/feed": (404, b""),
    }))
    with pytest.raises(SourceError) as exc:
        sources.fetch_blogs(settings_with(["https://ssl.test/feed", "https://gone.test/feed"]))
    assert "CERTIFICATE_VERIFY_FAILED" in str(exc.value) and "Status 404" in str(exc.value)
