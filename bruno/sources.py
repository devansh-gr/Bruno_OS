"""Where the briefing gets its material. Discovered, not hardcoded.

Rule for everything here: **nothing is predicted.** Every item carries the text
that was actually fetched and the URL it came from, so any claim Bruno makes can
be opened and read. The model's only jobs are selecting and compressing.

Podcast shows are discovered from the Apple charts per genre rather than listed
in code, so the set tracks what is actually being listened to instead of what was
popular the day the file was written. No API key.

X is included via bruno/x_source.py, which uses Nitter RSS with the syndication
embed backend as a fallback. I initially assumed X was unscrapeable and said so
here; probed properly, both paths work and return posts fresh to the minute.
Which accounts to follow is a decision for Devansh, not a default — X_ACCOUNTS
is empty until he says.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# Apple's genre IDs. Chart endpoints take these directly.
GENRES = {
    "business": 1321,
    "tech": 1318,
    "news": 1489,
    "politics": 1325,   # Government
}

NEWS_FEEDS = {
    "world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "tech": "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "science": "https://www.sciencedaily.com/rss/top/science.xml",
    "hn": "https://hnrss.org/frontpage?points=150",
}


@dataclass
class Item:
    """One thing that happened. `text` is fetched, never generated."""
    title: str
    text: str
    url: str
    source: str          # "hn", "podcast:Bloomberg Tech", "news:world"
    kind: str            # article | podcast | market
    published: datetime | None = None
    score: float = 0.0
    why: list[str] = field(default_factory=list)   # what made it rank

    def age_hours(self) -> float:
        if not self.published:
            return 999.0
        return (datetime.now(timezone.utc) - self.published).total_seconds() / 3600


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def _parsed_time(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime.fromtimestamp(time.mktime(t), timezone.utc)
    return None


def _feed(url: str, limit: int) -> list:
    import feedparser
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
    except requests.RequestException:
        return []
    return feedparser.parse(r.content).entries[:limit]


def news_items(topics=("world", "tech", "business", "hn"), per=8) -> list[Item]:
    out = []
    for topic in topics:
        url = NEWS_FEEDS.get(topic)
        if not url:
            continue
        for e in _feed(url, per):
            title = _clean(e.get("title", ""))
            if not title:
                continue
            out.append(Item(
                title=title,
                text=_clean(e.get("summary", ""))[:600] or title,
                url=e.get("link", ""),
                source=f"news:{topic}",
                kind="article",
                published=_parsed_time(e),
            ))
    return out


def discover_shows(genre: str, limit: int = 8) -> list[dict]:
    """Top shows in a genre, from the Apple charts. No key, no hardcoded list."""
    gid = GENRES.get(genre)
    if not gid:
        return []
    try:
        r = requests.get(
            f"https://itunes.apple.com/us/rss/toppodcasts/limit={limit}/genre={gid}/json",
            headers=UA, timeout=15)
        r.raise_for_status()
        entries = r.json().get("feed", {}).get("entry", [])
    except (requests.RequestException, ValueError, AttributeError):
        return []
    if isinstance(entries, dict):
        entries = [entries]

    shows = []
    for e in entries:
        name = (e.get("im:name") or {}).get("label")
        pid = (e.get("id") or {}).get("attributes", {}).get("im:id")
        if name and pid:
            shows.append({"name": name, "id": pid, "genre": genre})
    return shows


def show_feed_url(podcast_id: str) -> str | None:
    """The charts give an Apple ID; the lookup endpoint gives the real RSS feed."""
    try:
        r = requests.get("https://itunes.apple.com/lookup", headers=UA, timeout=15,
                         params={"id": podcast_id, "entity": "podcast"})
        r.raise_for_status()
        results = r.json().get("results", [])
    except (requests.RequestException, ValueError):
        return None
    return results[0].get("feedUrl") if results else None


def podcast_items(genres=("business", "tech", "news"), shows_per_genre=4,
                  episodes_per_show=2, max_age_days=7) -> list[Item]:
    """Recent episodes from currently-charting shows.

    Episode *descriptions* only — transcription would be the honest way to know
    what was actually said, and it costs minutes of compute per episode. The
    description is what the publisher claims the episode is about, which is
    enough to decide whether Devansh wants to hear more.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    out = []
    for genre in genres:
        for show in discover_shows(genre, shows_per_genre):
            feed = show_feed_url(show["id"])
            if not feed:
                continue
            for e in _feed(feed, episodes_per_show):
                when = _parsed_time(e)
                if when and when < cutoff:
                    continue
                title = _clean(e.get("title", ""))
                if not title:
                    continue
                out.append(Item(
                    title=title,
                    text=_clean(e.get("summary", ""))[:700] or title,
                    url=e.get("link") or feed,
                    source=f"podcast:{show['name']}",
                    kind="podcast",
                    published=when,
                ))
    return out


# Empty on purpose. Whose posts matter is Devansh's call and guessing at it would
# produce a feed that looks personalised and isn't.
X_ACCOUNTS: list[str] = []


def x_items(accounts: list[str] | None = None, per: int = 5,
            max_age_days: int = 3) -> list[Item]:
    """Posts from accounts Devansh follows deliberately."""
    from bruno import x_source

    accounts = accounts if accounts is not None else X_ACCOUNTS
    if not accounts:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    out = []
    for handle in accounts:
        for p in x_source.posts(handle, limit=per):
            if p["published"] and p["published"] < cutoff:
                continue
            text = p["text"]
            if not text or text.startswith("RT by"):
                continue
            out.append(Item(
                title=text[:140],
                text=text,
                url=p["url"],
                source=f"x:@{p['handle']}",
                kind="post",
                published=p["published"],
            ))
    return out


def market_items(symbols=("^GSPC", "^IXIC", "NVDA", "BTC-USD")) -> list[Item]:
    """Only surfaces a mover. A ticker that did nothing is not news."""
    from bruno.tools.public import market_quote

    out = []
    for sym in symbols:
        r = market_quote(symbol=sym)
        if not r.ok or not r.data or not r.data.get("previous_close"):
            continue
        d = r.data
        pct = (d["price"] - d["previous_close"]) / d["previous_close"] * 100
        if abs(pct) < 1.5:
            continue
        out.append(Item(
            title=f"{d['name']} {'up' if pct > 0 else 'down'} {abs(pct):.1f} percent",
            text=r.text,
            url=f"https://finance.yahoo.com/quote/{sym}",
            source="market",
            kind="market",
            published=datetime.now(timezone.utc),
        ))
    return out
