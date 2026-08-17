"""X/Twitter, without the $200/month API.

I initially wrote X off as unscrapeable and said so in code comments. That was
wrong, and it was also not my call to make. Probed properly, two independent
paths work today:

  1. **Nitter RSS** (nitter.net) — clean RSS, correctly ordered, fresh to the
     minute. Most public instances are dead because X revoked the guest tokens
     they relied on, but the ones running authenticated accounts still serve.
  2. **Syndication embed backend** (syndication.twitter.com) — the service that
     powers embedded timelines on other websites. Returns ~100 posts as JSON
     inside a __NEXT_DATA__ block. Not rate-limited the way the API is, because
     it exists to be called by arbitrary third-party pages.

Two paths matter more than either one being good. Instance availability is the
failure mode here, so the design assumption is that any single source is
temporary and the fetcher rotates until something answers.
"""

from __future__ import annotations

import html
import json
import re
import threading
import time
from datetime import datetime, timezone

from bruno import config

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

# Ordered by how well they worked when last probed. Cheap to extend; expect
# entries to die without notice, which is why the fetcher tries all of them.
NITTER_HOSTS = [
    "nitter.net",
    "nitter.tiekoetter.com",
    "nitter.poast.org",
    "lightbrd.com",
]

_HEALTH: dict[str, float] = {}      # host -> unix time it last failed
COOLDOWN = 900                      # don't retry a dead host for 15 minutes


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def _via_nitter(handle: str, limit: int) -> list[dict]:
    import feedparser

    for host in NITTER_HOSTS:
        if time.time() - _HEALTH.get(host, 0) < COOLDOWN:
            continue
        try:
            r = requests.get(f"https://{host}/{handle}/rss", headers=UA, timeout=12)
            if r.status_code != 200 or b"<rss" not in r.content[:400]:
                _HEALTH[host] = time.time()
                continue
            entries = feedparser.parse(r.content).entries[:limit]
            if not entries:
                _HEALTH[host] = time.time()
                continue
            out = []
            for e in entries:
                when = None
                if e.get("published_parsed"):
                    when = datetime.fromtimestamp(time.mktime(e.published_parsed),
                                                  timezone.utc)
                out.append({
                    "handle": handle,
                    "text": _clean(e.get("title", "")),
                    "url": e.get("link", ""),
                    "published": when,
                    "via": host,
                })
            return out
        except requests.RequestException:
            _HEALTH[host] = time.time()
    return []


def _via_syndication(handle: str, limit: int) -> list[dict]:
    """The backend behind embedded timelines. Independent of Nitter entirely."""
    try:
        r = requests.get(
            f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{handle}",
            headers=UA, timeout=15)
        r.raise_for_status()
    except requests.RequestException:
        return []

    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                  r.text, re.S)
    if not m:
        return []
    try:
        entries = (json.loads(m.group(1)).get("props", {}).get("pageProps", {})
                   .get("timeline", {}).get("entries", []))
    except (json.JSONDecodeError, AttributeError):
        return []

    out = []
    for e in entries:
        t = (e.get("content") or {}).get("tweet") or {}
        text = _clean(t.get("full_text") or "")
        if not text:
            continue
        when = None
        raw = t.get("created_at")
        if raw:
            try:
                when = datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y")
            except ValueError:
                pass
        out.append({
            "handle": handle,
            "text": text,
            "url": f"https://x.com/{handle}/status/{t.get('id_str', '')}",
            "published": when,
            "via": "syndication",
        })
    # Syndication does not guarantee reverse-chronological order the way the RSS
    # does -- pinned and popular posts surface first -- so sort before trimming.
    out.sort(key=lambda p: p["published"] or datetime.min.replace(tzinfo=timezone.utc),
             reverse=True)
    return out[:limit]


def posts(handle: str, limit: int = 10) -> list[dict]:
    """Recent posts from one account. Tries every path before giving up."""
    handle = handle.lstrip("@").strip()
    if not handle:
        return []
    return _via_nitter(handle, limit) or _via_syndication(handle, limit)


# --- caching -------------------------------------------------------------
#
# Both paths rate-limit hard. Nitter starts refusing after roughly forty requests
# even at 1.5s intervals, and the syndication endpoint returns 429 after similar
# volume. With 66 accounts, fetching them all per briefing is not possible at any
# polite request rate.
#
# So: cache per handle on disk, and each briefing refreshes only the few stalest
# accounts. Across a handful of briefings every account gets covered, the request
# rate stays low enough never to trip a limit, and a briefing always has content
# because stale posts are still posts.

CACHE = config.ROOT / "x_cache.json"
TTL_SECONDS = 6 * 3600
REFRESH_PER_RUN = 6


def _load_cache() -> dict:
    if not CACHE.exists():
        return {}
    try:
        return json.loads(CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(data: dict) -> None:
    try:
        CACHE.write_text(json.dumps(data, indent=1))
    except OSError:
        pass


def _serialise(posts_: list[dict]) -> list[dict]:
    return [{**p, "published": p["published"].isoformat() if p["published"] else None}
            for p in posts_]


def _deserialise(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        when = r.get("published")
        out.append({**r, "published": datetime.fromisoformat(when) if when else None})
    return out


_refreshing = threading.Lock()


def _all_hosts_cooling() -> bool:
    now = time.time()
    return all(now - _HEALTH.get(h, 0) < COOLDOWN for h in NITTER_HOSTS)


def _refresh_worker(handles: list[str], limit: int) -> None:
    """Fetch the stalest handles, slowly, off the critical path."""
    if not _refreshing.acquire(blocking=False):
        return                       # a refresh is already running
    try:
        cache = _load_cache()
        now = time.time()
        stale = sorted(handles, key=lambda h: cache.get(h, {}).get("at", 0))
        due = [h for h in stale if now - cache.get(h, {}).get("at", 0) > TTL_SECONDS]
        for handle in due[:REFRESH_PER_RUN]:
            if _all_hosts_cooling():
                break                # every path is rate-limited; try again later
            got = posts(handle, limit=limit)
            if got:
                cache[handle] = {"at": time.time(), "posts": _serialise(got)}
            else:
                # Record the attempt so a permanently dead handle doesn't
                # monopolise the refresh budget on every single run.
                cache.setdefault(handle, {"posts": []})["at"] = time.time()
            _save_cache(cache)
            time.sleep(2.0)          # deliberately slow; the caller isn't waiting
    finally:
        _refreshing.release()


def posts_cached(handles: list[str], limit: int = 5,
                 refresh: int = REFRESH_PER_RUN) -> dict[str, list[dict]]:
    """Recent posts for many accounts, served from cache, refreshed in the background.

    The briefing must not wait on X. Fetching even six accounts politely costs
    ~10 seconds, and the whole point of the cache is that stale posts are still
    posts. So this returns immediately from disk and kicks the refresh onto a
    background thread, which fills in for the *next* briefing.
    """
    cache = _load_cache()
    if refresh:
        threading.Thread(target=_refresh_worker, args=(handles, limit),
                         daemon=True).start()
    return {h: _deserialise(cache.get(h, {}).get("posts", []))
            for h in handles if cache.get(h, {}).get("posts")}


def health() -> dict:
    """Which paths answered last time. Worth surfacing rather than failing quietly —
    a briefing missing X entirely should say so, not just look thin."""
    cache = _load_cache()
    fresh = sum(1 for v in cache.values() if time.time() - v.get("at", 0) < TTL_SECONDS)
    return {
        "nitter_hosts": NITTER_HOSTS,
        "cooling_down": [h for h, t in _HEALTH.items() if time.time() - t < COOLDOWN],
        "cached_handles": len(cache),
        "fresh_handles": fresh,
    }
