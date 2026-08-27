"""Web search, because model memory measured 0/120 on SimpleQA.

Wikipedia found the answer 1 time in 10; DuckDuckGo HTML finds it 7 in 8.

Rate limiting arrives as HTTP 202 with an empty body -- not a 429 -- which reads
exactly like "no results" unless you look for it. Hence the pacing and Throttled.
"""

from __future__ import annotations

import html
import re
import threading
import time

import requests

UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# Both DDG endpoints share one rate limit, so a DDG-only fallback bought nothing.
# Of mojeek, searx.be, startpage and Bing, only Bing had the answer.
ENDPOINTS = [
    ("https://html.duckduckgo.com/html/", "post", "q"),
    ("https://www.bing.com/search", "get", "q"),
    ("https://lite.duckduckgo.com/lite/", "post", "q"),
]

MIN_INTERVAL = 3.0        # measured: below this the endpoint starts 202-ing
_last_call = 0.0
_lock = threading.Lock()
_session: requests.Session | None = None

# Repeat questions are common -- "tell me more about that", a re-asked question,
# a benchmark re-run -- and every cache hit is 3 seconds of pacing saved.
# Demoted, not dropped: a blip should not retire the best provider.
_FAILS: dict[str, int] = {}


def _demote(url: str) -> None:
    _FAILS[url] = _FAILS.get(url, 0) + 1


def _ordered() -> list[tuple[str, str, str]]:
    return sorted(ENDPOINTS, key=lambda e: _FAILS.get(e[0], 0))


_CACHE: dict[str, str] = {}
CACHE_MAX = 512


class Throttled(RuntimeError):
    """Every endpoint answered but returned nothing. Retry later, do not treat
    this as 'no results exist' -- that mistake turns a rate limit into a
    confident wrong answer downstream.

    Note the limit is per-IP, not per-process: a benchmark run and an interactive
    Bruno session will throttle each other. Nothing here can fix that, but the
    error type at least makes the cause obvious instead of surfacing as a model
    that suddenly cannot answer anything.
    """


def _sess() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(UA)
    return _session


def _pace() -> None:
    global _last_call
    with _lock:
        wait = MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


# The results page echoes the query in its own header and carries a long region
# picker. Both score highly against the question's own words, so the first
# "densest passage" came back as "All Regions Argentina Australia Austria..."
# -- chrome that mentions the query is still chrome.
# Bounded. An earlier `.*?(?=$)` ate Bing from "Settings Sign in" to EOF, turning
# 123 KB into 88 characters -- and an over-aggressive filter looks like an empty page.
_CHROME = re.compile(
    r"(All Regions(?:\s+\w+(?:\s*\(\w+\))?){0,80}?\s*(?=Safe search|Any time|\d{4})"
    r"|Safe search:\s*\w+"
    r"|Any time\s+Past day\s+Past week\s+Past month\s+Past year"
    r"|Past Week\s+Past Month\s+Past Year)", re.I)


def _strip(page: str) -> str:
    page = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", page)
    text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page))).strip()
    text = _CHROME.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def search(query: str, timeout: int = 20) -> str:
    """Return the flattened text of the results page, or raise Throttled."""
    query = (query or "").strip()
    if not query:
        return ""
    with _lock:
        hit = _CACHE.get(query)
    if hit is not None:
        return hit
    for url, method, param in _ordered():
        _pace()
        try:
            if method == "post":
                r = _sess().post(url, data={param: query}, timeout=timeout)
            else:
                r = _sess().get(url, params={param: query}, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError):
            # A refused or timed-out connection is not throttling -- DuckDuckGo
            # does this once an IP has been hammered, and it costs the *full*
            # timeout on every question afterwards. Demote the provider so the
            # rest of the run stops paying for it.
            _demote(url)
            continue
        except requests.RequestException:
            continue
        # A 202 with an empty body is the throttle signal, not an empty result set.
        if r.status_code == 200 and len(r.text) > 2000:
            text = _strip(r.text)
            if len(text) > 300:
                with _lock:
                    if len(_CACHE) >= CACHE_MAX:
                        _CACHE.pop(next(iter(_CACHE)))
                    _CACHE[query] = text
                return text
    raise Throttled(f"no endpoint returned results for {query[:50]!r}")
