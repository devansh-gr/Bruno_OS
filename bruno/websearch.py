"""Web search, because the model's memory measured at zero.

SimpleQA: the local 3B got 0/120 and the 8B got 2/120, and the wrong answers were
confident inventions rather than near-misses. Wikipedia keyword search on the full
question found the answer 1 time in 10 -- half the queries returned no article at
all, because SimpleQA questions are long descriptive sentences and Wikipedia's
search wants entities.

DuckDuckGo's HTML endpoint finds it 7 times in 8. That is the whole difference
between a tutor that can answer a factual question and one that cannot.

The catch is rate limiting: it starts returning HTTP 202 with an empty body after
two rapid requests. Not an error, not a 429 -- a 202 with nothing in it, which is
easy to mistake for "no results". So requests are paced, the session is reused so
cookies persist, and an empty 202 is treated as throttling rather than absence.
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

ENDPOINTS = [
    "https://html.duckduckgo.com/html/",
    "https://lite.duckduckgo.com/lite/",
]

MIN_INTERVAL = 3.0        # measured: below this the endpoint starts 202-ing
_last_call = 0.0
_lock = threading.Lock()
_session: requests.Session | None = None

# Repeat questions are common -- "tell me more about that", a re-asked question,
# a benchmark re-run -- and every cache hit is 3 seconds of pacing saved.
_CACHE: dict[str, str] = {}
CACHE_MAX = 512


class Throttled(RuntimeError):
    """Every endpoint answered but returned nothing. Retry later, do not treat
    this as 'no results exist' -- that mistake turns a rate limit into a
    confident wrong answer downstream."""


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
_CHROME = re.compile(
    r"(All Regions\b.*?(?=\bSafe search|\bAny time|$)"
    r"|Safe search:.*?(?=\bAny time|$)"
    r"|Any time\s+Past day.*?(?=\b[A-Z]|$)"
    r"|Feedback\b.*?(?=$)"
    r"|Settings\s+Sign in.*?(?=$))", re.I | re.S)


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
    for url in ENDPOINTS:
        _pace()
        try:
            r = _sess().post(url, data={"q": query}, timeout=timeout)
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
