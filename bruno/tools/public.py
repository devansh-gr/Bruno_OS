"""Public-data tools: news, video search, market quotes.

All three are cloud_ok — they read public sources and carry no personal data.
They exist first because they are the lowest-risk way to prove the tool loop
works end to end. The local_only tools (email, portfolio, vault) land in M4 on
top of the same protocol.

Every reply is written to be *heard once*, with nothing on screen: no raw URLs,
no bare tickers, no ISO timestamps. See the spoken-output rule in the vault spec.
"""

from __future__ import annotations

import html
import re

import requests

from bruno.tools.base import Privacy, ToolResult, tool

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

FEEDS = {
    "world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "tech": "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "science": "https://www.sciencedaily.com/rss/top/science.xml",
    "business": "https://feeds.bbci.co.uk/news/business/rss.xml",
}


def _clean(s: str) -> str:
    """Strip markup and collapse whitespace. Everything here gets spoken."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def _spoken_number(x: float) -> str:
    """Money read aloud. 308.26 -> '308 dollars 26'; avoids 'three zero eight point two six'."""
    whole, cents = int(abs(x)), round(abs(x) % 1 * 100)
    sign = "negative " if x < 0 else ""
    return f"{sign}{whole:,} dollars {cents:02d}" if cents else f"{sign}{whole:,} dollars"


@tool(
    name="news",
    description=(
        "Read current news headlines. Use for 'what's in the news', 'any news about X', "
        "'catch me up'. Topics: world, tech, science, business."
    ),
    privacy=Privacy.CLOUD_OK,
    arg="topic",
)
def news(topic: str = "world", limit: int = 5, **_) -> ToolResult:
    import feedparser

    topic = (topic or "world").lower().strip()
    url = FEEDS.get(topic, FEEDS["world"])
    try:
        r = requests.get(url, headers=UA, timeout=12)
        r.raise_for_status()
    except requests.RequestException as e:
        return ToolResult(f"Couldn't reach the news feed: {e}", Privacy.CLOUD_OK, ok=False)

    entries = feedparser.parse(r.content).entries[: max(1, min(limit, 8))]
    if not entries:
        return ToolResult(f"No {topic} headlines came back.", Privacy.CLOUD_OK, ok=False)

    items = [_clean(e.get("title", "")) for e in entries]
    body = ". ".join(f"{i}. {t}" for i, t in enumerate(items, 1))
    return ToolResult(
        f"Top {len(items)} in {topic}. {body}.",
        Privacy.CLOUD_OK,
        data=[{"title": _clean(e.get("title", "")), "link": e.get("link", "")} for e in entries],
    )


@tool(
    name="video_search",
    description=(
        "Find videos on YouTube by title, topic, or creator. Use for 'find me a video "
        "about X', 'that talk by Y', 'pull up the video where'."
    ),
    privacy=Privacy.CLOUD_OK,
    arg="query",
)
def video_search(query: str, limit: int = 3, **_) -> ToolResult:
    from yt_dlp import YoutubeDL

    if not (query or "").strip():
        return ToolResult("I need something to search for.", Privacy.CLOUD_OK, ok=False)

    limit = max(1, min(limit, 5))
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    except Exception as e:  # noqa: BLE001 — yt-dlp raises a wide variety
        return ToolResult(f"Video search failed: {e}", Privacy.CLOUD_OK, ok=False)

    hits = [e for e in (info or {}).get("entries", []) if e]
    if not hits:
        return ToolResult(f"Nothing came back for {query}.", Privacy.CLOUD_OK, ok=False)

    def spoken(e):
        title, who = _clean(e.get("title", "untitled")), _clean(e.get("uploader") or "")
        secs = e.get("duration") or 0
        mins = f", {int(secs // 60)} minutes" if secs >= 60 else ""
        return f"{title}{f' by {who}' if who else ''}{mins}"

    body = ". ".join(f"{i}. {spoken(e)}" for i, e in enumerate(hits, 1))
    return ToolResult(
        f"Found {len(hits)}. {body}. Say the number and I'll open it.",
        Privacy.CLOUD_OK,
        data=[{"title": _clean(e.get("title", "")), "url": e.get("url", ""),
               "uploader": _clean(e.get("uploader") or "")} for e in hits],
    )


@tool(
    name="market_quote",
    description=(
        "Get the current price and daily move for a public stock, ETF, index, or crypto "
        "ticker. Use for 'how is X trading', 'what's the price of Y', 'is the market up'. "
        "This is public market data only — it does not know what Devansh owns."
    ),
    privacy=Privacy.CLOUD_OK,
    arg="symbol",
)
def market_quote(symbol: str, **_) -> ToolResult:
    sym = (symbol or "").strip().upper()
    if not sym:
        return ToolResult("Which ticker?", Privacy.CLOUD_OK, ok=False)

    try:
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
            headers=UA, params={"range": "1d", "interval": "1d"}, timeout=12,
        )
        r.raise_for_status()
        result = (r.json().get("chart") or {}).get("result") or []
        if not result:
            return ToolResult(f"No ticker matching {sym}.", Privacy.CLOUD_OK, ok=False)
        meta = result[0]["meta"]
        price = meta["regularMarketPrice"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        return ToolResult(f"Couldn't get a quote for {sym}: {e}", Privacy.CLOUD_OK, ok=False)

    prev = meta.get("previousClose") or meta.get("chartPreviousClose")
    name = _clean(meta.get("longName") or meta.get("shortName") or sym)
    said = _spoken_number(price) if (meta.get("currency") == "USD") else f"{price:,.2f}"

    move = ""
    if prev:
        pct = (price - prev) / prev * 100
        move = f", {'up' if pct >= 0 else 'down'} {abs(pct):.1f} percent on the day"
    return ToolResult(
        f"{name} is at {said}{move}.",
        Privacy.CLOUD_OK,
        data={"symbol": sym, "price": price, "previous_close": prev, "name": name},
    )
