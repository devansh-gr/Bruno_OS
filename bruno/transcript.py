"""Getting what was actually *said* in a podcast, not what the publisher claims.

Three paths, cheapest first:

  1. **RSS <podcast:transcript>**: the Podcasting 2.0 standard. Correct and
     free when present, which is rare: 1 of 12 currently-charting shows.
  2. **YouTube auto-captions**: most of these shows post full episodes to
     YouTube, and yt-dlp pulls the caption track without downloading a byte of
     audio. Measured: a 75-minute All-In episode yields ~96k characters in a
     couple of seconds.
  3. **Local Whisper**: the honest fallback, and the expensive one. Minutes of
     compute per episode. Not wired up; the first two cover enough.

A 96k-character transcript cannot go near a 4k-context model, and summarising the
whole thing would be slow and mostly wasted. So extraction is targeted: find the
passages that mention what Devansh actually cares about, and keep only those.
An hour of podcast usually contains two minutes he wants.
"""

from __future__ import annotations

import re

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def from_rss_entry(entry) -> str | None:
    """Podcasting 2.0 transcript tag, when the publisher bothers."""
    links = entry.get("podcast_transcript") or entry.get("links") or []
    if isinstance(links, dict):
        links = [links]
    for link in links:
        url = link.get("url") or link.get("href")
        typ = (link.get("type") or "").lower()
        if not url or "transcript" not in f"{typ}{url}".lower():
            continue
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
        except requests.RequestException:
            continue
        text = r.text
        if "vtt" in typ or url.endswith(".vtt") or text.startswith("WEBVTT"):
            return _from_vtt(text)
        if "srt" in typ or url.endswith(".srt"):
            return _from_vtt(text)
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
    return None


def _from_vtt(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if (not line or line == "WEBVTT" or "-->" in line
                or line.isdigit() or line.startswith(("NOTE", "STYLE"))):
            continue
        lines.append(re.sub(r"<[^>]+>", "", line))
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def from_youtube(query: str, max_minutes: int = 240) -> tuple[str, str] | None:
    """Search YouTube for the episode and pull its caption track.

    Returns (transcript, url). Captions only: no audio is downloaded, which is
    what makes this fast enough to run across a whole briefing.
    """
    from yt_dlp import YoutubeDL

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with YoutubeDL(opts) as ydl:
            hit = (ydl.extract_info(f"ytsearch1:{query}", download=False)
                   .get("entries") or [None])[0]
            if not hit or not hit.get("id"):
                return None
            vid = hit["id"]
            full = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}",
                                    download=False)
    except Exception:  # noqa: BLE001: yt-dlp raises a wide variety
        return None

    if (full.get("duration") or 0) > max_minutes * 60:
        return None

    track = ((full.get("subtitles") or {}).get("en")
             or (full.get("automatic_captions") or {}).get("en") or [])
    json3 = [t for t in track if t.get("ext") == "json3"]
    if not json3:
        return None
    try:
        events = requests.get(json3[0]["url"], timeout=30).json().get("events", [])
    except (requests.RequestException, ValueError):
        return None

    text = " ".join(seg.get("utf8", "")
                    for ev in events for seg in (ev.get("segs") or []))
    text = re.sub(r"\s+", " ", text).strip()
    return (text, f"https://www.youtube.com/watch?v={vid}") if len(text) > 500 else None


MIN_HITS = 3   # below this, the "relevant" passage is a coincidence


def relevant_passages(transcript: str, interests: set[str],
                      window: int = 700, limit: int = 4) -> list[str]:
    """Keep only the parts that touch what he cares about.

    A 96k transcript against a 4k context is not a summarisation problem, it is a
    search problem. Find where the interesting words cluster, take a window around
    each, and hand the model a few hundred words instead of an hour of talk.
    """
    if not transcript or not interests:
        return []
    low = transcript.lower()
    hits: list[int] = []
    for term in interests:
        if len(term) < 4:
            continue
        for m in re.finditer(rf"\b{re.escape(term)}\b", low):
            hits.append(m.start())
    # A couple of stray matches in an hour of talk is noise, not relevance. The
    # first version returned a passage about someone's childhood because the word
    # "research" appeared twice nearby.
    if len(hits) < MIN_HITS:
        return []
    hits.sort()

    # Merge hits that fall inside the same window, so one dense passage is one
    # passage rather than five overlapping copies of itself.
    passages, start = [], hits[0]
    last = hits[0]
    for pos in hits[1:]:
        if pos - last > window:
            passages.append((start, last))
            start = pos
        last = pos
    passages.append((start, last))
    passages.sort(key=lambda p: -(p[1] - p[0]))   # densest clusters first

    out = []
    for lo, hi in passages[:limit]:
        a = max(0, lo - window // 2)
        b = min(len(transcript), hi + window)
        snippet = transcript[a:b].strip()
        if a:                      # don't start mid-word
            snippet = snippet.split(" ", 1)[-1]
        out.append(snippet)
    return out


def fetch(title: str, show: str, entry=None, interests: set[str] | None = None
          ) -> dict | None:
    """Best available transcript for an episode, plus the passages worth hearing."""
    text, url = None, None
    if entry is not None:
        text = from_rss_entry(entry)
        if text:
            url = "rss"
    if not text:
        got = from_youtube(f"{show} {title}")
        if got:
            text, url = got
    if not text:
        return None
    return {
        "text": text,
        "source": url,
        "chars": len(text),
        "passages": relevant_passages(text, interests or set()),
    }
