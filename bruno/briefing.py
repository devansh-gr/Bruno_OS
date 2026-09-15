"""Collect, dedupe, rank, and let Devansh refer to items by what they're about.

The design constraint that shapes everything: he wants to say *"pull up that
Nvidia one"*, not *"item three"*. Numbers are a UI that only works when you can
see a list, and he can't: he's listening. So items are addressed by content, and
the last briefing stays resident so a follow-up has something to match against.

Nothing here is generated. Ranking is arithmetic over fetched text; every item
keeps the URL it came from. The model's only job is compressing what was fetched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bruno.sources import Item, market_items, news_items, podcast_items, x_items

# What "relevant to Devansh" means, in the absence of him having said. These are
# read from his vault projects at runtime where possible; this is the fallback.
DEFAULT_INTERESTS = (
    "ai", "llm", "model", "neural", "robot", "robotics", "biomedical", "neuro",
    "bci", "eeg", "emg", "prosthetic", "nvidia", "anthropic", "openai", "claude",
    "chip", "semiconductor", "gpu", "hardware", "startup", "funding", "yc",
    "waterloo", "research", "paper", "benchmark", "sensor", "actuator",
)

STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "that", "this", "it", "is", "was", "are", "were", "be", "been", "as", "at",
    "by", "from", "has", "have", "had", "will", "would", "can", "could", "about",
    "one", "new", "says", "said", "after", "over", "into", "more", "than",
}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (text or "").lower())
            if w not in STOP}


def _interests() -> set[str]:
    """Prefer his actual project names over a guessed keyword list."""
    from pathlib import Path
    terms = set(DEFAULT_INTERESTS)
    vault = Path.home() / "Documents" / "Devansh AI Brain" / "03 Projects"
    if vault.is_dir():
        for d in vault.iterdir():
            if d.is_dir():
                terms |= {w for w in d.name.lower().replace("_", " ").split() if len(w) > 3}
    return terms


@dataclass
class Briefing:
    items: list[Item] = field(default_factory=list)

    def find(self, description: str) -> Item | None:
        """Match a spoken description against the briefing.

        'the Nvidia one', 'that thing about the chip export ban', 'the Bloomberg
        podcast': all of these are keyword overlap against titles and text, which
        is enough at ten items and avoids an embedding model in the latency path.
        """
        want = words(description)
        if not want or not self.items:
            return None
        best, best_score = None, 0.0
        for it in self.items:
            title_hits = len(want & words(it.title))
            body_hits = len(want & words(it.text))
            source_hits = len(want & words(it.source))
            # Title and source are weighted hardest: people refer to things by
            # their headline or their publisher, not by a phrase buried in body text.
            score = title_hits * 3 + source_hits * 2 + body_hits
            if score > best_score:
                best, best_score = it, score
        return best if best_score >= 2 else None


def dedupe(items: list[Item]) -> list[Item]:
    """The same story from five outlets is one story."""
    kept: list[Item] = []
    for it in items:
        w = words(it.title)
        if not w:
            continue
        dup = False
        for k in kept:
            kw = words(k.title)
            overlap = len(w & kw) / max(1, min(len(w), len(kw)))
            if overlap > 0.6:
                dup = True
                break
        if not dup:
            kept.append(it)
    return kept


def rank(items: list[Item], interests: set[str] | None = None) -> list[Item]:
    """Relevance to him, not popularity. Recency breaks ties rather than driving."""
    interests = interests or _interests()
    for it in items:
        hits = (words(it.title) | words(it.text)) & interests
        it.why = sorted(hits)[:4]
        score = len(hits) * 2.0
        age = it.age_hours()
        score += 3.0 if age < 12 else 1.5 if age < 36 else 0.0
        # A market move or a charting podcast is worth surfacing on its own terms.
        score += {"market": 2.0, "podcast": 1.0, "post": 1.5}.get(it.kind, 0.0)
        it.score = score
    return sorted(items, key=lambda i: (-i.score, i.age_hours()))


def collect(include_podcasts: bool = True, x_accounts: list[str] | None = None) -> list[Item]:
    items = news_items() + market_items() + x_items(x_accounts)
    if include_podcasts:
        items += podcast_items()
    return items


def build(limit: int = 8, include_podcasts: bool = True,
          x_accounts: list[str] | None = None) -> Briefing:
    return Briefing(rank(dedupe(collect(include_podcasts, x_accounts)))[:limit])


def speak(b: Briefing) -> str:
    """Read as a list of things, each identifiable by what it is about: no
    numbers, because a number is only useful to someone looking at a screen."""
    if not b.items:
        return "Nothing came back from any source."
    lines = []
    for it in b.items:
        where = it.source.split(":")[-1]
        lines.append(f"{it.title}. {where}")
    return (f"{len(lines)} things worth knowing. " + ". ".join(lines) +
            ". Say which one and I'll go deeper.")
