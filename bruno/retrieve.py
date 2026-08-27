"""Search, focus, ground: evidence the model reads instead of recalls.

The grounding prompt checks the question's premise before answering. Leading with
"answer only from the sources" turned the model into a summariser that stopped
questioning false premises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bruno import websearch

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "that",
    "this", "it", "is", "was", "were", "are", "be", "been", "as", "at", "by",
    "from", "what", "which", "who", "whom", "when", "where", "how", "why", "did",
    "does", "do", "has", "have", "had", "name", "give", "tell", "many", "much",
}


def key_terms(question: str) -> set[str]:
    """Content words, possessives stripped: `india's` never matches "India", which
    zeroed the most important term in the question and broke passage ranking."""
    out = set()
    for w in re.findall(r"[a-z0-9][a-z0-9'\-]{2,}", question.lower()):
        w = re.sub(r"'s$", "", w).strip("-'")
        if len(w) > 2 and w not in STOP:
            out.add(w)
    return out


@dataclass
class Evidence:
    question: str
    passages: list[str]
    chars_searched: int

    def as_prompt(self) -> str:
        body = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(self.passages, 1))
        return f"Search results:\n{body}\n\nQuestion: {self.question}"


def focus(text: str, question: str, window: int = 420, keep: int = 6) -> list[str]:
    """Densest windows by term overlap: ~5000 chars of results page down to ~2000
    that are about the question, which is what makes it fit the context."""
    terms = key_terms(question)
    if not terms or not text:
        return []
    low = text.lower()

    scored: list[tuple[int, int]] = []
    step = window // 2
    for start in range(0, max(1, len(text) - window), step):
        chunk = low[start:start + window]
        score = sum(1 for t in terms if t in chunk)
        if score:
            scored.append((score, start))
    if not scored:
        return []

    scored.sort(key=lambda s: -s[0])
    picked: list[tuple[int, int]] = []
    for score, start in scored:
        # Skip windows that mostly overlap one already taken, so `keep` distinct
        # passages are distinct rather than five slices of the same sentence.
        if any(abs(start - p) < window for _, p in picked):
            continue
        picked.append((score, start))
        if len(picked) >= keep:
            break

    return [text[s:s + window].strip() for _, s in sorted(picked, key=lambda p: p[1])]


def gather(question: str) -> Evidence | None:
    """Search and focus. Returns None only when the web genuinely had nothing;
    a throttle propagates so the caller never mistakes it for absence."""
    text = websearch.search(question)
    if not text:
        return None
    passages = focus(text, question)
    if not passages:
        return None
    return Evidence(question, passages, len(text))


GROUNDED_SYSTEM = (
    "You answer using ONLY the search results provided.\n\n"
    "First check the question's premise against the results. If they contradict "
    "it, say so and correct it before anything else.\n\n"
    "Then answer. Give the short factual answer only -- a name, a number, a date "
    "-- with no preamble and no explanation unless asked.\n\n"
    "If the results do not contain the answer, reply exactly: I don't know. "
    "Never fill the gap from memory: a wrong answer stated confidently is worse "
    "than no answer."
)
