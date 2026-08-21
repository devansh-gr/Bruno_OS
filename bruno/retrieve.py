"""Turn a question into evidence the model can read instead of recall.

Three stages, and the middle one is where the work is.

  1. SEARCH  — hand the question to the web, get back a wall of results text.
  2. FOCUS   — a results page is ~5000 characters of navigation, ads and
               snippets. Feeding that to a 4k-context model wastes most of the
               window on chrome. So passages are scored by overlap with the
               question's own content words and only the densest are kept.
  3. GROUND  — the model is told to answer ONLY from those passages, and to say
               so when they do not contain the answer.

Stage 3's prompt matters more than it looks. An earlier version led with "answer
only from the sources", and the model stopped questioning false premises -- it
became a summariser instead of a reader. Premise-checking now comes first.
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
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'\-]{2,}", question.lower())
            if w not in STOP}


@dataclass
class Evidence:
    question: str
    passages: list[str]
    chars_searched: int

    def as_prompt(self) -> str:
        body = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(self.passages, 1))
        return f"Search results:\n{body}\n\nQuestion: {self.question}"


def focus(text: str, question: str, window: int = 420, keep: int = 6) -> list[str]:
    """Keep the windows of text where the question's own words cluster.

    A results page is mostly chrome. Scoring by term overlap and taking the
    densest windows turns 5000 characters into roughly 2000 that are actually
    about the question, which is the difference between fitting in context and
    not.
    """
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
