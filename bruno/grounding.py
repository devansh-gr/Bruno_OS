"""Grounded answering: fetch sources first, answer only from them.

A bigger model reduces hallucination. Retrieval largely removes the *category*,
because the model is no longer recalling a fact — it is reading one. That matters
more here than parameter count: Devansh's stated requirement is a tutor that
doesn't bluff, and a 3B reading a Wikipedia extract beats a 30B recalling from
weights on exactly the questions where recall is shaky.

Deliberately not a vector database. For "what is a Kalman filter" the useful
source is an encyclopedia article, and a keyword lookup finds it in 200ms with no
index to build, no embeddings to keep fresh, and no similarity threshold to tune.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

UA = {"User-Agent": "BrunoOS/0.1 (personal assistant; contact devansh@zeonsystems.ai)"}
WIKI = "https://en.wikipedia.org/w/api.php"


@dataclass
class Source:
    title: str
    text: str
    url: str


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


STOPWORDS = {
    "what", "does", "do", "did", "is", "are", "was", "the", "a", "an", "for", "in",
    "of", "to", "and", "or", "how", "why", "when", "which", "who", "stand", "mean",
    "estimate", "produce", "used", "use", "with", "that", "this", "it", "its",
    "given", "since", "because", "explain", "define", "typical", "roughly", "most",
}


def key_terms(question: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]{1,}", question)
            if w.lower() not in STOPWORDS}


def relevant(question: str, source: "Source", threshold: float = 0.25) -> bool:
    """Does this source actually address the question?

    Wikipedia keyword search cannot disambiguate a bare question -- "What does ATP
    stand for?" returns the tennis tour, and the model then answers confidently
    from it, which reads exactly like a hallucination but is a retrieval failure.
    Three query-rewriting strategies were tried (domain keywords, a single domain
    word, key-terms-only) and none fixed it reliably.

    So instead of chasing perfect retrieval, make its failures safe: if the
    question's own terms barely appear in what came back, treat it as no source
    and let the model decline.
    """
    terms = key_terms(question)
    if not terms:
        return True
    haystack = f"{source.title} {source.text}".lower()
    hits = sum(1 for t in terms if t in haystack)
    return hits / len(terms) >= threshold


def wiki_lookup(query: str, sentences: int = 6) -> list[Source]:
    """Search Wikipedia and return lead extracts. Two calls: find, then fetch."""
    search = query
    try:
        r = requests.get(WIKI, headers=UA, timeout=10, params={
            "action": "query", "list": "search", "srsearch": search,
            "srlimit": 2, "format": "json",
        })
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
    except (requests.RequestException, ValueError):
        return []
    if not hits:
        return []

    titles = "|".join(h["title"] for h in hits)
    try:
        r = requests.get(WIKI, headers=UA, timeout=10, params={
            "action": "query", "prop": "extracts", "titles": titles,
            "exintro": 1, "explaintext": 1, "format": "json",
        })
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
    except (requests.RequestException, ValueError):
        return []

    out = []
    for page in pages.values():
        extract = _clean(page.get("extract", ""))
        if not extract:
            continue
        trimmed = " ".join(re.split(r"(?<=[.!?])\s+", extract)[:sentences])
        src = Source(
            title=page.get("title", ""),
            text=trimmed,
            url=f"https://en.wikipedia.org/wiki/{page.get('title', '').replace(' ', '_')}",
        )
        if relevant(query, src):
            out.append(src)
    return out


GROUNDED_SYSTEM = (
    "You are Bruno, answering a question out loud for Devansh, a biomedical "
    "engineering student. Answer in two or three sentences, direct and technical.\n\n"
    "FIRST: check the question's premise against the sources. Questions often assert "
    "something false as if it were established ('since X is true, why Y?'). If the "
    "sources contradict the premise, say so before anything else and correct it. "
    "Do not answer a question built on a false premise as though the premise held.\n\n"
    "THEN: answer from the sources. If they do not contain the answer, say so plainly "
    "rather than filling the gap from memory.\n\n"
    "Do not mention that you were given sources; just answer."
)


def build_prompt(question: str, sources: list[Source]) -> str:
    body = "\n\n".join(f"[{i}] {s.title}: {s.text}" for i, s in enumerate(sources, 1))
    return f"Sources:\n{body}\n\nQuestion: {question}"


def ground(question: str) -> tuple[str | None, list[Source]]:
    """Return (prompt, sources), or (None, []) when nothing useful was found —
    in which case the caller should fall back and say the answer is ungrounded."""
    sources = wiki_lookup(question)
    if not sources:
        return None, []
    return build_prompt(question, sources), sources
