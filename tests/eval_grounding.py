#!/usr/bin/env python3
"""The fair grounding test: same questions, with and without retrieval.

Every question has no answer in the weights, so this measures the thing the
earlier test could not — whether retrieval turns a guess into a fact.

Three outcomes per question, and the distinction matters:
    RIGHT     — matched the live ground truth
    DECLINED  — said it didn't know. Not a win, but not a lie either.
    BLUFFED   — asserted something, and it was wrong. The failure that counts.

    python -m tests.eval_grounding
"""

from __future__ import annotations

import sys

from bruno.backends.llm_llamacpp import LlamaCppLLM
from bruno.grounding import GROUNDED_SYSTEM, ground
from bruno.tools import calc, media, public, workspace  # noqa: F401 — registers tools
from tests.eval_tutor import NO_SOURCE, TUTOR_SYSTEM
from tests.tutor_score import declined
from tests.unknowable_questions import QUESTIONS, truth_for


def ask_ungrounded(llm, q: str) -> str:
    return "".join(llm.stream(q, system=TUTOR_SYSTEM)).strip()


# Only tools that RETRIEVE count as grounding. The calculator is not a source of
# facts, and letting the router hand it a spec question produced answers like
# "159.155" for an input impedance -- a leftover cutoff frequency, scored as a
# grounding result. That polluted the first run and inverted its conclusion.
RETRIEVAL_TOOLS = {"market_quote", "news", "video_search"}


def ask_grounded(llm, q: str) -> str:
    """Retrieval first, and for live facts the tools are the source, not Wikipedia."""
    from bruno.router import Router
    from bruno.tools.base import REGISTRY

    route = Router(llm=llm).route(q)
    tool = REGISTRY.get(route.tool) if route.tool in RETRIEVAL_TOOLS else None
    if tool is not None:
        result = tool.run(**{tool.arg: route.query})
        if result.ok:
            return result.text

    prompt, sources = ground(q)
    if prompt is None:
        return "".join(llm.stream(q, system=NO_SOURCE)).strip()
    return "".join(llm.stream(prompt, system=GROUNDED_SYSTEM)).strip()


def _numbers_in(text: str) -> list[float]:
    import re
    out = []
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            out.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return out


def classify(answer: str, truth: tuple[str, ...]) -> str:
    a = answer.lower()
    for t in truth:
        if t.startswith("~NUM~"):
            # Within 3%: the question is "did it look it up or guess", not
            # whether it matched to the cent while the market moved.
            target = float(t[5:])
            if any(abs(n - target) / target < 0.03 for n in _numbers_in(answer) if n):
                return "RIGHT"
        elif t.lower() in a:
            return "RIGHT"
    if declined(a):
        return "DECLINED"
    return "BLUFFED"


def main() -> int:
    llm = LlamaCppLLM()
    llm.start()
    llm.warmup()

    print(f"  {len(QUESTIONS)} questions with no answer in the weights\n")
    tally = {"ungrounded": {}, "grounded": {}}
    for mode in ("ungrounded", "grounded"):
        tally[mode] = {"RIGHT": 0, "DECLINED": 0, "BLUFFED": 0}

    for q in QUESTIONS:
        truth = truth_for(q)   # fetched per question, immediately before asking
        if not truth:
            print(f"  ?? skipped (no ground truth available): {q.question[:50]}")
            continue
        print(f"  {q.question[:62]}")
        for mode, fn in (("ungrounded", ask_ungrounded), ("grounded", ask_grounded)):
            answer = fn(llm, q.question)
            verdict = classify(answer, truth)
            tally[mode][verdict] += 1
            print(f"    {mode:11} {verdict:9} {answer[:66]}")
        print()

    llm.stop()
    print("=" * 62)
    for mode in ("ungrounded", "grounded"):
        t = tally[mode]
        n = sum(t.values()) or 1
        print(f"  {mode:11} right {t['RIGHT']:2}/{n}   declined {t['DECLINED']:2}   "
              f"BLUFFED {t['BLUFFED']:2}  ({t['BLUFFED'] / n * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
