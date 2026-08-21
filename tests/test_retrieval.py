#!/usr/bin/env python3
"""Retrieval unit checks. Fast, no model, no benchmark.

These exist because every bug in this pipeline so far has been silent: a throttle
that looked like no-results, chrome that outranked content, a token cap that
looked like a wrong answer. None of them raised. All of them would have been
caught by an assertion cheaper than the benchmark that eventually found them.

    python -m tests.test_retrieval
"""

from __future__ import annotations

import sys

from bruno import retrieve, websearch

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


def main() -> int:
    print("retrieval\n")

    # --- key term extraction ---------------------------------------------
    terms = retrieve.key_terms("What is the name of India's first nano-satellite?")
    check("keeps content words", {"india", "nano-satellite"} <= terms, str(sorted(terms)))
    check("drops stopwords", not ({"what", "the", "is", "of"} & terms), str(sorted(terms)))

    # --- focus ------------------------------------------------------------
    text = ("irrelevant filler " * 40 + "the Jugnu satellite was India's first nano-satellite "
            + "more filler " * 40)
    passages = retrieve.focus(text, "India's first nano-satellite")
    check("focus finds the dense window",
          any("jugnu" in p.lower() for p in passages), str(passages[:1]))
    check("focus keeps it small", sum(len(p) for p in passages) < len(text),
          f"{sum(len(p) for p in passages)} vs {len(text)}")

    # Overlapping windows should not be returned as separate passages.
    dense = "satellite india nano " * 30
    ps = retrieve.focus(dense, "india nano satellite")
    starts_distinct = len(ps) == len(set(ps))
    check("focus does not return duplicate slices", starts_distinct, f"{len(ps)} passages")

    check("focus on empty text returns nothing", retrieve.focus("", "anything") == [])
    check("focus with no matching terms returns nothing",
          retrieve.focus("completely unrelated prose here", "xylophone quasar") == [])

    # --- chrome stripping --------------------------------------------------
    chrome = ("query at DuckDuckGo All Regions Argentina Australia Austria Belgium "
              "Safe search: moderate Any time Past day Past week REAL CONTENT HERE")
    stripped = websearch._strip(f"<html><body>{chrome}</body></html>")
    check("chrome filter removes the region picker", "Argentina" not in stripped, stripped[:80])
    check("chrome filter keeps real content", "REAL CONTENT HERE" in stripped, stripped[:80])

    # --- throttle is not absence ------------------------------------------
    check("Throttled is its own error type", issubclass(websearch.Throttled, RuntimeError))
    check("empty query returns empty, does not raise", websearch.search("") == "")

    # --- grounded prompt ---------------------------------------------------
    ev = retrieve.Evidence("Q?", ["passage one", "passage two"], 1000)
    prompt = ev.as_prompt()
    check("prompt carries every passage",
          "passage one" in prompt and "passage two" in prompt)
    check("prompt carries the question", "Q?" in prompt)
    check("grounded prompt checks premise before answering",
          prompt.find("Question:") > prompt.find("passage one"))
    check("grounded system forbids filling from memory",
          "from memory" in retrieve.GROUNDED_SYSTEM.lower())
    check("grounded system gives an exact decline phrase",
          "i don't know" in retrieve.GROUNDED_SYSTEM.lower())

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("all retrieval checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
