#!/usr/bin/env python3
"""Run the tutor benchmark against one or more local models.

    python -m tests.eval_tutor                          current configured model
    python -m tests.eval_tutor --models a.gguf b.gguf   compare several
    python -m tests.eval_tutor --show-failures          print what it got wrong

Each model gets its own llama-server on a scratch port so a comparison run is
apples to apples: same questions, same prompt, same decoding settings.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from bruno import config
from bruno.backends.llm_llamacpp import LlamaCppLLM
from tests.tutor_questions import QUESTIONS
from tests.tutor_score import Report, score

# The tutor prompt under test. Permission to decline is explicit, because a model
# that has not been told it may say "I don't know" will invent rather than admit.
TUTOR_SYSTEM = (
    "You are Bruno, answering a question out loud for Devansh, a biomedical "
    "engineering student. Answer in two or three sentences. "
    "Be direct and technical. "
    "If you do not know something, or the question asks about private information, "
    "future events, or a term you do not recognise, say so plainly instead of "
    "guessing. If a question contains a false premise, correct it before answering."
)


def run_model(gguf: Path, port: int, show_failures: bool) -> Report:
    llm = LlamaCppLLM(gguf=gguf, port=port)
    print(f"  loading {gguf.name}...", flush=True)
    t0 = time.perf_counter()
    llm.start()
    llm.warmup()
    print(f"  ready in {time.perf_counter() - t0:.0f}s, asking {len(QUESTIONS)} questions",
          flush=True)

    report = Report(gguf.name)
    for i, (q, kind, expect) in enumerate(QUESTIONS, 1):
        answer = "".join(llm.stream(q, system=TUTOR_SYSTEM)).strip()
        report.results.append(score(q, kind, expect, answer))
        print(f"\r    {i}/{len(QUESTIONS)}", end="", flush=True)
    print()
    llm.stop()

    if show_failures:
        bad = [r for r in report.results if r.confident_wrong]
        if bad:
            print(f"\n  confident-wrong ({len(bad)}):")
            for r in bad:
                print(f"    [{r.kind}] {r.question}")
                print(f"      -> {r.answer[:150]}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="GGUF filenames inside models/")
    ap.add_argument("--show-failures", action="store_true")
    a = ap.parse_args()

    names = a.models or [config.LLM_GGUF.name]
    paths = [config.MODELS / n for n in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("missing:", ", ".join(p.name for p in missing))
        return 1

    reports = []
    for i, p in enumerate(paths):
        reports.append(run_model(p, 8090 + i, a.show_failures))
        print()

    print("=" * 66)
    for r in reports:
        print(r.summary())
        print()

    if len(reports) > 1:
        best = min(reports, key=lambda r: r.confident_wrong_rate)
        print(f"  lowest confident-wrong: {best.label} at {best.confident_wrong_rate:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
