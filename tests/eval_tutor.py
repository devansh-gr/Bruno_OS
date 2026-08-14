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


def run_model(gguf: Path, port: int, show_failures: bool, passes: int = 1) -> list[Report]:
    llm = LlamaCppLLM(gguf=gguf, port=port)
    print(f"  loading {gguf.name}...", flush=True)
    t0 = time.perf_counter()
    llm.start()
    llm.warmup()
    print(f"  ready in {time.perf_counter() - t0:.0f}s, asking {len(QUESTIONS)} questions",
          flush=True)

    # Generation is sampled, so a single pass is one draw from a distribution.
    # The first baseline run had the model invent a definition for a fake term;
    # the second had it decline the same question. Reporting one draw as "the
    # number" hides exactly the variance that matters here.
    reports = []
    for p in range(passes):
        report = Report(f"{gguf.name} (pass {p + 1})" if passes > 1 else gguf.name)
        for i, (q, kind, expect) in enumerate(QUESTIONS, 1):
            answer = "".join(llm.stream(q, system=TUTOR_SYSTEM)).strip()
            report.results.append(score(q, kind, expect, answer))
            print(f"\r    pass {p + 1}/{passes}  {i}/{len(QUESTIONS)}", end="", flush=True)
        reports.append(report)
    print()
    llm.stop()

    if show_failures:
        seen = {}
        for rep in reports:
            for r in rep.results:
                if r.confident_wrong:
                    seen.setdefault(r.question, []).append(r)
        if seen:
            print(f"\n  confident-wrong on {len(seen)} distinct question(s):")
            for q, rs in seen.items():
                print(f"    [{rs[0].kind}] {q}   ({len(rs)}/{passes} passes)")
                print(f"      -> {rs[0].answer[:140]}")
    return reports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="GGUF filenames inside models/")
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--passes", type=int, default=3,
                    help="repeats per model; generation is sampled so one pass is one draw")
    a = ap.parse_args()

    names = a.models or [config.LLM_GGUF.name]
    paths = [config.MODELS / n for n in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("missing:", ", ".join(p.name for p in missing))
        return 1

    reports = []
    for i, p in enumerate(paths):
        reports.extend(run_model(p, 8090 + i, a.show_failures, a.passes))
        print()

    print("=" * 66)
    for r in reports:
        print(r.summary())
        print()

    rates = [r.confident_wrong_rate for r in reports]
    if len(reports) > 1:
        print(f"  confident-wrong across {len(reports)} runs: "
              f"best {min(rates):.1f}%  worst {max(rates):.1f}%  "
              f"mean {sum(rates) / len(rates):.1f}%")
        print("  Plan against the worst case, not the mean -- the bad draw is the one")
        print("  that tells him something false with a straight face.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
