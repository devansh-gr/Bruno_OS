#!/usr/bin/env python3
"""Wake-phrase checks, including the mis-hearings whisper actually produces."""

from __future__ import annotations

import sys

from bruno.wake import sleeps, strip_wake, wakes

WAKES_ASLEEP = [
    "hey bruno", "Hey Bruno.", "hey, bruno", "hey bruno what's the time constant",
    "Hey Brunno, pull up the news", "hay bruno play some music",
    "yo bruno how's the portfolio", "ok bruno draft an email to adam",
    "hey bruno.  ", "Hey  Bruno. What did claude do overnight",
]
IGNORED_ASLEEP = [
    # The case that made the phrase long in the first place: a friend named Bruno.
    "bruno said he's coming over",
    "i told bruno about the lab",
    "what time is it",
    "can you pass that to bruno",
    "hey, can you check this",
    "brunos coming later",
    "",
]
WAKES_AWAKE = ["bruno pull that up", "Bruno, what about the nvidia one"]
SLEEPERS = ["that's all bruno", "thanks bruno", "go to sleep", "stand down",
            "we're done", "shut up"]


def main() -> int:
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name[:58]:60} {got} (want {want})")

    for t in WAKES_ASLEEP:
        check(f"wakes asleep: {t!r}", wakes(t, strict=True), True)
    for t in IGNORED_ASLEEP:
        check(f"ignored asleep: {t!r}", wakes(t, strict=True), False)
    for t in WAKES_AWAKE:
        check(f"bare name wakes when awake: {t!r}", wakes(t, strict=False), True)
        check(f"bare name ignored when asleep: {t!r}", wakes(t, strict=True), False)
    for t in SLEEPERS:
        check(f"sleeps: {t!r}", sleeps(t), True)

    for said, want in [
        ("hey bruno what's the time constant", "what's the time constant"),
        ("Hey Bruno, pull up the news", "pull up the news"),
        ("bruno play bohemian rhapsody", "play bohemian rhapsody"),
        ("what's the time constant", "what's the time constant"),
    ]:
        got = strip_wake(said)
        if got != want:
            fails.append(f"strip {said!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  strip {said[:38]!r:40} -> {got!r}")

    print(f"\n{'all wake checks passed' if not fails else str(len(fails)) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
