#!/usr/bin/env python3
"""Route what the wake stripper actually hands over, not idealised text.

Every case in routing_cases.py is already clean. In the live loop the router sees
whatever strip_wake left behind, and whisper's punctuation and capitalisation come
with it. This checks the two agree.
"""

from __future__ import annotations

import sys

from bruno.wake import strip_wake

# How he would actually say these out loud, with the address attached.
SPOKEN = [
    ("hey bruno what's the time constant of a ten k resistor",
     "what's the time constant of a ten k resistor"),
    ("Hey Bruno, pull up the news.", "pull up the news"),
    ("hey bruno play bohemian rhapsody", "play bohemian rhapsody"),
    ("Bruno, how's the bruno os project coming along",
     "how's the bruno os project coming along"),
    ("hey bruno check my inbox", "check my inbox"),
    ("yo bruno what did claude do overnight", "what did claude do overnight"),
    ("hey, bruno. explain what a kalman filter is", "explain what a kalman filter is"),
    ("hey bruno write me a python script to rename files",
     "write me a python script to rename files"),
]


def main() -> int:
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name[:56]:58} {got!r}")

    for said, want in SPOKEN:
        check(f"strip {said[:34]!r}", strip_wake(said), want)

    # The stripped text must route the same as the clean text it becomes. Anything
    # else means the address is shifting classifications, which the router was never
    # scored against.
    from bruno.router import Router
    from bruno.tools import brief, calc, email, media, public, workspace  # noqa: F401

    r = Router()
    r.route("warm up")
    for said, clean in SPOKEN:
        a, b = r.route(strip_wake(said)), r.route(clean)
        same = (a.tier, a.tool) == (b.tier, b.tool)
        if not same:
            fails.append(f"routing differs for {said!r}")
        print(f"  {'PASS' if same else 'FAIL'}  routes as clean text: {clean[:38]!r:42} "
              f"{a.tier.value}/{a.tool}")

    print(f"\n{'all spoken-routing checks passed' if not fails else str(len(fails)) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
