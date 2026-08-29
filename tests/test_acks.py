#!/usr/bin/env python3
"""Which turns get an immediate acknowledgement, and which must not."""

from __future__ import annotations

import sys
import types

from bruno.tools.base import Tier
from bruno.voice import Voice


def main() -> int:
    fails = []
    v = Voice.__new__(Voice)
    route = lambda tier, tool: types.SimpleNamespace(tier=tier, tool=tool)

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name:54} {got!r} (want {want!r})")

    # Reflex is faster than the acknowledgement would take to say. An ack there
    # makes the exchange slower while sounding more responsive.
    check("reflex gets no ack", v._ack_for(route(Tier.REFLEX, None)), None)
    check("codegen is announced before the terminal opens",
          v._ack_for(route(Tier.HEAVY, None)) is not None, True)
    check("tutoring is acked because it retrieves",
          v._ack_for(route(Tier.AGENT, None)) is not None, True)

    for tool in ("news", "briefing", "market_quote", "check_email", "play_song"):
        if v._ack_for(route(Tier.AGENT, tool)) is None:
            fails.append(f"{tool} has no ack")
    check("every network tool has an ack",
          [t for t in ("news", "briefing", "market_quote") if not v._ack_for(route(Tier.AGENT, t))], [])

    # Acks are spoken, so they are held to the same rule as any other reply: never
    # a bare monosyllable, which is unintelligible through a speaker.
    from bruno import config
    bad = [a for a in v.ACKS.values() if a.lower().strip(" .!?") in config.BANNED_UTTERANCES]
    check("no ack is a banned monosyllable", bad, [])
    check("every ack is more than one word",
          [a for a in v.ACKS.values() if len(a.split()) < 2], [])

    # An ack that names the wrong action is worse than none.
    check("the email ack mentions mail", "mail" in v.ACKS["check_email"].lower(), True)

    print(f"\n{'all ack checks passed' if not fails else str(len(fails)) + ' failed: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
