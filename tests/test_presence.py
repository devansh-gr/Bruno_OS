#!/usr/bin/env python3
"""Sleep mode: the gate, the phrases, and that it survives a restart."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from bruno import presence
from bruno.presence import State, Urgency, going_to_sleep, may_speak, waking_up


def main() -> int:
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name:56} {got} (want {want})")

    for t in ["i'm going to bed", "going to sleep", "goodnight", "good night",
              "heading to bed", "turning in", "do not disturb", "don't disturb me"]:
        if not going_to_sleep(t):
            fails.append(f"missed sleep phrase {t!r}")
    check("every sleep phrase is recognised",
          [t for t in ["i'm going to bed", "goodnight", "do not disturb"]
           if not going_to_sleep(t)], [])

    for t in ["good morning", "i'm up", "i'm awake", "wake up bruno", "i'm back"]:
        if not waking_up(t):
            fails.append(f"missed wake phrase {t!r}")
    check("every wake phrase is recognised",
          [t for t in ["good morning", "i'm up"] if not waking_up(t)], [])

    # A request must never be mistaken for either.
    for t in ["what is the time constant", "play some music", "pull up the news",
              "how's the bed frame project going", "what's the morning briefing"]:
        if going_to_sleep(t) or waking_up(t):
            fails.append(f"false trigger on {t!r}")
    check("ordinary requests trigger neither",
          [t for t in ["pull up the news", "what's the morning briefing"]
           if going_to_sleep(t) or waking_up(t)], [])

    asleep, awake = State(asleep=True), State(asleep=False)
    check("asleep blocks chatter", may_speak(Urgency.CHATTER, asleep), False)
    check("asleep blocks normal", may_speak(Urgency.NORMAL, asleep), False)
    check("asleep lets an emergency through", may_speak(Urgency.EMERGENCY, asleep), True)
    check("awake lets everything through",
          all(may_speak(u, awake) for u in Urgency), True)

    # The whole point of persisting: a restart must not forget.
    with tempfile.TemporaryDirectory() as d:
        old = presence.STATE_PATH
        presence.STATE_PATH = Path(d) / "presence.json"
        try:
            presence.save(State(asleep=True, since=123.0,
                                held=["Adam emailed you.", "A run stopped."]))
            back = presence.load()
            check("state survives a restart",
                  (back.asleep, len(back.held)), (True, 2))
            check("and the text is kept, not just a count",
                  back.held[0], "Adam emailed you.")
            # An older state file stored a count. It must load, not crash.
            presence.STATE_PATH.write_text('{"asleep": true, "since": 1, "held": 4}')
            check("an old count-format file still loads", presence.load().held, [])
            presence.STATE_PATH.write_text("{ this is not json")
            check("a corrupt file reads as awake, not as a crash",
                  presence.load().asleep, False)
        finally:
            presence.STATE_PATH = old

    print(f"\n{'all presence checks passed' if not fails else str(len(fails)) + ' failed: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
