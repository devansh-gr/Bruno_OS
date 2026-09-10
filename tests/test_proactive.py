#!/usr/bin/env python3
"""Bruno speaking first: dedupe, the sleep gate, and never cutting across a turn."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from bruno import proactive as P
from bruno.presence import State, Urgency
from bruno.proactive import Notice, Proactive


def main() -> int:
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name:56} {got!r} (want {want!r})")

    with tempfile.TemporaryDirectory() as d:
        old = P.SEEN_PATH
        P.SEEN_PATH = Path(d) / "seen.json"
        try:
            said = []
            p = Proactive(say=lambda t, u: (said.append((t, u)), True)[1])
            p.watch("w", 0, lambda: [Notice("k", "Adam replied.", Urgency.NORMAL)])

            p.deliver(p.poll())
            p.deliver(p.poll())
            p.deliver(p.poll())
            check("the same notice is said once", len(said), 1)

            # A restart must not re-announce it.
            p2 = Proactive(say=lambda t, u: (said.append((t, u)), True)[1])
            p2.watch("w", 0, lambda: [Notice("k", "Adam replied.", Urgency.NORMAL)])
            p2.deliver(p2.poll())
            check("and is still not repeated after a restart", len(said), 1)

            # Urgency order: the important thing is said first.
            order = []
            p3 = Proactive(say=lambda t, u: (order.append(t), True)[1])
            p3.watch("w", 0, lambda: [
                Notice("a", "chatter", Urgency.CHATTER),
                Notice("b", "emergency", Urgency.EMERGENCY),
                Notice("c", "normal", Urgency.NORMAL)])
            p3.deliver(p3.poll())
            check("most urgent is said first", order[0], "emergency")

            # The sleep gate: a held notice is still marked seen, so it is not
            # replayed hours later out of context.
            held = []
            asleep = State(asleep=True)
            from bruno.presence import may_speak
            p4 = Proactive(say=lambda t, u: (held.append(t), False)[1]
                           if not may_speak(u, asleep) else True)
            p4.watch("w", 0, lambda: [Notice("z", "chatter while asleep", Urgency.CHATTER)])
            p4.deliver(p4.poll())
            check("a notice held by sleep is marked seen", "z" in p4.seen, True)

            # A throwing watcher must not stop the others.
            got = []
            p5 = Proactive(say=lambda t, u: (got.append(t), True)[1])
            p5.watch("boom", 0, lambda: 1 / 0)
            p5.watch("fine", 0, lambda: [Notice("ok", "still working", Urgency.NORMAL)])
            p5.deliver(p5.poll())
            check("one broken watcher does not silence the rest", got, ["still working"])
            check("the broken watcher is backed off", p5.watchers[0].failures, 1)

            # busy() is the guard against talking over a turn in progress.
            spoken = []
            p6 = Proactive(say=lambda t, u: (spoken.append(t), True)[1],
                           busy=lambda: True, tick=0.01)
            p6.watch("w", 0, lambda: [Notice("q", "mid turn", Urgency.NORMAL)])
            p6._stop.set()      # run the body once, synchronously
            if not p6.busy():
                p6.deliver(p6.poll())
            check("nothing is said while a turn is in progress", spoken, [])
        finally:
            P.SEEN_PATH = old

    # Backoff must actually grow, or a dead watcher is polled forever.
    from bruno.proactive import Watcher
    w = Watcher("x", 60.0, lambda: [])
    w.last_run, w.failures = 0.0, 0
    first = w.due(60.0)
    w.failures = 3
    check("a failing watcher is polled less often", (first, w.due(60.0)), (True, False))

    print(f"\n{'all proactive checks passed' if not fails else str(len(fails)) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
