"""Bruno speaking first.

Everything else answers. This is the only part that starts a conversation, and it
is the piece that decides whether living with him feels like an assistant or a
search box.

Design constraints, all of them learned rather than assumed:
  - Nothing is said twice, across restarts. Being told about the same email each
    time a server reloads is worse than not being told.
  - Nothing interrupts a conversation already in progress.
  - Everything goes through speak_unprompted, so the sleep gate cannot be bypassed
    by a watcher that forgets it exists.
  - A watcher that throws is skipped, not fatal. Mail.app being closed must not
    take the loop down.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Callable

from bruno import config
from bruno.presence import Urgency

SEEN_PATH = config.ROOT / "proactive_seen.json"
SEEN_MAX = 500


@dataclass
class Notice:
    key: str                       # stable across polls; identity for dedupe
    text: str                      # said out loud, so one or two sentences
    urgency: Urgency = Urgency.NORMAL


@dataclass
class Watcher:
    name: str
    every: float                   # seconds between checks
    check: Callable[[], list[Notice]]
    last_run: float = 0.0
    failures: int = 0

    def due(self, now: float) -> bool:
        # Back off a failing watcher rather than hammering it every interval.
        delay = self.every * (2 ** min(self.failures, 4))
        return now - self.last_run >= delay


def _load_seen() -> list[str]:
    try:
        return list(json.loads(SEEN_PATH.read_text()))[-SEEN_MAX:]
    except Exception:
        return []


def _save_seen(keys: list[str]) -> None:
    try:
        SEEN_PATH.write_text(json.dumps(keys[-SEEN_MAX:]))
    except Exception:
        pass


class Proactive:
    """Polls watchers and hands whatever they raise to `say`.

    `say` is Voice.speak_unprompted, which applies the sleep gate and returns
    whether it actually spoke. A held notice is still marked seen: it was raised,
    he will be told the count when he wakes, and repeating it later out of context
    would be worse than dropping it.
    """

    def __init__(self, say, busy=lambda: False, tick: float = 20.0):
        self.say = say
        self.busy = busy          # True while a turn is in progress
        self.tick = tick
        self.watchers: list[Watcher] = []
        self.seen: list[str] = _load_seen()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def watch(self, name: str, every: float, check) -> None:
        self.watchers.append(Watcher(name, every, check))

    def poll(self) -> list[Notice]:
        """One pass. Returns what was raised, before the gate."""
        now = time.monotonic()
        out: list[Notice] = []
        for w in self.watchers:
            if not w.due(now):
                continue
            w.last_run = now
            try:
                out.extend(w.check() or [])
                w.failures = 0
            except Exception:
                w.failures += 1
        return out

    def deliver(self, notices: list[Notice]) -> int:
        """Say what has not been said. Most urgent first."""
        said = 0
        for n in sorted(notices, key=lambda n: -int(n.urgency)):
            if n.key in self.seen:
                continue
            self.seen.append(n.key)
            _save_seen(self.seen)
            if self.say(n.text, n.urgency):
                said += 1
        return said

    def _loop(self) -> None:
        while not self._stop.wait(self.tick):
            if self.busy():
                continue      # never cut across a turn already happening
            try:
                self.deliver(self.poll())
            except Exception:
                pass

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
