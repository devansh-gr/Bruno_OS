"""Whether Devansh is available to be spoken to.

He asked for this directly: Bruno speaks first, unless he says he is sleeping, in
which case only emergencies. That is two separate things and only the second is
here. This holds the state and gates outbound speech; what counts as an emergency
is decided by whoever raises it, not guessed at here.

Persisted, because the state has to survive a restart. Telling him at 2am that
Bruno forgot he was asleep because a server was reloaded is the whole failure.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum

from bruno import config

STATE_PATH = config.ROOT / "presence.json"


class Urgency(IntEnum):
    """Ordered, so the gate is a comparison rather than a table."""

    CHATTER = 0     # interesting, not important. The briefing, a finished build.
    NORMAL = 1      # worth saying now. A reply from a human, a failed deploy.
    EMERGENCY = 2   # worth waking him for. Nothing else gets through while asleep.


SLEEPING = re.compile(
    r"\b(i'?m going to (bed|sleep)|going to (bed|sleep)|good ?night|"
    r"i'?m off to bed|heading to bed|turning in|i'?m asleep|do not disturb|"
    r"don'?t disturb me|leave me alone)\b", re.I)

WAKING = re.compile(
    r"\b(good ?morning|i'?m up|i'?m awake|wake up bruno|i'?m back|"
    r"stop do not disturb|disturb me)\b", re.I)


@dataclass
class State:
    asleep: bool = False
    since: float = 0.0
    # Raised while asleep and held back. The text, not a count: "four things
    # waited" is useless, and he cannot ask what they were once they are gone.
    held: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"asleep": self.asleep, "since": self.since,
                           "held": list(self.held)[-20:]})


def load() -> State:
    if not STATE_PATH.exists():
        return State()
    try:
        d = json.loads(STATE_PATH.read_text())
        held = d.get("held", [])
        if isinstance(held, int):        # older format stored a count
            held = []
        return State(bool(d.get("asleep")), float(d.get("since", 0)), list(held))
    except (json.JSONDecodeError, ValueError, TypeError):
        return State()


def save(s: State) -> None:
    STATE_PATH.write_text(s.to_json())


def going_to_sleep(text: str) -> bool:
    return bool(SLEEPING.search(text or ""))


def waking_up(text: str) -> bool:
    return bool(WAKING.search(text or ""))


def may_speak(urgency: Urgency, state: State | None = None) -> bool:
    """Can Bruno start talking unprompted right now?

    Only gates speech he initiates. A direct question is always answered: if he is
    awake enough to ask, he is awake enough to hear the answer.
    """
    s = state if state is not None else load()
    return urgency >= Urgency.EMERGENCY if s.asleep else True


def hours_asleep(s: State) -> float:
    return (time.time() - s.since) / 3600 if s.asleep and s.since else 0.0
