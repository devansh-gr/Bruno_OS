"""Short-term conversation memory.

Exists because half of what Devansh actually says depends on the turn before it:
"find me a video that explains *this*", "ping Claude and ask what *it's* been up
to". A router that classifies each utterance in isolation cannot resolve those.

The privacy wrinkle: conversation history is a prompt input like any other, so a
turn that touched email or portfolio data must not ride along into a cloud call
just because it happened to be recent. Every turn therefore carries the privacy
class of whatever produced it, and `context_for` filters by destination rather
than handing over the whole buffer.

Scope is deliberately one session — nothing is written to disk yet. Persistent
history across restarts is a database and a retrieval problem, and it is not
needed to make pronouns work.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from bruno.tools.base import Privacy

MAX_TURNS = 12          # what the buffer holds
CONTEXT_TURNS = 4       # what the router is shown — enough for pronouns, short enough to stay fast


@dataclass
class Turn:
    user: str
    bruno: str
    privacy: Privacy
    tool: str | None = None

    def as_lines(self) -> str:
        return f"Devansh: {self.user}\nBruno: {self.bruno}"


@dataclass
class Conversation:
    turns: deque[Turn] = field(default_factory=lambda: deque(maxlen=MAX_TURNS))

    def add(self, user: str, bruno: str, privacy: Privacy, tool: str | None = None) -> None:
        self.turns.append(Turn(user.strip(), bruno.strip(), privacy, tool))

    def clear(self) -> None:
        self.turns.clear()

    def context_for(self, destination: Privacy, limit: int = CONTEXT_TURNS) -> str:
        """Recent turns this destination is allowed to see.

        A cloud destination gets only cloud_ok turns. Note this *drops* private
        turns rather than truncating at the first one — losing a turn of context
        is a worse outcome than losing all of it, but leaking is worse than both.
        """
        allowed = list(self.turns)
        if destination is Privacy.CLOUD_OK:
            allowed = [t for t in allowed if t.privacy is Privacy.CLOUD_OK]
        recent = allowed[-limit:]
        return "\n".join(t.as_lines() for t in recent)

    def last_subject(self) -> str | None:
        """Best guess at what 'this' or 'it' refers to — the most recent turn that
        had real content. Used only as a hint to the router, never as a fact."""
        for t in reversed(self.turns):
            if len(t.user.split()) >= 3:
                return t.user
        return None

    def __len__(self) -> int:
        return len(self.turns)
