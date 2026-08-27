"""Wake phrase matching, on the whisper transcript rather than a separate model."""

from __future__ import annotations

import re

# Whisper has no bias entry for "Bruno" and reaches for lookalikes.
_NAME = r"(?:bruno|brunno|bruneau|brunos|brown[oe]|bru no|prune?o|bueno)"
_HEY = r"(?:hey|hay|hi|yo|ok|okay|hey there)"

WAKE = re.compile(rf"\b{_HEY}\b[\s,.\-!]*{_NAME}\b", re.I)
# Safe only because the voiceprint gates it.
BARE = re.compile(rf"^[\s,.]*{_NAME}\b[\s,.!?]*", re.I)

# Ending the conversation explicitly, so he stops listening without a timeout.
SLEEP = re.compile(
    rf"\b(?:that'?s all|thanks?\s+{_NAME}|go to sleep|nevermind|never mind|"
    rf"stand down|dismissed|shut up|stop listening|we'?re done|that'?ll be all)\b",
    re.I)


def wakes(text: str, *, strict: bool = True) -> bool:
    """strict=True (asleep) needs the greeting; awake, the bare name is enough."""
    t = (text or "").strip()
    if not t:
        return False
    if WAKE.search(t):
        return True
    return not strict and bool(BARE.match(t))


def sleeps(text: str) -> bool:
    return bool(SLEEP.search(text or ""))


def strip_wake(text: str) -> str:
    """Strip the address; a leading "hey bruno" shifts every router classification."""
    t = (text or "").strip()
    t = WAKE.sub("", t, count=1)
    t = BARE.sub("", t, count=1)
    return t.strip(" ,.!?-").strip()
