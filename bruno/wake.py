"""Wake phrase matching: "Hey Bruno".

Two words, not four, because speaker verification carries the load the extra words
used to. The original phrase was "Hey Bruno, wake up" specifically so a friend named
Bruno could not trigger it. A voiceprint solves that directly -- the phrase says
"I am addressing you", the voiceprint says "and it is me" -- so the phrase gets to
be short and natural.

Matching runs on the transcript rather than a dedicated wake-word model. Whisper is
already in the loop and already fast (38ms with the truncated encoder), and a
dedicated model would be a second thing to train and carry for a phrase whisper
catches anyway.

The tolerance is fuzzy on purpose. Whisper writes "hey, bruno." and "hey bruno" and
"hey brunno" for the same sound, and a speaker two metres from a laptop mic gets the
worst of it.
"""

from __future__ import annotations

import re

# Whisper's actual spellings of the name, collected rather than imagined. "Bruno"
# is not in its bias vocabulary, so it reaches for common words that sound alike.
_NAME = r"(?:bruno|brunno|bruneau|brunos|brown[oe]|bru no|prune?o|bueno)"
_HEY = r"(?:hey|hay|hi|yo|ok|okay|hey there)"

WAKE = re.compile(rf"\b{_HEY}\b[\s,.\-!]*{_NAME}\b", re.I)
# Bare name alone still wakes him, because in practice that is what gets said once
# a conversation is running. The voiceprint is what makes this safe.
BARE = re.compile(rf"^[\s,.]*{_NAME}\b[\s,.!?]*", re.I)

# Ending the conversation explicitly, so he stops listening without a timeout.
SLEEP = re.compile(
    rf"\b(?:that'?s all|thanks?\s+{_NAME}|go to sleep|nevermind|never mind|"
    rf"stand down|dismissed|shut up|stop listening|we'?re done|that'?ll be all)\b",
    re.I)


def wakes(text: str, *, strict: bool = True) -> bool:
    """Does this transcript address Bruno?

    `strict` requires the greeting form and is what runs while he is asleep. Once
    awake the bare name is enough, because nobody says "hey bruno" twice in a row --
    which is the whole difference between this and a phone assistant.
    """
    t = (text or "").strip()
    if not t:
        return False
    if WAKE.search(t):
        return True
    return not strict and bool(BARE.match(t))


def sleeps(text: str) -> bool:
    return bool(SLEEP.search(text or ""))


def strip_wake(text: str) -> str:
    """The request with the address removed.

    "hey bruno what's the time constant" has to reach the router as "what's the
    time constant" -- the router is scored on utterances, and a leading "hey bruno"
    is noise that shifts every classification.
    """
    t = (text or "").strip()
    t = WAKE.sub("", t, count=1)
    t = BARE.sub("", t, count=1)
    return t.strip(" ,.!?-").strip()
