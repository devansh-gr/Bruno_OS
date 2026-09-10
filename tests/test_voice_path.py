#!/usr/bin/env python3
"""The voice loop end to end, with synthesized speech instead of a microphone.

Every piece has its own test; none of them proves the pieces connect. This drives
a real Turn through wake matching, speaker verification, routing and the agent, and
asserts the transitions rather than the audio.
"""

from __future__ import annotations

import sys
import types

import numpy as np

from bruno import config
from bruno.listen import Turn

SR = config.SAMPLE_RATE
_tts = None


def say(text: str) -> np.ndarray:
    global _tts
    if _tts is None:
        from bruno.backends.tts_piper import PiperTTS
        _tts = PiperTTS()
        _tts.warmup()
    a = _tts.synth(text)
    n = int(len(a) * SR / _tts.sample_rate)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def turn_of(text: str) -> Turn:
    a = say(text)
    return Turn(audio=a, seconds=len(a) / SR, speech_ms=len(a) / SR * 1000)


def main() -> int:
    from bruno.voice import Voice

    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name:54} {got!r} (want {want!r})")

    # Build a Voice without booting whisper, TTS or the router.
    v = Voice.__new__(Voice)
    v.awake, v.awake_until, v.pending = False, 0.0, None
    v.speak, v.speaker, v.tts = False, None, None
    v.full_duplex = False
    v.listener = types.SimpleNamespace(vad=types.SimpleNamespace(reset=lambda: None),
                                       paused=False)
    v._speaking = types.SimpleNamespace(set=lambda: None, clear=lambda: None,
                                        is_set=lambda: False)
    v._interrupted = False
    v._say = lambda text: None

    quiet = turn_of("hello there")

    check("asleep, unaddressed speech is ignored",
          v._admit("what time is it", quiet), None)
    check("asleep, a friend named Bruno does not wake him",
          v._admit("bruno said he's coming over", quiet), None)
    check("the wake phrase wakes him and strips itself",
          v._admit("hey bruno what's the time constant", quiet), "what's the time constant")
    check("he is now awake", v.awake, True)
    check("awake, the bare name works",
          v._admit("bruno pull up the news", quiet), "pull up the news")
    check("awake, a plain sentence is accepted",
          v._admit("what about nvidia", quiet), "what about nvidia")
    check("the phrase alone is a summons, not a request",
          v._admit("hey bruno", quiet), "")

    v.awake = True
    check("he closes the conversation on request",
          v._admit("that's all bruno", quiet), None)
    check("and is asleep afterwards", v.awake, False)

    # The window must actually expire, or a conversation two rooms away reaches him.
    import time as _t
    v.awake, v.awake_until = True, _t.perf_counter() - 1
    check("an expired window drops him back to asleep",
          v._admit("what about nvidia", quiet), None)

    rc = sleep_cycle()
    print(f"\n{'all voice-path checks passed' if not fails and not rc else 'failed'}")
    return 1 if (fails or rc) else 0




def sleep_cycle() -> int:
    """Going to sleep, holding notices, and waking up, through the real methods.

    The type of `held` changed from a count to a list and one reset was missed, so
    the first notice held while asleep raised AttributeError inside the proactive
    thread. Nothing surfaced: the loop just stopped speaking.
    """
    import tempfile
    import types as _t
    from pathlib import Path

    from bruno import presence
    from bruno.presence import Urgency
    from bruno.voice import Voice

    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name:54} {got!r} (want {want!r})")

    with tempfile.TemporaryDirectory() as d:
        old = presence.STATE_PATH
        presence.STATE_PATH = Path(d) / "presence.json"
        try:
            said = []
            v = Voice.__new__(Voice)
            v.awake, v.awake_until, v.pending = False, 0.0, None
            v.speak, v.speaker, v.tts = False, None, None
            v.full_duplex, v._in_turn, v._interrupted = False, False, False
            v.listener = _t.SimpleNamespace(vad=_t.SimpleNamespace(reset=lambda: None),
                                            paused=False)
            v._speaking = _t.SimpleNamespace(set=lambda: None, clear=lambda: None,
                                             is_set=lambda: False)
            v._say = lambda text: said.append(text)
            v.presence = presence.load()
            quiet = turn_of("hello")

            # awake_until must be in the future, or _admit expires the window first
            # and the turn is correctly ignored as unaddressed.
            import time as _t2
            v.awake, v.awake_until = True, _t2.perf_counter() + 60
            v._admit("i'm going to bed", quiet)
            check("going to bed sets asleep", v.presence.asleep, True)
            check("and held is a list, not a count", v.presence.held, [])

            # The line that used to crash.
            ok = v.speak_unprompted("Adam emailed you.", Urgency.NORMAL)
            check("a normal notice is held while asleep", ok, False)
            check("and its text is kept", v.presence.held, ["Adam emailed you."])

            check("an emergency still gets through",
                  v.speak_unprompted("Your flight is in two hours.", Urgency.EMERGENCY), True)

            said.clear()
            v.awake, v.awake_until = True, _t2.perf_counter() + 60
            v._admit("good morning", quiet)
            check("waking clears the state", v.presence.asleep, False)
            check("and he is told what waited", "Adam emailed you." in said[0], True)
        finally:
            presence.STATE_PATH = old

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
