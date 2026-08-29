#!/usr/bin/env python3
"""The noise guard, pinned against what whisper actually produced on this build.

These are not invented examples. Every string in NON_SPEECH came back from
whisper.cpp when it was given silence, room noise, a hiss or a click.
"""

from __future__ import annotations

import sys

from bruno.hearing import clean, is_noise

NON_SPEECH = ["you", "(static)", "(water running)", "[drum beats]", "[BLANK_AUDIO]",
              "", "   ", "Thank you.", "Thanks for watching!", ".", "okay",
              "[silence]", "(wind blowing)", "[MUSIC]"]

REAL = ["hey bruno", "what is the time constant of a ten k resistor",
        "pull up the news", "play bohemian rhapsody", "that's all bruno",
        "who won the nobel prize in physics last year"]


def main() -> int:
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name:56} {got} (want {want})")

    bad = [t for t in NON_SPEECH if not is_noise(t)]
    check("every observed non-speech string is rejected", bad, [])

    kept = [t for t in REAL if is_noise(t, speech_ms=1200)]
    check("no real request is rejected", kept, [])

    # The VAD cross-check is what catches a hallucination that looks like words.
    check("a real phrase with no voiced audio is rejected",
          is_noise("hey bruno", speech_ms=80), True)
    check("the same phrase with speech behind it is kept",
          is_noise("hey bruno", speech_ms=900), False)
    check("speech_ms is optional", is_noise("pull up the news"), False)

    check("inline annotation is stripped from real speech",
          clean("[cough] what is the time constant"), "what is the time constant")
    check("clean leaves ordinary text alone",
          clean("play bohemian rhapsody"), "play bohemian rhapsody")

    # A wake phrase must survive the guard, or Bruno can never be woken.
    from bruno.wake import wakes
    check("the wake phrase survives the guard",
          wakes(clean("hey bruno what's the time constant")), True)

    print(f"\n{'all hearing checks passed' if not fails else str(len(fails)) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
