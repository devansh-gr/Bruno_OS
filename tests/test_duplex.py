#!/usr/bin/env python3
"""Duplex-mode selection and the barge-in trigger, without a microphone."""

from __future__ import annotations

import sys

from bruno import audio


def main() -> int:
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name:52} {got} (want {want})")

    hints = [
        ("Devansh's AirPods", True),
        ("AirPods Pro", True),
        ("Beats Studio Buds", True),
        ("Sony WH-1000XM5 Headphones", True),
        ("USB Headset", True),
        ("MacBook Pro Speakers", False),
        ("HP Z27", False),
        ("Microsoft Teams Audio", False),
        ("External Monitor", False),
    ]
    for name, want in hints:
        got = any(h in name.lower() for h in audio._HEADPHONE_HINTS)
        check(f"{name!r} reads as worn" if want else f"{name!r} reads as room", got, want)

    # The live query must never raise, whatever the device situation is.
    try:
        audio.output_is_headphones()
        audio.output_name()
        ok = True
    except Exception:
        ok = False
    check("live device query never raises", ok, True)

    from bruno.vad import FRAME_MS
    from bruno.voice import Voice
    frames = int(Voice.BARGE_IN_MS / FRAME_MS)
    # A click is one or two frames. The threshold has to sit well above that and
    # still well below a spoken word.
    check("barge-in needs more than a click", frames > 4, True)
    check("barge-in fires inside half a second", Voice.BARGE_IN_MS < 500, True)

    print(f"\n{'all duplex checks passed' if not fails else str(len(fails)) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
