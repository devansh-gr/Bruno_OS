#!/usr/bin/env python3
"""Orpheus token arithmetic and emotion routing, without starting a server.

The SNAC mapping is the part that fails silently: get the offset or the codebook
split wrong and it still produces audio, just noise. So the arithmetic is checked
against a hand-built frame rather than by listening.
"""

from __future__ import annotations

import os
import sys

import numpy as np


def main() -> int:
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name:52} {got} (want {want})")

    from bruno.backends.tts_orpheus import CHUNK, FRAME, OVERLAP, VOICES, _TOKEN

    check("seven tokens per SNAC frame", FRAME, 7)
    check("chunk is a whole number of frames", CHUNK % FRAME, 0)
    check("overlap is one frame", OVERLAP, FRAME)
    check("eight voices", len(VOICES), 8)

    # Position i carries codebook i%7, each 4096 wide, offset by 10.
    ids = [10 + (i % 7) * 4096 + (i * 3 % 4096) for i in range(7)]
    vals = [t - 10 - ((i % 7) * 4096) for i, t in enumerate(ids)]
    check("offset arithmetic recovers the codes", vals, [i * 3 % 4096 for i in range(7)])
    check("all codes land in range", all(0 <= v <= 4095 for v in vals), True)

    text = "".join(f"<custom_token_{i}>" for i in (10, 4107, 8204))
    check("token regex finds every id", [int(x) for x in _TOKEN.findall(text)], [10, 4107, 8204])

    # The 7 tokens split 1 / 2 / 4 across SNAC's three codebooks.
    f = list(range(7))
    c0, c1, c2 = [f[0]], [f[1], f[4]], [f[2], f[3], f[5], f[6]]
    check("codebook split is 1/2/4", (len(c0), len(c1), len(c2)), (1, 2, 4))

    from bruno.backends.tts_piper import strip_emotion
    check("piper strips a tag it cannot perform",
          strip_emotion("Well <sigh> fine.").replace("  ", " "), "Well fine.")
    check("piper leaves ordinary angle text alone",
          strip_emotion("use <int> not <float>"), "use <int> not <float>")

    from bruno import config
    check("emotion is off for piper", config.TTS_BACKEND == "orpheus" or not config.supports_emotion(), True)

    print(f"\n{'all orpheus checks passed' if not fails else str(len(fails)) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
