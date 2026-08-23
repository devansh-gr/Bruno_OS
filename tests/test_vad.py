#!/usr/bin/env python3
"""Endpointing checks, run against synthesized speech rather than a microphone.

Synthesized speech is not a substitute for a real voice -- no room tone, no
disfluency, no plosives on the mic. But it has one property a recording does not:
the boundaries are known exactly, so a regression in the thresholds shows up as a
number rather than as "it felt worse".
"""

from __future__ import annotations

import sys

import numpy as np

from bruno import config
from bruno.vad import FRAME, FRAME_MS, VAD, Endpoint, segments

SR = config.SAMPLE_RATE
_tts = None


def synth(text: str) -> np.ndarray:
    global _tts
    if _tts is None:
        from bruno.backends.tts_piper import PiperTTS
        _tts = PiperTTS()
        _tts.warmup()
    a = _tts.synth(text)
    n = int(len(a) * SR / _tts.sample_rate)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def sil(ms: float) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.float32)


def main() -> int:
    v = VAD()
    fails = []

    def check(name, got, want):
        ok = got == want
        fails.append(name) if not ok else None
        print(f"  {'PASS' if ok else 'FAIL'}  {name:44} got {got}, want {want}")

    speech = synth("hey bruno what is the time constant of a ten k resistor")
    one = np.concatenate([sil(800), speech, sil(800)])
    segs = segments(v.probs(one))
    check("one utterance is one turn", len(segs), 1)

    if segs:
        # Onset lands a frame or two late, which the recorder's 300ms pre-roll covers.
        # Anything worse than a quarter second would clip the first word.
        start_err = abs(segs[0][0] * FRAME_MS - 800)
        ok = start_err < 250
        fails.append("onset accuracy") if not ok else None
        print(f"  {'PASS' if ok else 'FAIL'}  {'onset within 250ms of truth':44} off by {start_err:.0f}ms")

    a, b = synth("play bohemian rhapsody"), synth("actually skip that")
    check("1.2s gap splits into two turns",
          len(segments(v.probs(np.concatenate([sil(500), a, sil(1200), b, sil(500)])))), 2)
    # The one that matters for feeling natural: thinking mid-sentence is not an endpoint.
    check("0.3s pause stays one turn",
          len(segments(v.probs(np.concatenate([sil(500), a, sil(300), b, sil(500)])))), 1)
    check("silence produces no turns", len(segments(v.probs(sil(4000)))), 0)

    # A click or a chair scrape is not a turn.
    click = np.zeros(SR, dtype=np.float32)
    click[SR // 2:SR // 2 + 300] = np.random.RandomState(0).randn(300).astype(np.float32)
    check("a click is not a turn", len(segments(v.probs(click))), 0)

    # Hysteresis: leaving speech must be harder than entering it, or a natural
    # amplitude dip inside a word ends the turn.
    ep = Endpoint()
    ok = ep.end_prob < ep.start_prob
    fails.append("hysteresis") if not ok else None
    print(f"  {'PASS' if ok else 'FAIL'}  {'end threshold below start threshold':44} "
          f"{ep.end_prob} < {ep.start_prob}")

    print(f"\n{'all VAD checks passed' if not fails else str(len(fails)) + ' failed: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
