#!/usr/bin/env python3
"""Listener checks, driven by a fake recorder instead of a microphone.

The fake exposes the same three methods the Listener actually uses -- since,
window, pre_roll -- and hands out a prepared clip in real-time-sized pieces. That
keeps the test honest about the streaming behaviour (partial frames, starvation)
while remaining deterministic and mic-free.
"""

from __future__ import annotations

import sys
import threading

import numpy as np

from bruno import config
from bruno.listen import Listener
from bruno.vad import Endpoint

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


class FakeRecorder:
    """Streams a fixed clip in 512-sample steps. Not real-time -- as fast as the
    consumer asks -- because the point is the state machine, not the clock."""

    def __init__(self, clip: np.ndarray, pre_roll: float = 0.3):
        self.clip = clip
        self.pre_roll = pre_roll
        self._w = 0
        self.exhausted = threading.Event()

    def open(self): pass
    def close(self): pass

    def since(self, cursor):
        if cursor is None:
            return np.zeros(0, dtype=np.float32), 0
        step = 512
        lo, hi = cursor, min(cursor + step, len(self.clip))
        if lo >= len(self.clip):
            self.exhausted.set()
            return np.zeros(0, dtype=np.float32), lo
        self._w = hi
        return self.clip[lo:hi], hi

    def window(self, lo):
        return self.clip[max(0, lo):self._w]


def drain(rec, lis, limit=6):
    """Run the listener to completion, stopping it once the clip is exhausted.

    A watchdog rather than a loop condition, because next() blocks: the frame
    generator sleeps when starved, so nothing outside it can notice the end.
    """
    def watchdog():
        rec.exhausted.wait(timeout=20)
        lis.stop()
    threading.Thread(target=watchdog, daemon=True).start()
    out = []
    for turn in lis.turns():
        out.append(turn)
        if len(out) >= limit:
            lis.stop()
            break
    return out


def collect(clip, ep=None, limit=6):
    rec = FakeRecorder(clip)
    lis = Listener(rec=rec, ep=ep or Endpoint())
    lis.open()
    return drain(rec, lis, limit)


def main() -> int:
    fails = []

    def check(name, got, want):
        ok = got == want
        if not ok:
            fails.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:46} got {got}, want {want}")

    a = synth("hey bruno what is the time constant of a ten k resistor")
    b = synth("actually never mind")

    turns = collect(np.concatenate([sil(600), a, sil(1500), sil(400)]))
    check("one utterance yields one turn", len(turns), 1)
    if turns:
        # The clip must contain the speech plus pre-roll, never start mid-word.
        ok = turns[0].seconds >= len(a) / SR
        if not ok:
            fails.append("turn contains whole utterance")
        print(f"  {'PASS' if ok else 'FAIL'}  {'turn holds the whole utterance':46} "
              f"{turns[0].seconds:.2f}s >= {len(a)/SR:.2f}s")

    turns = collect(np.concatenate([sil(500), a, sil(1500), b, sil(1500), sil(400)]))
    check("two utterances yield two turns", len(turns), 2)

    check("a 300ms pause does not split a turn",
          len(collect(np.concatenate([sil(500), a, sil(300), b, sil(1500), sil(400)]))), 1)
    check("silence yields nothing", len(collect(sil(4000))), 0)

    # Pausing is what runs while Bruno is speaking. It must yield nothing at all,
    # or he answers his own voice coming back through the laptop speaker.
    rec = FakeRecorder(np.concatenate([sil(300), a, sil(1500), sil(400)]))
    lis = Listener(rec=rec)
    lis.open()
    lis.paused = True
    check("paused listener yields nothing", len(drain(rec, lis)), 0)

    print(f"\n{'all listener checks passed' if not fails else str(len(fails)) + ' failed: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
