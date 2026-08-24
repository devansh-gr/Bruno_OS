#!/usr/bin/env python3
"""Speaker verification, measured with Piper voices standing in for people.

Synthetic voices are NOT real speakers. Two people with the same accent and register
are harder to separate than two TTS models, and neither has room noise or a mic two
metres away. So this measures the pipeline and the enrolment design -- not the
threshold that will actually apply to Devansh. That number comes from enrolment.

What it does establish is the thing worth establishing before asking him to record
anything: that averaging several enrolment clips is load-bearing. Single-clip
matching overlaps (margin -0.058); a six-clip centroid separates cleanly (+0.218).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ENROL = ["hey bruno what is the time constant of a ten k resistor",
         "pull up the news and tell me what happened overnight",
         "play something by kendrick and turn it up a bit",
         "draft an email to professor sarma about the readout",
         "how is the bruno os project coming along today",
         "what did claude get done while I was asleep"]
HELD = ["explain what a kalman filter actually does",
        "check nvidia and tell me if anything moved"]


def _clips(vp, sents):
    from piper import PiperVoice
    v = PiperVoice.load(str(vp))
    out = []
    for s in sents:
        pcm = b"".join(c.audio_int16_bytes for c in v.synthesize(s))
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        n = int(len(a) * 16000 / v.config.sample_rate)
        out.append(np.interp(np.linspace(0, len(a) - 1, n),
                             np.arange(len(a)), a).astype(np.float32))
    return out


def main() -> int:
    from bruno import speaker

    voices = sorted(Path("models/piper").glob("*.onnx"))[:6]
    if len(voices) < 3:
        print("  need at least 3 piper voices in models/piper to run this")
        return 0

    emb = speaker.Embedder()
    prints, held = {}, {}
    for vp in voices:
        name = vp.stem
        e = [emb.embed(c) for c in _clips(vp, ENROL)]
        m = np.mean(e, axis=0)
        prints[name] = m / np.linalg.norm(m)
        held[name] = [emb.embed(c) for c in _clips(vp, HELD)]

    genuine = [speaker.similarity(h, prints[n]) for n, hs in held.items() for h in hs]
    impostor = [speaker.similarity(h, prints[o]) for n, hs in held.items()
                for h in hs for o in prints if o != n]

    margin = min(genuine) - max(impostor)
    fr = sum(g < speaker.THRESHOLD for g in genuine)
    fa = sum(i >= speaker.THRESHOLD for i in impostor)

    print(f"  genuine  min {min(genuine):.3f}  mean {np.mean(genuine):.3f}  (n={len(genuine)})")
    print(f"  impostor max {max(impostor):.3f}  mean {np.mean(impostor):.3f}  (n={len(impostor)})")
    print(f"  margin   {margin:+.3f}")

    fails = []
    for name, ok in [
        ("held-out clips separate from impostors", margin > 0.05),
        ("no false accepts at the shipped threshold", fa == 0),
        ("no false rejects at the shipped threshold", fr == 0),
        ("threshold sits inside the gap",
         max(impostor) < speaker.THRESHOLD < min(genuine)),
    ]:
        if not ok:
            fails.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    print(f"\n{'all speaker checks passed' if not fails else str(len(fails)) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
