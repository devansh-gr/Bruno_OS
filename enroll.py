#!/usr/bin/env python3
"""Record Devansh's voiceprint. One-time, about 30 seconds.

    python enroll.py

Six short prompts, read aloud. Six rather than one because a single clip bakes in
that sentence's phonemes and that moment's mic position; verification then fails on
a different sentence spoken from a different angle. Measured on synthetic voices,
a six-clip centroid separates cleanly (+0.194 margin) where single-clip matching
overlaps (-0.058).

The voiceprint is a 192-number vector written to voiceprint.json. It is gitignored,
it never leaves the machine, and it cannot be turned back into audio.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from bruno import audio, config, speaker
from bruno.vad import VAD, FRAME, segments

# Deliberately the kinds of sentence he will actually say, not phonetic pangrams.
# The voiceprint should be built from his conversational register.
PROMPTS = [
    "Hey Bruno, what's the time constant of a ten k resistor?",
    "Pull up the news and tell me what happened overnight.",
    "Play something by Kendrick and turn it up a bit.",
    "Draft an email to Professor Sarma about the readout.",
    "How's the Bruno OS project coming along today?",
    "Hey Bruno, what did Claude get done while I was asleep?",
]


def trim(clip: np.ndarray) -> np.ndarray:
    """Keep only the speech. Leading and trailing silence drags the embedding
    toward the room rather than the voice."""
    v = VAD()
    segs = segments(v.probs(clip))
    if not segs:
        return clip
    lo, hi = segs[0][0] * FRAME, segs[-1][1] * FRAME
    return clip[max(0, lo - FRAME):min(len(clip), hi + FRAME)]


def main() -> int:
    problem = audio.check_mic()
    if problem:
        print(f"\n  {problem}\n")
        return 1

    print(__doc__.split("\n\n", 1)[1].strip())
    print(f"\n  {len(PROMPTS)} sentences. ENTER to start each one, ENTER again when done.\n")

    rec = audio.Recorder()
    rec.open()
    clips = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"  [{i}/{len(PROMPTS)}]  {prompt}")
        try:
            input("           ENTER, then read it aloud ")
            rec.start()
            input("           recording... ENTER when done ")
            clip = trim(rec.stop())
        except (KeyboardInterrupt, EOFError):
            print("\n  cancelled, nothing written")
            rec.close()
            return 1
        secs = len(clip) / config.SAMPLE_RATE
        if secs < speaker.MIN_SECONDS:
            print(f"           too short ({secs:.1f}s of speech), skipping\n")
            continue
        clips.append(clip)
        print(f"           got {secs:.1f}s of speech\n")

    rec.close()
    if len(clips) < 3:
        print("  need at least three usable clips; nothing written")
        return 1

    rc = speaker.enroll(clips)
    print(f"  enrolled from {rc['clips']} clips -> {speaker.PRINT_PATH.name}")
    print(f"  self-similarity: min {rc['self_similarity_min']:.3f}  "
          f"mean {rc['self_similarity_mean']:.3f}")
    if rc["self_similarity_min"] < 0.55:
        # The clips disagree with each other, so no threshold separates him from
        # anyone else. Usually one clip caught a cough, a laugh, or the mic bumped.
        print("\n  WARNING: those clips disagree with each other more than expected.")
        print("  Re-run somewhere quieter, at a steady distance from the mic.")
    else:
        print(f"\n  done. threshold is {speaker.THRESHOLD}; anything above that is you.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
