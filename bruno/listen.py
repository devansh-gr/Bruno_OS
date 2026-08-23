"""Continuous listening: turns in, no key presses.

The M1 harness needs ENTER to start and ENTER to stop. This replaces both with the
VAD, which is the whole difference between a CLI that takes dictation and something
you talk to.

The loop is deliberately a generator rather than a callback. A turn is a unit of
work with a beginning and an end, and the caller already has a natural place to put
it -- `for clip in listener.turns()`. Callbacks would put the transcribe/route/answer
chain on the audio thread, where a slow answer means dropped frames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from bruno import config
from bruno.audio import Recorder
from bruno.vad import FRAME, FRAME_MS, VAD, Endpoint


@dataclass
class Turn:
    audio: np.ndarray
    seconds: float
    speech_ms: float
    # Wall-clock at which the VAD called the end. Everything downstream measures
    # from here, because this is the instant a person expects an answer to start.
    endpointed_at: float = field(default_factory=time.perf_counter)


class Listener:
    """Ring buffer -> VAD -> whole utterances.

    Holds no audio of its own; it reads through the recorder's ring with a cursor,
    so a slow consumer falls behind visibly instead of corrupting a private copy.
    """

    def __init__(self, rec: Recorder | None = None, ep: Endpoint | None = None,
                 vad: VAD | None = None):
        self.rec = rec or Recorder()
        self.ep = ep or Endpoint()
        self.vad = vad or VAD()
        self.cursor: int | None = None
        self._buf = np.zeros(0, dtype=np.float32)
        # Paused rather than stopped while Bruno speaks: the stream stays open
        # because reopening it costs the first half second of the next turn.
        self.paused = False
        # Cleared to end the loop. Without it `turns()` is unstoppable: the frame
        # generator sleeps when starved, so a caller blocked in next() stays blocked
        # forever once audio stops -- including on shutdown.
        self.running = True

    def open(self) -> None:
        self.rec.open()
        _, self.cursor = self.rec.since(None)  # start at the live edge, no backlog

    def close(self) -> None:
        self.rec.close()

    def stop(self) -> None:
        self.running = False

    def _frames(self):
        """Yield aligned 512-sample frames as they arrive, sleeping when starved."""
        while self.running:
            chunk, self.cursor = self.rec.since(self.cursor)
            if len(chunk):
                self._buf = np.concatenate([self._buf, chunk])
            n = len(self._buf) // FRAME
            if not n:
                time.sleep(FRAME_MS / 2000)  # half a frame; long enough not to spin
                continue
            for i in range(n):
                yield self._buf[i * FRAME:(i + 1) * FRAME]
            self._buf = self._buf[n * FRAME:]

    def turns(self):
        """Yield a Turn each time someone stops talking.

        Pre-roll is taken from the ring rather than from frames already seen, because
        the VAD needs a frame or two of speech before it fires and those frames hold
        the first phoneme. Measured onset lag is ~32ms; the pre-roll covers 300ms.
        """
        hang = int(self.ep.hang_ms / FRAME_MS)
        min_frames = int(self.ep.min_speech_ms / FRAME_MS)
        max_frames = int(self.ep.max_turn_ms / FRAME_MS)
        pre = int(self.rec.pre_roll * config.SAMPLE_RATE)

        start_at = None      # absolute sample index where speech began
        held: list[np.ndarray] = []
        silence = 0

        for frame in self._frames():
            if self.paused:
                # Keep the cursor moving and the RNN state clean, but decide nothing.
                # Bruno's own voice is in these frames on a laptop speaker.
                start_at, held, silence = None, [], 0
                self.vad.reset()
                continue

            p = self.vad.prob(frame)
            if start_at is None:
                if p >= self.ep.start_prob:
                    start_at = self.cursor - len(self._buf) - FRAME
                    held, silence = [frame], 0
                continue

            held.append(frame)
            silence = 0 if p >= self.ep.end_prob else silence + 1

            done = silence > hang or len(held) >= max_frames
            if not done:
                continue

            voiced = len(held) - silence
            if voiced >= min_frames:
                audio = self.rec.window(max(0, start_at - pre))
                yield Turn(audio=audio,
                           seconds=len(audio) / config.SAMPLE_RATE,
                           speech_ms=voiced * FRAME_MS)
            start_at, held, silence = None, [], 0
