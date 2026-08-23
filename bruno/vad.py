"""Voice activity detection and endpointing.

Endpointing is the moment the harness has been faking since M1: it starts its
clock when recording *stops*, because a real VAD would have decided the turn was
over there. This is that decision, made for real.

Silero v5 via onnxruntime, not the `silero-vad` package. The package depends on
torch unconditionally -- 501 MB on disk and 373 ms to import, against 2.3 MB and
9 ms for the ONNX file it ships. For a process whose whole point is answering
inside 500 ms, that is not a close call. The .onnx is vendored in models/.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from bruno import config

# Silero v5 is trained on exactly 512-sample frames at 16 kHz. Feeding it any
# other length runs, and returns quietly meaningless probabilities.
FRAME = 512

# The ONNX graph expects the previous frame's last 64 samples prepended, because
# the JIT wrapper that ships with the package does that before every call. Feed it
# a bare 512 and it runs happily and reports ~0.001 on clear speech -- no error, no
# warning, just a VAD that never fires. Cost me the first version of this file.
CONTEXT = 64
FRAME_MS = FRAME * 1000 / config.SAMPLE_RATE  # 32ms, matches the recorder's block


@dataclass
class Endpoint:
    """Thresholds in milliseconds, because that is how the behaviour is described.

    Speech and silence use different thresholds on purpose. Entering speech should
    be eager -- a clipped first syllable is unrecoverable. Leaving it should be
    patient, because a pause mid-sentence is not the end of a turn.
    """

    start_prob: float = 0.5
    end_prob: float = 0.35          # hysteresis: harder to leave speech than to enter
    min_speech_ms: float = 160.0    # below this it is a cough, a click, a door
    hang_ms: float = 700.0          # silence tolerated inside one turn
    max_turn_ms: float = 30_000.0   # a hard stop, so a stuck mic cannot hang the loop


class VAD:
    """Frame-by-frame speech probability, with the RNN state carried across calls.

    The state is the whole point -- Silero is recurrent, so scoring frames
    independently throws away the context that makes it better than an energy gate.
    """

    def __init__(self, model_path=None):
        self.path = str(model_path or config.MODELS / "silero_vad.onnx")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1  # one frame at a time; threads only add latency
        self.sess = ort.InferenceSession(self.path, opts,
                                         providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(CONTEXT, dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample frame."""
        if len(frame) != FRAME:
            frame = (np.pad(frame, (0, FRAME - len(frame))) if len(frame) < FRAME
                     else frame[:FRAME])
        frame = frame.astype(np.float32)
        x = np.concatenate([self._ctx, frame]).reshape(1, -1)
        out, self._state = self.sess.run(
            None,
            {"input": x,
             "state": self._state,
             "sr": np.array(config.SAMPLE_RATE, dtype=np.int64)},
        )
        self._ctx = frame[-CONTEXT:]
        return float(out[0][0])

    def probs(self, audio: np.ndarray) -> np.ndarray:
        """Probability per frame over a whole clip. Offline use — tests, tuning."""
        self.reset()
        n = len(audio) // FRAME
        return np.array([self.prob(audio[i * FRAME:(i + 1) * FRAME]) for i in range(n)])


def segments(probs: np.ndarray, ep: Endpoint = Endpoint()) -> list[tuple[int, int]]:
    """Frame spans of speech, as (start, end) inclusive-exclusive.

    Written against a probability array rather than live audio so the thresholds can
    be tuned and regression-tested without a microphone in the loop.
    """
    hang_frames = int(ep.hang_ms / FRAME_MS)
    min_frames = int(ep.min_speech_ms / FRAME_MS)
    out: list[tuple[int, int]] = []
    start = None
    silence = 0
    for i, p in enumerate(probs):
        if start is None:
            if p >= ep.start_prob:
                start, silence = i, 0
        elif p >= ep.end_prob:
            silence = 0
        else:
            silence += 1
            if silence > hang_frames:
                end = i - silence + 1
                if end - start >= min_frames:
                    out.append((start, end))
                start = None
    if start is not None and len(probs) - start >= min_frames:
        out.append((start, len(probs)))
    return out
