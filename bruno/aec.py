"""Acoustic echo cancellation, so Bruno can be interrupted.

Measured on the built-in speakers: Bruno's own voice returns to the mic at peak
0.435 against a room floor of 0.019, and Silero scores it 1.000 on 137 of 227
frames. Left listening while he talks, he answers himself.

The approach is subtraction with an adaptive filter, not the OS voice-processing
unit. Apple's VPIO wants the input and output to be the same device, which is
exactly what they are not here -- built-in mic with AirPods or an external monitor
for output is the normal case, and VPIO fails on mismatched pairs.

Subtraction is tractable here for a reason that does not apply to a phone call:
Bruno *generates* the far-end signal, so the reference is known exactly rather than
estimated. The only unknown is how many samples late it reaches the mic, and that
is measurable once per device pair with a chirp.

Two pieces:
  delay  -- cross-correlate a chirp against its own recording
  NLMS   -- normalised least-mean-squares adaptive filter, frozen during double-talk
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from bruno import config

SR = config.SAMPLE_RATE


def chirp(seconds: float = 0.25, f0: float = 300.0, f1: float = 6000.0,
          sr: int = SR) -> np.ndarray:
    """Linear sweep, windowed. A sweep rather than a click because its
    autocorrelation is a sharp single peak, which is what makes the delay estimate
    unambiguous; a click smears across the speaker's impulse response."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t ** 2))
    return (x * np.hanning(len(x))).astype(np.float32) * 0.5


def estimate_delay(reference: np.ndarray, captured: np.ndarray,
                   max_ms: float = 500.0, sr: int = SR) -> tuple[int, float]:
    """Samples of delay between a played signal and its recording, plus a
    confidence in [0,1].

    Confidence is the peak of the normalised cross-correlation. Below ~0.2 the
    estimate is noise -- the speaker was muted, the mic was blocked, or the room
    swallowed the sweep -- and the caller should refuse to enable cancellation
    rather than subtract at a wrong offset, which makes things worse, not better.
    """
    n = min(len(reference), len(captured))
    if n < sr // 20:
        return 0, 0.0
    ref = reference[:n] - reference[:n].mean()
    cap = captured[:n] - captured[:n].mean()
    corr = np.correlate(cap, ref, mode="full")
    mid = len(ref) - 1
    hi = min(len(corr) - 1, mid + int(max_ms * sr / 1000))
    window = corr[mid:hi + 1]
    if not len(window):
        return 0, 0.0
    k = int(np.argmax(np.abs(window)))
    denom = np.linalg.norm(ref) * np.linalg.norm(cap) + 1e-12
    return k, float(np.abs(window[k]) / denom)


@dataclass
class Calibration:
    delay_samples: int
    confidence: float
    measured_at: float = 0.0

    @property
    def delay_ms(self) -> float:
        return self.delay_samples * 1000 / SR

    @property
    def usable(self) -> bool:
        # Below this the correlation peak is not distinguishable from the room.
        return self.confidence >= 0.20


class NLMS:
    """Normalised least-mean-squares echo canceller.

    Estimates the room's impulse response from speaker to mic and subtracts the
    predicted echo. Normalised because the step size has to be independent of how
    loud Bruno is talking, or the filter diverges the moment the volume changes.

    `taps` sets how much room reverberation is modelled. 1024 taps at 16 kHz is
    64 ms, which covers a desk and a laptop; a hard-walled room needs more, at
    quadratic cost in nothing but a longer convolution per frame.
    """

    def __init__(self, taps: int = 512, mu: float = 0.30, delay: int = 0,
                 dt_margin: float = 1.2):
        self.taps = taps
        self.mu = mu
        self.delay = delay
        # How far above the learned echo-path gain counts as his voice. Swept on a
        # synthetic room: 1.2 preserves his speech at r=0.969 against 0.923 for
        # anything >=2.0, and costs 0.5 dB of ERLE. Above 1.5 the detector stops
        # firing at all and the freeze is decorative.
        self.dt_margin = dt_margin
        self._gain: float | None = None
        self.w = np.zeros(taps, dtype=np.float32)
        self._ref = np.zeros(taps, dtype=np.float32)   # newest-first reference window
        # Priming the queue with `delay` zeros is what actually applies the measured
        # acoustic latency: mic sample i then lines up with reference sample i-delay.
        # Without it the filter has to spend its taps learning the delay as part of
        # the impulse response -- with a 400-sample delay and a 300-tap room that
        # needs 700 taps, and 512 taps plateaued at 15 dB while 1024 reached 55 dB.
        # The delay was being measured accurately and then ignored.
        self._pending = np.zeros(delay, dtype=np.float32)
        self.last_doubletalk = False

    def reset(self) -> None:
        self.w[:] = 0
        self._ref[:] = 0
        self._gain = None
        self._pending = np.zeros(self.delay, dtype=np.float32)

    def feed_reference(self, x: np.ndarray) -> None:
        """Hand over samples that were just sent to the speaker."""
        self._pending = np.concatenate([self._pending, np.asarray(x, dtype=np.float32)])

    def doubletalk(self, mic: np.ndarray, ref_block: np.ndarray) -> bool:
        """Is Devansh talking over Bruno right now?

        Geigel compares the mic against the reference and calls double-talk when
        the mic is the louder of the two. A fixed threshold does not survive
        contact with a real room: the echo path here has a gain around 0.6, so
        |mic| exceeds 0.5*|ref| on pure echo and the detector fires permanently --
        measured, it froze adaptation before the filter ever converged and ERLE
        collapsed from 24.9 dB to 2.5 dB.

        So the echo-path gain is learned instead of assumed. The ratio of mic to
        reference is tracked while single-talk holds, and double-talk is declared
        when the current ratio jumps well above it. The estimate is only updated
        when double-talk is NOT declared, so his voice cannot inflate the very
        baseline used to detect his voice.

        This is the guard on the failure he was warned about: adapting while he
        speaks makes the filter model his voice as echo and subtract it, so the
        harder he talks the more thoroughly he is erased.
        """
        if not len(mic) or not len(ref_block):
            return False
        r = float(np.max(np.abs(ref_block)))
        m = float(np.max(np.abs(mic)))
        if r < 1e-4:
            # Nothing is playing, so nothing can be echo. Any speech is his, and
            # there is no adaptation to protect.
            return m > 1e-3
        ratio = m / r
        if self._gain is None:
            self._gain = ratio
            return False
        if ratio > self._gain * self.dt_margin:
            return True
        # Slow tracking: fast enough to follow a volume change, slow enough that a
        # single loud block does not redefine the baseline.
        self._gain += 0.05 * (ratio - self._gain)
        return False

    def process(self, mic: np.ndarray, adapt: bool | None = None) -> np.ndarray:
        """Subtract the predicted echo from a block of microphone audio.

        `adapt=None` detects double-talk and freezes adaptation itself, which is
        what callers should use. Pass an explicit bool only to measure the two
        behaviours against each other.

        Freezing rather than stopping: the filter keeps subtracting with the
        coefficients it already has, so the echo stays cancelled while he talks. It
        just stops *learning* from a block that is no longer mostly echo.
        """
        mic = np.asarray(mic, dtype=np.float32)
        out = np.empty_like(mic)
        # The queue was primed with `delay` zeros at construction, so consuming it
        # in step with the mic keeps the two aligned. Short-fill with silence when
        # playback has ended but the room is still ringing.
        if len(self._pending) < len(mic):
            self._pending = np.concatenate(
                [self._pending,
                 np.zeros(len(mic) - len(self._pending), dtype=np.float32)])
        ref_block = self._pending[:len(mic)]
        self._pending = self._pending[len(mic):]
        if adapt is None:
            adapt = not self.doubletalk(mic, ref_block)
        self.last_doubletalk = not adapt

        for i, d in enumerate(mic):
            self._ref[1:] = self._ref[:-1]
            self._ref[0] = ref_block[i]
            y = float(np.dot(self.w, self._ref))     # predicted echo
            e = d - y                                 # what is left: near-end speech
            out[i] = e
            if adapt:
                p = float(np.dot(self._ref, self._ref)) + 1e-6
                self.w += (self.mu * e / p) * self._ref
        return out


def erle_db(mic: np.ndarray, cancelled: np.ndarray) -> float:
    """Echo Return Loss Enhancement, the standard AEC metric: how much quieter the
    echo got, in dB. 0 means the filter did nothing; 20 dB is a tenfold reduction
    in amplitude and is generally considered usable for barge-in."""
    a = float(np.mean(np.square(mic))) + 1e-12
    b = float(np.mean(np.square(cancelled))) + 1e-12
    return 10.0 * np.log10(a / b)
