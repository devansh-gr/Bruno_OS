"""Echo cancellation, so Bruno can be interrupted on laptop speakers.

Measured there: his own voice returns at peak 0.435 against a 0.019 room floor and
Silero scores it 1.000. Through AirPods it is 0.06, so this mainly matters on
speakers.

Subtraction rather than Apple's VPIO, which needs input and output to be the same
device. Tractable because Bruno generates the far-end signal, so the reference is
exact and only the acoustic delay is unknown.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from bruno import config

SR = config.SAMPLE_RATE


def chirp(seconds: float = 0.25, f0: float = 300.0, f1: float = 6000.0,
          sr: int = SR) -> np.ndarray:
    """Windowed linear sweep. Sharp autocorrelation peak, unlike a click."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t ** 2))
    return (x * np.hanning(len(x))).astype(np.float32) * 0.5


def estimate_delay(reference: np.ndarray, captured: np.ndarray,
                   max_ms: float = 500.0, sr: int = SR) -> tuple[int, float]:
    """Delay in samples, plus normalised-correlation confidence. Below ~0.2 it is
    noise: subtracting at a wrong offset makes things worse, not better."""
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
    """NLMS echo canceller. Normalised so the step size does not depend on playback
    volume. Taps sets modelled reverb: 512 at 16 kHz is 32 ms."""

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
        """Is he talking over Bruno? Guards the failure where adapting during his
        speech makes the filter model his voice as echo and subtract it.

        Geigel with a learned rather than fixed threshold: the echo path gain is
        ~0.6, so a fixed 0.5 fired permanently and ERLE fell from 24.9 to 2.5 dB.
        The gain is only updated when double-talk is NOT declared, so his voice
        cannot inflate the baseline used to detect his voice.
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

        adapt=None self-detects double-talk and freezes learning while still
        subtracting with the coefficients it already has.
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
    """ERLE in dB, the standard AEC metric. 20 dB is the usual bar for barge-in."""
    a = float(np.mean(np.square(mic))) + 1e-12
    b = float(np.mean(np.square(cancelled))) + 1e-12
    return 10.0 * np.log10(a / b)
