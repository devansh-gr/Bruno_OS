"""Portable TTS backend: Piper (VITS over ONNX Runtime). Same model file on macOS and Windows.

Chosen on measured latency, not preference. Kokoro sounds better but runs at
RTF 0.4-1.8 on this M4 Pro, which spends the entire budget on TTS alone. Piper
runs at RTF ~0.02. See 07 Iteration Logs for the numbers.

Synthesis starts on the first clause, never on the full response: waiting for the
LLM to finish would put all of generation inside the latency budget.
"""

from __future__ import annotations

import re
import threading
from typing import Iterator

import numpy as np

from bruno import config

# Flush at a clause boundary, or once the buffer is long enough that waiting for
# one costs more than synthesizing early.
_BOUNDARY = re.compile(r"[.!?;:,]\s|\n")

# The first chunk is the one the 500ms gate actually measures, so it gets cut
# short at a word boundary rather than waiting for punctuation. Measured: waiting
# for the first comma cost up to 3.3s on answers that open with a long clause.
# Later chunks are allowed to be long, which reads better and costs nothing: 
# by then audio is already playing and synthesis is 40x faster than playback.
_FIRST_MIN, _FIRST_MAX = 8, 36
_MIN_CHARS, _MAX_CHARS = 12, 120


_EMO = re.compile(r"<(?:laugh|chuckle|sigh|gasp|groan|yawn|cough|sniffle)>")


def strip_emotion(text: str) -> str:
    """Piper has no idea what <sigh> is and would spell it out."""
    return _EMO.sub("", text)


def expand_monosyllables(chunk: str) -> str:
    """Replace a chunk that is nothing but a short confirmation.

    A synthesized "Go?" is unintelligible through a speaker. There is no
    surrounding audio for the ear to disambiguate one syllable against. The system
    prompt forbids these, but a 3B drifts, so the rule is enforced here too.
    """
    key = chunk.strip().lower()
    return config.BANNED_UTTERANCES.get(key, chunk)


def chunk_text(deltas: Iterator[str]) -> Iterator[str]:
    buf = ""
    first = True

    def bounds():
        return (_FIRST_MIN, _FIRST_MAX) if first else (_MIN_CHARS, _MAX_CHARS)

    for d in deltas:
        buf += d
        while True:
            lo, hi = bounds()
            m = next((c for c in _BOUNDARY.finditer(buf) if c.end() >= lo), None)
            if m and m.end() <= hi:
                head, buf = buf[: m.end()].strip(), buf[m.end():]
                if head:
                    yield head
                    first = False
                continue
            if len(buf) >= hi:
                cut = buf.rfind(" ", 0, hi)
                cut = cut if cut > lo else hi
                head, buf = buf[:cut].strip(), buf[cut:]
                if head:
                    yield head
                    first = False
                continue
            break
    if buf.strip():
        yield buf.strip()


class PiperTTS:
    name = "piper"

    def __init__(self, model=None, speed=1.0):
        self.model_path = model or config.PIPER_MODEL
        self.speed = speed
        self._v = None
        self._cancel = threading.Event()
        self.sample_rate = 22_050  # corrected from the model config on load

    def _load(self):
        if self._v is None:
            from piper import PiperVoice

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"missing Piper voice: {self.model_path}\n"
                    f"  python -m piper.download_voices {self.model_path.stem} "
                    f"--data-dir {self.model_path.parent}"
                )
            self._v = PiperVoice.load(str(self.model_path))
            self.sample_rate = self._v.config.sample_rate
        return self._v

    def synth(self, text: str) -> np.ndarray:
        text = strip_emotion(text)
        text = expand_monosyllables(text)
        pcm = [
            np.frombuffer(c.audio_int16_bytes, dtype=np.int16)
            for c in self._load().synthesize(text)
        ]
        if not pcm:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pcm).astype(np.float32) / 32768.0

    def stream(self, text_chunks: Iterator[str]) -> Iterator[np.ndarray]:
        self._cancel.clear()
        for chunk in text_chunks:
            if self._cancel.is_set():
                return
            audio = self.synth(chunk)
            if self._cancel.is_set():
                return
            if len(audio):
                yield audio

    def cancel(self) -> None:
        self._cancel.set()

    def warmup(self) -> None:
        self.synth("Ready.")
