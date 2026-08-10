"""Portable TTS backend: Kokoro over ONNX Runtime. Same model file on macOS and Windows.

Synthesis starts on the first clause, not the first full response. Waiting for the
LLM to finish would put the whole generation inside the latency budget.
"""

from __future__ import annotations

import re
import threading
from typing import Iterator

import numpy as np

from bruno import config

# Flush a chunk at a clause boundary, or once it gets long enough that waiting
# for one costs more than synthesizing early.
_BOUNDARY = re.compile(r"[.!?;:,]\s|\n")
_MIN_CHARS = 12
_MAX_CHARS = 120


def chunk_text(deltas: Iterator[str]) -> Iterator[str]:
    buf = ""
    for d in deltas:
        buf += d
        while True:
            m = None
            for candidate in _BOUNDARY.finditer(buf):
                if candidate.end() >= _MIN_CHARS:
                    m = candidate
                    break
            if m:
                head, buf = buf[: m.end()].strip(), buf[m.end():]
                if head:
                    yield head
                continue
            if len(buf) >= _MAX_CHARS:
                cut = buf.rfind(" ", 0, _MAX_CHARS)
                cut = cut if cut > _MIN_CHARS else _MAX_CHARS
                head, buf = buf[:cut].strip(), buf[cut:]
                if head:
                    yield head
                continue
            break
    if buf.strip():
        yield buf.strip()


class KokoroTTS:
    name = "kokoro-onnx"
    sample_rate = 24_000

    def __init__(self, model=None, voices=None, voice=None, speed=1.0):
        self.model_path = model or config.TTS_MODEL
        self.voices_path = voices or config.TTS_VOICES
        self.voice = voice or config.TTS_VOICE
        self.speed = speed
        self._k = None
        self._cancel = threading.Event()

    def _load(self):
        if self._k is None:
            from kokoro_onnx import Kokoro

            for p in (self.model_path, self.voices_path):
                if not p.exists():
                    raise FileNotFoundError(f"missing Kokoro asset: {p}")
            self._k = Kokoro(str(self.model_path), str(self.voices_path))
        return self._k

    def synth(self, text: str) -> np.ndarray:
        samples, sr = self._load().create(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        )
        assert sr == self.sample_rate, f"unexpected Kokoro rate {sr}"
        return np.asarray(samples, dtype=np.float32)

    def stream(self, text_chunks: Iterator[str]) -> Iterator[np.ndarray]:
        self._cancel.clear()
        for chunk in text_chunks:
            if self._cancel.is_set():
                return
            audio = self.synth(chunk)
            if self._cancel.is_set():
                return
            yield audio

    def cancel(self) -> None:
        self._cancel.set()

    def warmup(self) -> None:
        self.synth("Ready.")
