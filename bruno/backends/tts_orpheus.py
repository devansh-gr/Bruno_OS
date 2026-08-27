"""Orpheus TTS: a Llama-3B emitting SNAC audio codes, served by llama.cpp.

Measured on the M4 Pro: 382 ms to first audio against Chatterbox's 1801 ms, at
0.92x realtime so speech never runs dry mid-sentence. Emotion is trained inline
tags (<laugh>, <sigh>, <chuckle>, <gasp>, <groan>, <yawn>, <cough>, <sniffle>),
so the model places feeling itself rather than a dial applying to a whole utterance.

Reuses the llama.cpp runtime Bruno already runs, on its own port.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from typing import Iterator

import numpy as np
import requests

from bruno import config

VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")
FRAME = 7           # tokens per SNAC frame
CHUNK = 28          # 4 frames, the smallest that decodes cleanly
OVERLAP = 7         # one frame of context, trimmed after decoding, to hide joins
_TOKEN = re.compile(r"<custom_token_(\d+)>")
_ANY_TAG = re.compile(r"<[a-z_]{2,16}>", re.I)


def sanitize(text: str) -> str:
    """Drop any angle-bracket tag Orpheus was not trained on.

    An unknown tag is not ignored by the model, it is spoken as its letters. A 3B
    that has been told it may write <sigh> will eventually write <thinking> or
    <pause>, and the answer then contains the word said out loud.
    """
    return _ANY_TAG.sub(
        lambda m: m.group(0) if m.group(0).lower() in config.ORPHEUS_TAGS else "", text)


class TTSUnavailable(RuntimeError):
    """The voice failed. The turn still happened; say it some other way."""


class OrpheusTTS:
    name = "orpheus"

    def __init__(self, voice: str | None = None, port: int = 8099):
        self.voice = voice or config.ORPHEUS_VOICE
        if self.voice not in VOICES:
            # An unknown name is not rejected by the model. It is read as the first
            # word of the sentence, so Bruno introduces himself as "jarvis:" and the
            # voice is whatever the model defaults to. Silent and baffling.
            raise ValueError(
                f"unknown Orpheus voice {self.voice!r}; pick one of {', '.join(VOICES)}")
        self.sample_rate = 24_000
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self._proc: subprocess.Popen | None = None
        self._snac = None
        self._cancel = threading.Event()

    def _server(self) -> None:
        try:
            if requests.get(f"{self.base}/health", timeout=1).ok:
                return
        except requests.RequestException:
            pass
        log = open(config.LOGS / "orpheus-server.log", "ab")
        self._proc = subprocess.Popen(
            ["llama-server", "-m", str(config.ORPHEUS_GGUF), "--port", str(self.port),
             "--ctx-size", "4096", "-ngl", "99"],
            stdout=log, stderr=log)
        for _ in range(120):
            try:
                if requests.get(f"{self.base}/health", timeout=1).ok:
                    return
            except requests.RequestException:
                time.sleep(0.5)
        raise TimeoutError("orpheus llama-server did not come up")

    def _decoder(self):
        if self._snac is None:
            import torch
            from snac import SNAC
            self._snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            with torch.no_grad():   # warm the graph off the latency path
                self._snac.decode([torch.zeros(1, 4, dtype=torch.long),
                                   torch.zeros(1, 8, dtype=torch.long),
                                   torch.zeros(1, 16, dtype=torch.long)])
        return self._snac

    def _decode(self, vals: list[int]) -> np.ndarray:
        import torch
        n = (len(vals) // FRAME) * FRAME
        c0, c1, c2 = [], [], []
        for j in range(n // FRAME):
            f = vals[FRAME * j:FRAME * j + FRAME]
            c0.append(f[0]); c1 += [f[1], f[4]]; c2 += [f[2], f[3], f[5], f[6]]
        if not c0:
            return np.zeros(0, dtype=np.float32)
        codes = [torch.tensor(c, dtype=torch.long).unsqueeze(0) for c in (c0, c1, c2)]
        with torch.no_grad():
            return self._decoder().decode(codes).squeeze().cpu().numpy().astype(np.float32)

    def synth(self, text: str) -> np.ndarray:
        return np.concatenate(list(self._speak(text)) or [np.zeros(0, dtype=np.float32)])

    def _speak(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio while the model is still generating. Chunks carry one frame of
        overlap and drop its samples after decoding, so joins have SNAC context."""
        self._server()
        self._decoder()
        text = sanitize(text)
        prompt = (f"<custom_token_3><|begin_of_text|>{self.voice}: {text}"
                  f"<|eot_id|><custom_token_4><custom_token_5><custom_token_1>")
        try:
            r = requests.post(f"{self.base}/completion", json={
                "prompt": prompt, "n_predict": 2400, "temperature": 0.6, "top_p": 0.9,
                "repeat_penalty": 1.1, "cache_prompt": False, "stream": True,
            }, stream=True, timeout=600)
            r.raise_for_status()
        except requests.RequestException:
            # A dead TTS server must not take the turn down with it. The answer is
            # already computed and printed; losing the voice is recoverable, losing
            # the loop is not.
            self._proc = None
            raise TTSUnavailable(f"orpheus server on {self.base} did not answer")

        vals: list[int] = []
        emitted = 0          # frames already yielded
        overlap_samples = 0  # samples one frame decodes to, learned on the first join
        for line in r.iter_lines():
            if self._cancel.is_set():
                r.close()
                return
            if not line or not line.startswith(b"data: "):
                continue
            for m in _TOKEN.findall(json.loads(line[6:]).get("content", "")):
                i = len(vals)
                vals.append(int(m) - 10 - ((i % FRAME) * 4096))
            while len(vals) - emitted * FRAME >= CHUNK:
                lo = max(0, emitted * FRAME - OVERLAP)
                block = vals[lo:emitted * FRAME + CHUNK]
                audio = self._decode(block)
                if emitted:
                    if not overlap_samples:
                        overlap_samples = int(len(audio) * OVERLAP / (len(block) // FRAME * FRAME))
                    audio = audio[overlap_samples:]
                emitted += CHUNK // FRAME
                if len(audio):
                    yield audio
        tail = vals[max(0, emitted * FRAME - OVERLAP):]
        if len(tail) >= FRAME:
            audio = self._decode(tail)
            if emitted and overlap_samples:
                audio = audio[overlap_samples:]
            if len(audio):
                yield audio

    def stream(self, text_chunks: Iterator[str]) -> Iterator[np.ndarray]:
        self._cancel.clear()
        for chunk in text_chunks:
            if self._cancel.is_set():
                return
            yield from self._speak(chunk)

    def cancel(self) -> None:
        self._cancel.set()

    def warmup(self) -> None:
        self._server()
        self._decoder()

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
