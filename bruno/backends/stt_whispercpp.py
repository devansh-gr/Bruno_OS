"""Portable STT backend: whisper.cpp server over HTTP.

Metal on macOS, CUDA on Windows. Server mode keeps the model resident, so a turn
never pays model-load cost. That is the whole reason this is not a CLI call.
"""

from __future__ import annotations

import io
import subprocess
import time
import wave

import numpy as np
import requests

from bruno import config


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class WhisperCppSTT:
    name = "whisper.cpp"

    def __init__(self, ggml=None, host=None, port=None, threads=8, audio_ctx=None):
        self.ggml = ggml or config.STT_GGML
        self.host = host or config.WHISPER_HOST
        self.port = port or config.WHISPER_PORT
        self.threads = threads
        self.audio_ctx = config.STT_AUDIO_CTX if audio_ctx is None else audio_ctx
        self.base = f"http://{self.host}:{self.port}"
        self._proc: subprocess.Popen | None = None

    @property
    def max_seconds(self) -> float:
        """Whisper's encoder covers 30s across 1500 context frames. Truncating the
        context is the single biggest latency lever (633ms -> 38ms on base.en), but
        it hard-caps how much audio a turn can contain."""
        return 30.0 * self.audio_ctx / 1500 if self.audio_ctx else 30.0

    def is_up(self) -> bool:
        try:
            # whisper-server has no /health; any response means the socket is live.
            requests.get(self.base, timeout=0.5)
            return True
        except requests.RequestException:
            return False

    def start(self, timeout=120.0) -> None:
        if self.is_up():
            return
        if not self.ggml.exists():
            raise FileNotFoundError(f"missing GGML model: {self.ggml}")
        config.LOGS.mkdir(exist_ok=True)
        log = open(config.LOGS / "whisper-server.log", "ab")
        args = [
            "whisper-server",
            "-m", str(self.ggml),
            "--host", self.host,
            "--port", str(self.port),
            "-t", str(self.threads),
            "--no-timestamps",
            "-nf",  # no fallback: one decode pass, predictable latency
        ]
        if self.audio_ctx:
            args += ["-ac", str(self.audio_ctx)]
        self._proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_up():
                return
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"whisper-server died, see {config.LOGS / 'whisper-server.log'}"
                )
            time.sleep(0.25)
        raise TimeoutError("whisper-server did not come up")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        secs = len(audio) / sample_rate
        if secs > self.max_seconds:
            # Silently transcribing only the first N seconds would drop the user's
            # words with no signal. Truncate loudly instead.
            print(
                f"  [stt] clip is {secs:.1f}s but audio_ctx={self.audio_ctx} covers "
                f"{self.max_seconds:.1f}s: tail dropped",
                flush=True,
            )
            audio = audio[: int(self.max_seconds * sample_rate)]
        r = requests.post(
            f"{self.base}/inference",
            files={"file": ("turn.wav", _to_wav(audio, sample_rate), "audio/wav")},
            data={"response_format": "json", "temperature": "0.0"},
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()

    def warmup(self) -> None:
        self.start()
        self.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32), config.SAMPLE_RATE)
