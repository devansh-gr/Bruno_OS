"""Cross-platform audio I/O via PortAudio.

Device selection lives here and nowhere else, so built-in mic, AirPods, a wired
headset, or an external interface are a config change rather than a code change.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd

from bruno import config


def list_devices() -> str:
    return str(sd.query_devices())


def set_devices(input_name: str | None = None, output_name: str | None = None) -> None:
    """Match by substring against device names. None keeps the OS default."""
    devices = sd.query_devices()

    def find(name, kind):
        want = "max_input_channels" if kind == "in" else "max_output_channels"
        for i, d in enumerate(devices):
            if d[want] > 0 and name.lower() in d["name"].lower():
                return i
        raise ValueError(f"no {kind}put device matching {name!r}")

    cur_in, cur_out = sd.default.device if isinstance(sd.default.device, (list, tuple)) else (None, None)
    sd.default.device = (
        find(input_name, "in") if input_name else cur_in,
        find(output_name, "out") if output_name else cur_out,
    )


def check_mic(timeout=8.0) -> str | None:
    """Return None if the mic is usable, else a message explaining why not.

    On macOS an InputStream opened without a TCC grant blocks forever rather than
    raising, so this probes on a thread with a deadline instead of trusting it.
    """
    result: dict = {}

    def probe():
        try:
            with sd.InputStream(
                samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=config.BLOCK_SAMPLES,
            ):
                pass
            result["ok"] = True
        except Exception as e:  # noqa: BLE001 — surfaced to the user verbatim
            result["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=probe, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        if config.IS_MAC:
            return (
                "microphone blocked — no response from Core Audio.\n"
                "  macOS has not granted this terminal microphone access.\n"
                "  System Settings > Privacy & Security > Microphone > enable your terminal,\n"
                "  then restart it. The permission is per-app, not per-script."
            )
        return "microphone blocked — the input stream never opened."
    return result.get("err")


class Recorder:
    """Push-to-talk capture. Wake word and VAD endpointing land on top of this in M2."""

    def __init__(self, sample_rate=config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    def _cb(self, indata, frames, time_info, status):
        self._q.put(indata[:, 0].copy())

    def start(self) -> None:
        while not self._q.empty():
            self._q.get_nowait()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=config.BLOCK_SAMPLES,
            callback=self._cb,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        blocks = []
        while not self._q.empty():
            blocks.append(self._q.get_nowait())
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


class Speaker:
    """Streaming playback. Barge-in needs stop() to cut audio mid-buffer, so the
    queue is drained rather than played out."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._done = threading.Event()
        self._tail = np.zeros(0, dtype=np.float32)

    def _cb(self, outdata, frames, time_info, status):
        need = frames
        out = []
        while need > 0:
            if len(self._tail) == 0:
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._done.set()
                    break
                self._tail = nxt
            take = min(need, len(self._tail))
            out.append(self._tail[:take])
            self._tail = self._tail[take:]
            need -= take
        buf = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
        if len(buf) < frames:
            buf = np.concatenate([buf, np.zeros(frames - len(buf), dtype=np.float32)])
        outdata[:, 0] = buf

    def start(self) -> None:
        self._done.clear()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", callback=self._cb
        )
        self._stream.start()

    def write(self, audio: np.ndarray) -> None:
        self._q.put(np.asarray(audio, dtype=np.float32))

    def finish(self) -> None:
        self._q.put(None)

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def stop(self) -> None:
        """Immediate cut for barge-in."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._tail = np.zeros(0, dtype=np.float32)
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None
        self._done.set()
