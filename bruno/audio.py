"""Cross-platform audio I/O via PortAudio.

Device selection lives here and nowhere else, so built-in mic, AirPods, a wired
headset, or an external interface are a config change rather than a code change.
"""

from __future__ import annotations

import queue
import time
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
        except Exception as e:  # noqa: BLE001: surfaced to the user verbatim
            result["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=probe, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        if config.IS_MAC:
            return (
                "microphone blocked: no response from Core Audio.\n"
                "  macOS has not granted this terminal microphone access.\n"
                "  System Settings > Privacy & Security > Microphone > enable your terminal,\n"
                "  then restart it. The permission is per-app, not per-script."
            )
        return "microphone blocked: the input stream never opened."
    return result.get("err")


class Recorder:
    """Continuous capture into a ring buffer, with arm/disarm rather than open/close.

    A freshly opened Core Audio input stream delivers silent blocks for up to ~0.5s.
    Measured on the built-in mic: of five consecutive open-record-close cycles, one
    returned a full second of pure zeros and another was silent for its first 15
    blocks. Opening a stream per turn would clip the start of every sentence.

    So the stream opens once and stays open. Arming only moves a read pointer, which
    also lets a turn include PRE-ROLL (audio from just before the key was pressed)
    so the first word survives even if the user starts talking early. VAD endpointing
    and the wake word both need continuous audio anyway.
    """

    def __init__(self, sample_rate=config.SAMPLE_RATE, ring_seconds=30.0, pre_roll=0.3):
        self.sample_rate = sample_rate
        self.pre_roll = pre_roll
        self._n = int(ring_seconds * sample_rate)
        self._ring = np.zeros(self._n, dtype=np.float32)
        self._w = 0            # total samples ever written
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._armed_at: int | None = None

    def _cb(self, indata, frames, time_info, status):
        x = indata[:, 0]
        with self._lock:
            i = self._w % self._n
            end = i + len(x)
            if end <= self._n:
                self._ring[i:end] = x
            else:
                split = self._n - i
                self._ring[i:] = x[:split]
                self._ring[: end - self._n] = x[split:]
            self._w += len(x)

    def open(self, warmup=True, timeout=3.0) -> None:
        """Open the stream and, by default, block until real samples are flowing."""
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=config.BLOCK_SAMPLES,
            callback=self._cb,
        )
        self._stream.start()
        if not warmup:
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.05)
            with self._lock:
                w = self._w
                if w >= config.BLOCK_SAMPLES * 3:
                    tail = self._read(max(0, w - config.BLOCK_SAMPLES * 3), w)
                    if np.abs(tail).max() > 0:
                        return
        # Dead silence is possible in a very quiet room; do not fail the run on it.

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _read(self, lo: int, hi: int) -> np.ndarray:
        """Samples in absolute index range [lo, hi). Caller holds the lock."""
        lo = max(lo, hi - self._n, 0)
        if hi <= lo:
            return np.zeros(0, dtype=np.float32)
        a, b = lo % self._n, hi % self._n
        if hi - lo == self._n or a >= b:
            return np.concatenate([self._ring[a:], self._ring[:b]])
        return self._ring[a:b].copy()

    def since(self, cursor: int | None) -> tuple[np.ndarray, int]:
        """New samples since an absolute cursor, and the cursor to pass next time.

        Absolute indices rather than an offset into the ring, so a slow consumer
        cannot silently read wrapped garbage -- `_read` clamps to what is still
        held and the gap shows up as missing audio, not as audio from 30s ago.
        Passing None starts at the live edge, which is what a listener wants: no
        backlog of whatever was said before it started listening.
        """
        with self._lock:
            if cursor is None:
                return np.zeros(0, dtype=np.float32), self._w
            return self._read(cursor, self._w), self._w

    def window(self, lo: int) -> np.ndarray:
        """Everything from absolute index `lo` to the live edge."""
        with self._lock:
            return self._read(lo, self._w)

    def start(self) -> None:
        self.open()
        with self._lock:
            self._armed_at = max(0, self._w - int(self.pre_roll * self.sample_rate))

    def stop(self) -> np.ndarray:
        with self._lock:
            if self._armed_at is None:
                return np.zeros(0, dtype=np.float32)
            clip = self._read(self._armed_at, self._w)
            self._armed_at = None
        return clip


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


# Echo depends entirely on whether sound reaches the mic through the air. Measured:
# laptop speakers put Bruno's own voice in the mic at peak 0.435 against a 0.019 room
# floor; AirPods put it at 0.06. Same code, same room, two different problems.
_HEADPHONE_HINTS = ("airpod", "headphone", "earbud", "buds", "headset", "beats")


def output_is_headphones() -> bool:
    """True when playback goes to something worn, so nothing reaches the mic."""
    try:
        devs = sd.query_devices()
        out = sd.default.device[1]
        name = devs[out]["name"].lower() if out is not None else ""
    except Exception:
        return False
    return any(h in name for h in _HEADPHONE_HINTS)


def output_name() -> str:
    try:
        return sd.query_devices()[sd.default.device[1]]["name"]
    except Exception:
        return "unknown"
