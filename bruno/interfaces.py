"""The four platform seams.

Everything above these protocols is one codebase on every OS. A Windows port
implements these and nothing else. If you find yourself writing `if platform ==`
outside a backend module, you are breaking the rule.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class STT(Protocol):
    """Speech to text. Implementations keep the model resident between calls."""

    name: str

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Mono float32 in [-1, 1] -> text."""
        ...

    def warmup(self) -> None: ...


@runtime_checkable
class LLM(Protocol):
    """Token generation. Must stream — first token is a latency gate."""

    name: str

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield text deltas. The first yield is the first-token measurement."""
        ...

    def cancel(self) -> None:
        """Abort in-flight generation. Required for barge-in."""
        ...

    def warmup(self) -> None: ...


@runtime_checkable
class TTS(Protocol):
    """Text to speech. Must stream — first audio chunk is a latency gate."""

    name: str
    sample_rate: int

    def stream(self, text_chunks: Iterator[str]) -> Iterator[np.ndarray]:
        """Consume text deltas, yield float32 audio chunks as soon as each is ready."""
        ...

    def cancel(self) -> None:
        """Stop mid-utterance. Required for barge-in."""
        ...

    def warmup(self) -> None: ...


@runtime_checkable
class TerminalLauncher(Protocol):
    """Spawns the Claude Code CLI in a visible terminal. AppleScript on macOS,
    PowerShell / Windows Terminal on Windows."""

    name: str

    def launch(self, cwd: str, prompt: str) -> None: ...
