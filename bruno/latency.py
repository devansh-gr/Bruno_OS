"""Stage timing. Every number Bruno reports comes from here, never from an estimate."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Stage:
    name: str
    start: float
    end: float | None = None

    @property
    def ms(self) -> float:
        if self.end is None:
            raise RuntimeError(f"stage {self.name!r} never closed")
        return (self.end - self.start) * 1000


@dataclass
class Trace:
    """One voice turn. Stages may overlap — streaming stages usually do."""

    t0: float = field(default_factory=time.perf_counter)
    stages: list[Stage] = field(default_factory=list)
    marks: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        s = Stage(name, time.perf_counter())
        self.stages.append(s)
        try:
            yield s
        finally:
            s.end = time.perf_counter()

    def mark(self, name: str) -> float:
        """Record a point-in-time event, in ms since the turn started."""
        ms = (time.perf_counter() - self.t0) * 1000
        self.marks[name] = ms
        return ms

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def report(self) -> str:
        width = max((len(s.name) for s in self.stages), default=0)
        width = max(width, max((len(m) for m in self.marks), default=0))
        lines = ["", "  stage" + " " * max(0, width - 5) + "     ms"]
        lines.append("  " + "-" * (width + 9))
        for s in self.stages:
            lines.append(f"  {s.name:<{width}}  {s.ms:7.1f}")
        if self.marks:
            lines.append("  " + "-" * (width + 9))
            for name, ms in self.marks.items():
                lines.append(f"  {name:<{width}}  {ms:7.1f}  (from turn start)")
        lines.append("  " + "-" * (width + 9))

        gate = self.marks.get("first_audio_out")
        if gate is not None:
            verdict = "PASS" if gate < 500 else "FAIL"
            lines.append(f"  {'GATE first_audio_out < 500ms':<{width}}  {gate:7.1f}  {verdict}")
        lines.append(f"  {'total':<{width}}  {self.total_ms:7.1f}")
        return "\n".join(lines)
