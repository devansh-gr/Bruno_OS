#!/usr/bin/env python3
"""Preflight: that it catches a broken install rather than deferring to a timeout."""

from __future__ import annotations

import sys
from pathlib import Path

from bruno import config, preflight


def main() -> int:
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(name)
        print(f"  {'PASS' if got == want else 'FAIL'}  {name:52} {got} (want {want})")

    checks = preflight.run(need_mic=False)
    names = {c.name for c in checks}
    for needed in ("llm weights", "stt weights", "llama-server on PATH", "vad model"):
        if needed not in names:
            fails.append(f"missing check: {needed}")
    check("every fatal dependency is checked",
          [n for n in ("llm weights", "vad model", "llama-server on PATH") if n not in names], [])

    check("this machine passes", preflight.report(checks) if False else
          all(c.ok for c in checks if c.fatal), True)

    # Unenrolled must warn, never block: Bruno is usable without a voiceprint.
    vp = [c for c in checks if c.name == "voiceprint"][0]
    check("voiceprint is a warning, not fatal", vp.fatal, False)

    # A missing model must be caught here, not surface as a server timeout.
    real = config.LLM_GGUF
    try:
        config.LLM_GGUF = Path("/nonexistent/model.gguf")
        broken = preflight.run(need_mic=False)
        llm = [c for c in broken if c.name == "llm weights"][0]
        check("a missing model fails preflight", llm.ok, False)
        check("and it is fatal", llm.fatal, True)
        check("the path is in the message", "nonexistent" in llm.detail, True)
    finally:
        config.LLM_GGUF = real

    print(f"\n{'all preflight checks passed' if not fails else str(len(fails)) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
