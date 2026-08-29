"""A record of every spoken turn, for debugging what cannot be reproduced.

A voice bug is a bad transcript, a wrong route, or a wrong answer, and by the time
it is noticed the audio is gone. This writes one JSON line per turn so a mis-hearing
can be read back instead of re-enacted.

Local only, gitignored. Transcripts are the same class of data as the corpus.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from bruno import config

LOG_PATH = config.LOGS / "turns.jsonl"
# Roughly a month of heavy use. Old turns stop being useful once the code moved on.
MAX_LINES = 5000


def record(**fields) -> None:
    """Append one turn. Never raises: logging must not be able to end a turn."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fields["at"] = time.time()
        with LOG_PATH.open("a") as fh:
            fh.write(json.dumps(fields, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def trim() -> None:
    try:
        if not LOG_PATH.exists():
            return
        lines = LOG_PATH.read_text().splitlines()
        if len(lines) > MAX_LINES:
            LOG_PATH.write_text("\n".join(lines[-MAX_LINES:]) + "\n")
    except Exception:
        pass


def recent(n: int = 20) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    out = []
    for line in LOG_PATH.read_text().splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
