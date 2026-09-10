#!/usr/bin/env python3
"""Every suite, one command. Prints a table and exits non-zero if any fail.

Ordered cheapest first, so a broken import fails in seconds instead of after the
suites that load models.
"""

from __future__ import annotations

import subprocess
import sys
import time

SUITES = [
    ("wake", "tests.test_wake"),
    ("hearing", "tests.test_hearing"),
    ("presence", "tests.test_presence"),
    ("proactive", "tests.test_proactive"),
    ("watchers", "tests.test_watchers"),
    ("acks", "tests.test_acks"),
    ("preflight", "tests.test_preflight"),
    ("spoken routing", "tests.test_spoken_routing"),
    ("orpheus", "tests.test_orpheus"),
    ("duplex", "tests.test_duplex"),
    ("privacy", "tests.test_privacy"),
    ("retrieval", "tests.test_retrieval"),
    ("aec", "tests.test_aec"),
    ("corpus", "tests.test_corpus"),
    ("vad", "tests.test_vad"),
    ("listen", "tests.test_listen"),
    ("voice path", "tests.test_voice_path"),
]


def main() -> int:
    rows, failed = [], []
    for name, mod in SUITES:
        t = time.perf_counter()
        p = subprocess.run([sys.executable, "-m", mod], capture_output=True, text=True)
        dt = time.perf_counter() - t
        ok = p.returncode == 0
        rows.append((name, ok, dt, p))
        if not ok:
            failed.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name:12} {dt:6.1f}s")

    if failed:
        print(f"\n{len(failed)} suite(s) failed: {', '.join(failed)}\n")
        for name, ok, _, p in rows:
            if not ok:
                print(f"--- {name}\n{p.stdout[-1500:]}{p.stderr[-800:]}")
        return 1
    print(f"\nall {len(SUITES)} suites passed in {sum(r[2] for r in rows):.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
