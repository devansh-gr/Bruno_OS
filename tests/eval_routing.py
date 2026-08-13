#!/usr/bin/env python3
"""Score the router against the hand-labelled set.

Two numbers matter and they are not the same. Tier accuracy is whether the right
*kind* of thing answered — routing a tool request to codegen wastes Pro quota and
routing codegen to the local 3B produces garbage. Tool accuracy is only scored on
the cases that should have picked a tool.

The confusion table is the useful output: it says which direction the mistakes go.
"""

from __future__ import annotations

import collections
import statistics
import sys

from bruno.router import Router
from bruno.tools import media, public, workspace  # noqa: F401 — registers the tools
from bruno.tools.base import Tier
from tests.routing_cases import CASES


def main() -> int:
    r = Router()
    print(f"scoring {len(CASES)} cases (first call warms the server)...\n")
    r.route("warm up")

    tier_hits = tool_hits = tool_total = 0
    confusion: collections.Counter = collections.Counter()
    misses, times = [], []

    for text, want_tier, want_tool in CASES:
        got = r.route(text)
        times.append(got.ms)
        tier_ok = got.tier == want_tier
        tier_hits += tier_ok
        confusion[(want_tier.value, got.tier.value)] += 1

        tool_ok = True
        if want_tool is not None:
            tool_total += 1
            tool_ok = got.tool == want_tool
            tool_hits += tool_ok

        if not (tier_ok and tool_ok):
            misses.append((text, want_tier.value, want_tool, got.tier.value, got.tool, got.query))

    n = len(CASES)
    print(f"  tier accuracy  {tier_hits}/{n}  {tier_hits / n * 100:.1f}%")
    if tool_total:
        print(f"  tool accuracy  {tool_hits}/{tool_total}  {tool_hits / tool_total * 100:.1f}%"
              "   (only on cases that should pick a tool)")
    print(f"  latency        median {statistics.median(times):.0f}ms  max {max(times):.0f}ms")

    print("\n  confusion (expected -> got):")
    for (want, got), count in sorted(confusion.items(), key=lambda kv: -kv[1]):
        mark = " " if want == got else "*"
        print(f"   {mark} {want:7} -> {got:7}  {count}")

    if misses:
        print(f"\n  {len(misses)} misses:")
        for text, wt, wtool, gt, gtool, q in misses:
            print(f"    {text[:44]:46} want {wt}/{wtool}  got {gt}/{gtool}  query={q!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
