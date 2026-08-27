#!/usr/bin/env python3
"""Privacy enforcement. This is the test that has to actually hold.

The claim being tested: no local_only content can reach a cloud backend. It is
tested against a fake cloud backend that records everything it is given and a
registry seeded with a local_only tool whose output contains a canary string. If
the canary ever appears in the cloud transcript, the test fails.

Run: python -m tests.test_privacy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from bruno.agent import Agent, Composer, PrivacyViolation, strictest
from bruno.router import Route
from bruno.tools.base import REGISTRY, Privacy, Tier, ToolResult

CANARY = "SALARY-412-CONFIDENTIAL"
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


class RecordingComposer(Composer):
    """A composer whose cloud backend is a spy. Nothing real is ever called."""

    def __init__(self):
        super().__init__()
        self.cloud_transcript: list[str] = []
        self.local_transcript: list[str] = []

    def pick_backend(self, privacy):
        # Force cloud to be *available* so the test exercises the real decision,
        # not an accidental pass caused by having no API key on this machine.
        return "local" if privacy is Privacy.LOCAL_ONLY else "claude"

    def _claude(self, prompt, system):
        self.cloud_transcript.append(f"{system}\n{prompt}")
        return "cloud answered"

    def _local(self, prompt, system):
        self.local_transcript.append(f"{system}\n{prompt}")
        return "local answered"


def _register_fakes():
    class Private:
        name, description = "read_email", "fake private tool"
        privacy, confirm, arg = Privacy.LOCAL_ONLY, False, "query"

        def run(self, **_):
            return ToolResult(f"Your inbox mentions {CANARY}.", Privacy.LOCAL_ONLY)

    class Liar:
        """Declares local_only, returns cloud_ok. The registry contradicting itself
        is the failure mode most likely to leak in real code."""
        name, description = "liar", "declares private, returns public"
        privacy, confirm, arg = Privacy.LOCAL_ONLY, False, "query"

        def run(self, **_):
            return ToolResult("leaked", Privacy.CLOUD_OK)

    class NeedsConfirm:
        name, description = "send_email", "fake sending tool"
        privacy, confirm, arg = Privacy.LOCAL_ONLY, True, "query"

        def run(self, **_):
            return ToolResult("sent", Privacy.LOCAL_ONLY)

    for t in (Private(), Liar(), NeedsConfirm()):
        if REGISTRY.get(t.name) is None:
            REGISTRY.register(t)


def main() -> int:
    os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"  # RecordingComposer intercepts
    from bruno.tools import public  # noqa: F401 — real cloud_ok tools

    _register_fakes()
    comp = RecordingComposer()
    agent = Agent(composer=comp)

    print("privacy enforcement\n")

    check("strictest() lets one private input dominate",
          strictest(Privacy.CLOUD_OK, Privacy.LOCAL_ONLY) is Privacy.LOCAL_ONLY)
    check("strictest() stays public when all inputs are public",
          strictest(Privacy.CLOUD_OK, Privacy.CLOUD_OK) is Privacy.CLOUD_OK)

    check("local_only always picks the local backend",
          comp.pick_backend(Privacy.LOCAL_ONLY) == "local")
    check("cloud_ok uses cloud when a key is present",
          comp.pick_backend(Privacy.CLOUD_OK) == "claude")

    # The real thing: run a private tool, then confirm the canary never surfaces.
    ans = agent.run(Route(Tier.AGENT, "read_email", "unread", Privacy.LOCAL_ONLY, False, "test"))
    check("private tool output is returned to the user", CANARY in ans.text)
    check("private tool answer is tagged local_only", ans.privacy is Privacy.LOCAL_ONLY)

    # Compose over that output the way a summarize step would.
    comp.compose(f"Summarize this: {ans.text}", ans.privacy)
    joined = "\n".join(comp.cloud_transcript)
    check("canary never appears in the cloud transcript", CANARY not in joined,
          f"leaked into: {joined[:120]}")
    check("canary did reach the local backend (so the test is real)",
          any(CANARY in t for t in comp.local_transcript))

    # A composer asked to send private text to cloud must raise, not silently degrade.
    raised = False
    try:
        Composer().compose.__wrapped__  # noqa: B018 — probe only
    except AttributeError:
        pass
    try:
        c = RecordingComposer()
        c.pick_backend = lambda privacy: "claude"  # simulate a future bug
        c.compose("contains " + CANARY, Privacy.LOCAL_ONLY)
    except PrivacyViolation:
        raised = True
    check("a mis-picked cloud backend raises PrivacyViolation", raised)

    # Registry lying about its own class must be caught at run time.
    raised = False
    try:
        agent.run(Route(Tier.AGENT, "liar", "x", Privacy.LOCAL_ONLY, False, "test"))
    except PrivacyViolation:
        raised = True
    check("tool contradicting its declared privacy raises", raised)

    # Side-effecting tools must not fire without confirmation.
    ans = agent.run(Route(Tier.AGENT, "send_email", "hi", Privacy.LOCAL_ONLY, True, "test"))
    check("confirm-required tool does not run unconfirmed",
          "awaiting confirmation" in ans.notes and "sent" not in ans.text)
    ans = agent.run(Route(Tier.AGENT, "send_email", "hi", Privacy.LOCAL_ONLY, True, "test"),
                    confirmed=True)
    check("confirm-required tool runs once confirmed", ans.text == "sent")

    # Conversation history is a prompt input, so it needs the same rule.
    from bruno.memory import Conversation
    convo = Conversation()
    convo.add("what's in my inbox", f"You have mail about {CANARY}.", Privacy.LOCAL_ONLY, "read_email")
    convo.add("what's in the news", "Top 3 in tech...", Privacy.CLOUD_OK, "news")
    convo.add("how is nvidia doing", "NVIDIA is at 217 dollars.", Privacy.CLOUD_OK, "market_quote")

    cloud_ctx = convo.context_for(Privacy.CLOUD_OK)
    local_ctx = convo.context_for(Privacy.LOCAL_ONLY)
    check("cloud context excludes the private turn", CANARY not in cloud_ctx,
          f"leaked: {cloud_ctx[:100]}")
    check("cloud context still carries the public turns", "NVIDIA" in cloud_ctx)
    check("local context sees everything", CANARY in local_ctx)

    # Composing over a cloud-destined prompt built from history must stay clean.
    comp.cloud_transcript.clear()
    comp.compose(f"Given:\n{cloud_ctx}\nAnswer the follow-up.", Privacy.CLOUD_OK)
    check("canary absent from a cloud prompt built from history",
          CANARY not in "\n".join(comp.cloud_transcript))

    # Every registered tool must declare a real class — no defaults, no None.
    check("every registered tool declares a Privacy",
          all(isinstance(t.privacy, Privacy) for t in REGISTRY.all()))

    # The two artefacts that are the most sensitive things this project produces:
    # a year of his private messages, and a biometric of his voice. Neither may be
    # committed, and a stale .gitignore is the only thing standing between them and
    # a public push.
    import subprocess
    from bruno import corpus, speaker

    for label, path in (("the tone corpus", corpus.CORPUS_DIR / "train.jsonl"),
                        ("the voiceprint", speaker.PRINT_PATH)):
        rel = path.relative_to(ROOT) if path.is_absolute() and ROOT in path.parents else path
        ignored = subprocess.run(["git", "check-ignore", "-q", str(rel)],
                                 cwd=ROOT).returncode == 0
        check(f"{label} is gitignored", ignored, str(rel))

    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split("\n")
    check("no corpus file is tracked",
          not any(t.startswith("corpus/") for t in tracked))
    check("no voiceprint is tracked", "voiceprint.json" not in tracked)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("all privacy checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
