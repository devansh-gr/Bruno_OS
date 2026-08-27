"""Executes a Route: runs the tool, composes what Bruno says, enforces privacy.

The privacy rule is enforced here structurally rather than by convention. Every
piece of text that could enter a prompt arrives tagged with a Privacy class, and
`Composer.pick_backend` takes the strictest class present. There is no code path
that sends local_only text to a cloud backend, because the backend is *derived
from* the data rather than chosen alongside it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from bruno import config
from bruno.router import Route
from bruno.tools.base import REGISTRY, Privacy, Tier, ToolResult


class PrivacyViolation(RuntimeError):
    """Raised when local_only content would reach a cloud backend. Never caught."""


@dataclass
class Answer:
    text: str
    tier: Tier
    backend: str
    privacy: Privacy
    degraded: bool = False   # answered by a weaker backend than the tier wanted
    notes: list[str] = field(default_factory=list)


def strictest(*classes: Privacy) -> Privacy:
    """LOCAL_ONLY dominates. One private input makes the whole turn private."""
    return Privacy.LOCAL_ONLY if Privacy.LOCAL_ONLY in classes else Privacy.CLOUD_OK


def cloud_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class Composer:
    """Turns tool output into something worth hearing, on a backend the data allows."""

    def __init__(self, llm=None):
        self.llm = llm

    def pick_backend(self, privacy: Privacy) -> str:
        """The only place a backend is chosen. Derived from the data, never passed in."""
        if privacy is Privacy.LOCAL_ONLY:
            return "local"
        return "claude" if cloud_available() else "local"

    def _local(self, prompt: str, system: str) -> str:
        from bruno.backends.llm_llamacpp import LlamaCppLLM

        self.llm = self.llm or LlamaCppLLM()
        self.llm.start()
        return "".join(self.llm.stream(prompt, system=system)).strip()

    def _claude(self, prompt: str, system: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=config.CLOUD_MODEL,
            max_tokens=config.CLOUD_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            return "Claude declined that one."
        return next((b.text for b in resp.content if b.type == "text"), "").strip()

    def compose(self, prompt: str, privacy: Privacy, system: str | None = None) -> tuple[str, str]:
        backend = self.pick_backend(privacy)
        system = system or config.system_prompt()
        if backend == "claude":
            if privacy is Privacy.LOCAL_ONLY:  # unreachable by construction; assert anyway
                raise PrivacyViolation("local_only content routed to cloud")
            return self._claude(prompt, system), backend
        return self._local(prompt, system), backend


class Agent:
    def __init__(self, composer: Composer | None = None, launcher=None):
        self.composer = composer or Composer()
        self.launcher = launcher

    def run(self, route: Route, confirmed: bool = False) -> Answer:
        if route.tier is Tier.HEAVY:
            return self._heavy(route, confirmed)
        if route.tier is Tier.AGENT and route.tool:
            return self._tool(route, confirmed)
        if route.tier is Tier.AGENT:
            return self._teach(route)
        return self._reflex(route)

    # -- tiers --------------------------------------------------------------

    def _reflex(self, route: Route) -> Answer:
        text, backend = self.composer.compose(route.query, Privacy.LOCAL_ONLY)
        return Answer(text, Tier.REFLEX, backend, Privacy.LOCAL_ONLY)

    def _teach(self, route: Route) -> Answer:
        """The tutor path.

        Measured on SimpleQA, answering from memory scores 0/120 on the 3B and
        2/120 on the 8B, and the wrong answers are confident inventions. Web
        search puts the answer in front of the model 7 times in 8. So this
        retrieves first and only falls back to memory when the web has nothing --
        and says which one happened, because the two have very different
        trustworthiness."""
        if config.RETRIEVE_FOR_FACTS:
            grounded = self._retrieved_answer(route)
            if grounded is not None:
                return grounded

        privacy = Privacy.CLOUD_OK
        backend = self.composer.pick_backend(privacy)
        system = (
            config.system_prompt() + " Answer the question directly and correctly. "
            "If you are not confident, say so plainly instead of guessing."
        )
        text, backend = self.composer.compose(route.query, privacy, system)
        if backend == "local":
            # This branch now means retrieval found nothing and the answer came
            # from weights alone. On SimpleQA that path scores 0/120, so the
            # caveat here is not politeness -- it is the measured truth. A
            # retrieved answer gets no caveat because it was read, not recalled.
            caveat = " — that came from memory, not a source, so check it."
            return Answer(
                f"{text}{caveat}", Tier.AGENT, backend, privacy, degraded=True,
                notes=[f"answered locally; measured worst-case confident-wrong "
                       f"{config.LOCAL_CONFIDENT_WRONG_PCT:.1f}%"],
            )
        return Answer(text, Tier.AGENT, backend, privacy)

    def _retrieved_answer(self, route: Route) -> Answer | None:
        """Answer from the web, or return None so the caller can fall back."""
        from bruno import retrieve, websearch

        try:
            evidence = retrieve.gather(route.query)
        except websearch.Throttled:
            # Distinct from "nothing found", but both fall back to memory. The
            # caller marks that answer degraded either way, which is the honest
            # outcome: an answer from weights is worth less than one from sources.
            return None
        if evidence is None:
            return None

        text, backend = self.composer.compose(
            evidence.as_prompt(), Privacy.CLOUD_OK, retrieve.GROUNDED_SYSTEM)
        if not text:
            return None
        return Answer(text, Tier.AGENT, f"{backend}+search", Privacy.CLOUD_OK,
                      notes=[f"answered from {len(evidence.passages)} search passages"])

    def _tool(self, route: Route, confirmed: bool) -> Answer:
        tool = REGISTRY.get(route.tool)
        if tool is None:
            return Answer(f"I don't have a tool called {route.tool}.", route.tier, "none",
                          Privacy.LOCAL_ONLY, notes=["unknown tool"])
        if tool.confirm and not confirmed:
            return Answer(f"That would {tool.name.replace('_', ' ')}. Do you want me to?",
                          route.tier, "none", tool.privacy, notes=["awaiting confirmation"])

        result: ToolResult = tool.run(**{tool.arg: route.query})
        # The tool's declared class and its result's class must agree, or the
        # registry is lying about where its output may go.
        if result.privacy is not tool.privacy:
            raise PrivacyViolation(
                f"{tool.name} declares {tool.privacy.value} but returned {result.privacy.value}"
            )
        return Answer(result.text, Tier.AGENT, "tool", result.privacy,
                      notes=[] if result.ok else ["tool reported failure"])

    def _heavy(self, route: Route, confirmed: bool) -> Answer:
        if not confirmed:
            return Answer(
                f"That's a codegen job. I'd hand it to Claude Code. Do you want me to start it?",
                Tier.HEAVY, "none", Privacy.CLOUD_OK, notes=["awaiting confirmation"],
            )
        if not config.CLI_TARGET_DIR:
            return Answer("I don't know which project to run that in yet.", Tier.HEAVY,
                          "none", Privacy.CLOUD_OK, notes=["no target directory set"])
        from bruno.backends.launcher import get_launcher

        launcher = self.launcher or get_launcher()
        launcher.launch(config.CLI_TARGET_DIR, route.query)
        return Answer(f"Claude Code is running in {os.path.basename(config.CLI_TARGET_DIR)}.",
                      Tier.HEAVY, launcher.name, Privacy.CLOUD_OK)
