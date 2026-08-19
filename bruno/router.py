"""Which tier answers, and with what tool.

Two stages, in the CPD pattern. A deterministic rule layer catches the cases that
never need a model, then a schema-constrained local classifier picks among the
rest. The 3B is only ever asked to choose from a closed enum — never to produce
free-form routing text, and never to decide anything about privacy.

Privacy is not a model output. The classifier picks *which* tool runs; the tool's
own declared class decides where its output may go. A misrouted request calls the
wrong tool; it cannot leak a local-only tool's output to the cloud.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests

from bruno import config
from bruno.memory import Conversation  # noqa: F401 — type reference
from bruno.tools.base import REGISTRY, Privacy, Tier

# Reserved actions that are not tools.
CHAT, CODE, TEACH = "chat", "code", "teach"

SYSTEM = """You route Devansh's spoken requests. Reply with JSON only.

Pick one action:
{catalog}
- teach: explain a concept, define something, answer a factual or academic question
- code: write, debug, refactor, or review code; or do heavy mathematics
- chat: greetings, acknowledgements, cancels, and anything needing no outside data

Anything asking to build, write, fix, debug, implement, refactor, or test software
is "code" — including when it names a framework, a file, or a test.

Then set "query" to the argument that action needs:
- news -> one of: world, tech, science, business
- video_search -> the search phrase
- market_quote -> the ticker symbol alone, uppercase (Nvidia is NVDA, Apple is AAPL, Bitcoin is BTC-USD)
- teach, code, chat -> echo the user's own words back, shortened. Never copy a query
  from the examples below; they show the shape, not the answer.

Examples:
"how is nvidia doing" -> {{"action": "market_quote", "query": "NVDA"}}
"what's going on in the world" -> {{"action": "news", "query": "world"}}
"find me that video on transformers" -> {{"action": "video_search", "query": "transformers explained"}}
"explain what a kalman filter is" -> {{"action": "teach", "query": "kalman filter"}}
"why is the sky blue" -> {{"action": "teach", "query": "why the sky is blue"}}
"any email" -> {{"action": "check_email", "query": ""}}
"draft an email to adam about the lab" -> {{"action": "draft_email", "query": "email adam about the lab"}}
"send it" -> {{"action": "send_email", "query": ""}}
"wait no don't send it" -> {{"action": "undo_send", "query": ""}}
"catch me up on today" -> {{"action": "briefing", "query": ""}}
"tell me more about the nvidia one" -> {{"action": "tell_me_more", "query": "nvidia"}}
"pull up that article" -> {{"action": "pull_up", "query": "that article"}}
"pull up a video on transformers" -> {{"action": "video_search", "query": "transformers explained"}}
"what's the cutoff of a 1k 1uf filter" -> {{"action": "calculate", "query": "1/(2*pi*1*k*1*u)"}}
"check amd for me" -> {{"action": "market_quote", "query": "AMD"}}
"how'd the s and p do today" -> {{"action": "market_quote", "query": "^GSPC"}}
"what's 15 percent of 340" -> {{"action": "calculate", "query": "0.15*340"}}
"what's the difference between a butterworth and chebyshev filter" -> {{"action": "teach", "query": "butterworth vs chebyshev filter"}}
"how's bruno os coming along" -> {{"action": "project_status", "query": "bruno os"}}
"is claude still working on that" -> {{"action": "claude_sessions", "query": ""}}
"play bohemian rhapsody" -> {{"action": "play_song", "query": "bohemian rhapsody"}}
"put on some kendrick" -> {{"action": "play_song", "query": "kendrick lamar"}}
"skip this song" -> {{"action": "media_control", "query": "skip"}}
"write me a script to rename files" -> {{"action": "code", "query": "script to rename files"}}
"build me an endpoint for the parser" -> {{"action": "code", "query": "endpoint for the parser"}}
"fix the failing test in the decoder" -> {{"action": "code", "query": "fix failing decoder test"}}
"implement the retry logic" -> {{"action": "code", "query": "implement retry logic"}}
"thanks bruno" -> {{"action": "chat", "query": "thanks"}}"""

# Only unambiguous patterns belong here. Over-ruling defeats the classifier and
# hides its real accuracy.
_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^\s*(pause|resume|unpause)\s*(the\s+)?(music|song|track|it)?\s*[.!]?\s*$", re.I),
     "media_control", "transport"),
    (re.compile(r"^\s*(skip|next)\s*(this\s+|the\s+)?(song|track|one)?\s*[.!]?\s*$", re.I),
     "media_control", "transport"),
    (re.compile(r"^\s*(go back|previous|last)\s*(to\s+)?(the\s+)?(song|track|one)?\s*[.!]?\s*$", re.I),
     "media_control", "transport"),
    (re.compile(r"^\s*(stop|cancel|never ?mind|shut up|quiet|forget it)\b", re.I), CHAT, "stop"),
    (re.compile(r"^\s*(thanks|thank you|cheers|nice|cool|got it|ok|okay)\s*[.!]?\s*$", re.I), CHAT, "ack"),
    (re.compile(r"\b(hand|give|send|pass)\s+(this|that|it)\s+(to|over to)\s+claude\b", re.I), CODE, "explicit handoff"),
    (re.compile(r"\bclaude code\b", re.I), CODE, "explicit handoff"),
]


@dataclass
class Route:
    tier: Tier
    tool: str | None
    query: str
    privacy: Privacy
    confirm: bool
    reason: str
    ms: float = 0.0

    def describe(self) -> str:
        what = self.tool or self.reason
        return (f"{self.tier.value:6} {what:14} privacy={self.privacy.value:10} "
                f"confirm={'yes' if self.confirm else 'no ':3} {self.ms:6.1f}ms")


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": REGISTRY.names() + [TEACH, CODE, CHAT]},
            "query": {"type": "string"},
        },
        "required": ["action", "query"],
        "additionalProperties": False,
    }


def _resolve(action: str, query: str, reason: str, ms: float) -> Route:
    """Map an action onto a tier, and take privacy from the registry — never the model."""
    t = REGISTRY.get(action)
    if t is not None:
        return Route(Tier.AGENT, t.name, query, t.privacy, t.confirm, reason, ms)
    if action == CODE:
        # The CLI handoff spawns a visible terminal and burns Pro quota. Always confirm.
        return Route(Tier.HEAVY, None, query, Privacy.CLOUD_OK, True, "codegen", ms)
    if action == TEACH:
        return Route(Tier.AGENT, None, query, Privacy.CLOUD_OK, False, "tutoring", ms)
    return Route(Tier.REFLEX, None, query, Privacy.LOCAL_ONLY, False, reason or "chat", ms)


class Router:
    def __init__(self, llm=None):
        self.llm = llm
        self._base = None

    def _server(self) -> str:
        if self._base is None:
            from bruno.backends.llm_llamacpp import LlamaCppLLM

            self.llm = self.llm or LlamaCppLLM()
            self.llm.start()
            self._base = self.llm.base
        return self._base

    def route(self, text: str, convo=None) -> Route:
        t0 = time.perf_counter()
        stripped = (text or "").strip()
        if not stripped:
            return _resolve(CHAT, "", "empty", 0.0)

        for pattern, action, why in _RULES:
            if pattern.search(stripped):
                return _resolve(action, stripped, f"rule:{why}",
                                (time.perf_counter() - t0) * 1000)

        # Routing runs on the local model, so it may see private turns; the filter
        # that matters is on the cloud path in Composer, not here.
        history = convo.context_for(Privacy.LOCAL_ONLY) if convo and len(convo) else ""
        user_msg = f"Conversation so far:\n{history}\n\nNow classify: {stripped}" if history else stripped
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM.format(catalog=REGISTRY.catalog())},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "route", "schema": _schema(), "strict": True},
            },
            # 80 was enough before the prompt grew. A long "query" now truncates
            # mid-JSON, which surfaced as a mysterious misroute rather than an
            # error -- the schema guarantees valid *shape*, not that generation
            # finishes inside the cap.
            "max_tokens": 400,
            "temperature": 0.0,
        }
        last_error = None
        for attempt in (1, 2):
            try:
                r = requests.post(f"{self._server()}/v1/chat/completions",
                                  json=body, timeout=30)
                r.raise_for_status()
                parsed = json.loads(r.json()["choices"][0]["message"]["content"])
                action = parsed["action"]
                query = (parsed.get("query") or stripped).strip()
                break
            except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                last_error = e
        else:
            # Falling back to chat keeps a routing failure cheap and local rather
            # than silently escalating to a paid tier. It is reported in `reason`
            # so a failure never masquerades as a confident classification --
            # that is exactly how a truncation bug hid as a misroute.
            return _resolve(CHAT, stripped,
                            f"classifier failed: {type(last_error).__name__}",
                            (time.perf_counter() - t0) * 1000)

        return _resolve(action, query, "classifier", (time.perf_counter() - t0) * 1000)
