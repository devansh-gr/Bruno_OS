"""Seam five: tools.

A tool declares its own privacy class. The router never infers it and the model
never chooses it — the classifier picks *which* tool runs, and the tool's declared
class decides where its output is allowed to go. That ordering is the whole point:
a model that misclassifies can call the wrong tool, but it cannot leak a
local-only tool's output to the cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable


class Privacy(str, Enum):
    LOCAL_ONLY = "local_only"  # output must never enter a cloud prompt
    CLOUD_OK = "cloud_ok"      # public data, safe to send anywhere


class Tier(str, Enum):
    REFLEX = "reflex"  # local 3B — sub-500ms, no tool
    AGENT = "agent"    # Claude API — tool selection and reasoning
    HEAVY = "heavy"    # Claude Code CLI — codegen and hard mathematics


@dataclass
class ToolResult:
    text: str
    privacy: Privacy
    ok: bool = True
    data: Any = None

    def __post_init__(self):
        # A result carries its privacy class with it, so downstream code cannot
        # lose track of where it came from.
        if not isinstance(self.privacy, Privacy):
            raise TypeError(f"privacy must be a Privacy, got {type(self.privacy)}")


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str      # what the classifier reads to pick this tool
    privacy: Privacy
    confirm: bool         # side effects require a spoken yes before running
    arg: str              # name of the parameter the router's `query` maps onto

    def run(self, **kwargs) -> ToolResult: ...


@dataclass
class Registry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[n] for n in self.names()]

    def catalog(self) -> str:
        """The tool menu shown to the classifier. Names and descriptions only —
        privacy is deliberately withheld so the model cannot reason about it."""
        return "\n".join(f"- {t.name}: {t.description}" for t in self.all())


REGISTRY = Registry()


def tool(name: str, description: str, privacy: Privacy, arg: str, confirm: bool = False):
    """Register a plain function as a tool. `arg` is the parameter the router's
    single `query` string maps onto — a 3B extracts one argument reliably and a
    polymorphic object badly, so tools take one primary parameter."""

    def wrap(fn: Callable[..., ToolResult]):
        class _Fn:
            pass

        t = _Fn()
        t.name, t.description, t.privacy, t.confirm, t.arg = name, description, privacy, confirm, arg
        t.run = fn
        REGISTRY.register(t)
        return fn

    return wrap
