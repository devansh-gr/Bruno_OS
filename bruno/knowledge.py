"""Things Bruno learned by asking, and shouldn't have to ask twice.

Devansh's vault names and repo names don't match and never will: the mapping
lives in his head, not on disk. Rather than demanding a config file up front,
Bruno reports what he finds, asks when something is unfamiliar, and writes the
answer here. The map gets built by using it.

Deliberately a flat JSON file, not a database. It's tens of entries, it should be
readable and hand-editable, and it should survive being looked at.
"""

from __future__ import annotations

import json
from pathlib import Path

from bruno import config

STORE = config.ROOT / "knowledge.json"

_DEFAULT = {
    "project_aliases": {},   # any name -> canonical project name
    "ignored_projects": [],  # things Bruno should stop asking about
    "facts": {},             # free-form key -> value, learned in conversation
}


def _load() -> dict:
    if not STORE.exists():
        return json.loads(json.dumps(_DEFAULT))
    try:
        data = json.loads(STORE.read_text())
    except (json.JSONDecodeError, OSError):
        return json.loads(json.dumps(_DEFAULT))
    for k, v in _DEFAULT.items():
        data.setdefault(k, v)
    return data


def _save(data: dict) -> None:
    STORE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def canonical(name: str) -> str:
    """Resolve any known alias to the canonical project name."""
    aliases = _load()["project_aliases"]
    seen = set()
    cur = name
    # Follow the chain, but never loop: a bad edit shouldn't hang the assistant.
    while cur in aliases and cur not in seen:
        seen.add(cur)
        cur = aliases[cur]
    return cur


def link(alias: str, canonical_name: str) -> None:
    data = _load()
    data["project_aliases"][alias] = canonical_name
    _save(data)


def ignore(name: str) -> None:
    data = _load()
    if name not in data["ignored_projects"]:
        data["ignored_projects"].append(name)
    _save(data)


def ignored() -> set[str]:
    return set(_load()["ignored_projects"])


def remember(key: str, value: str) -> None:
    data = _load()
    data["facts"][key] = value
    _save(data)


def recall(key: str) -> str | None:
    return _load()["facts"].get(key)


def unknowns(names: list[str]) -> list[str]:
    """Names Bruno has neither mapped nor been told to ignore: i.e. worth asking about."""
    data = _load()
    known = set(data["project_aliases"]) | set(data["project_aliases"].values())
    return [n for n in names if n not in known and n not in set(data["ignored_projects"])]
