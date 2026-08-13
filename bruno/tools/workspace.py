"""Bruno knowing about Devansh's world: project status, and running Claude sessions.

Both are LOCAL_ONLY. Repo state, vault notes and session transcripts are all his
work in progress, and none of it should ride along into a cloud prompt.

"How's it going" means different things per project, which is why this fuses two
sources: git says what changed, and the vault's Current Status block says what it
means. Research_Lab_Outreach's real status is "94 contacts, 1 email sent" — git
knows nothing about that. Bruno_OS's is a commit an hour ago. Neither source alone
produces a report worth hearing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from bruno import knowledge
from bruno.tools.base import Privacy, ToolResult, tool

VAULT = Path.home() / "Documents" / "Devansh AI Brain" / "03 Projects"
CODE_DIRS = [Path.home() / "Documents" / "Projects", Path.home() / "Developer"]
SESSIONS = Path.home() / ".claude" / "projects"


def _ago(dt: datetime) -> str:
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    for limit, div, unit in ((3600, 60, "minute"), (86400, 3600, "hour"),
                             (604800, 86400, "day"), (2629800, 604800, "week")):
        if secs < limit:
            n = max(1, int(secs // div))
            return f"{n} {unit}{'s' if n != 1 else ''} ago"
    return f"{max(1, int(secs // 2629800))} months ago"


def _git(repo: Path) -> dict | None:
    if not (repo / ".git").is_dir():
        return None
    def run(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    iso = run("log", "-1", "--format=%cI")
    if not iso:
        return {"name": repo.name, "empty": True}
    dirty = [l for l in run("status", "--porcelain").splitlines() if l.strip()]
    ahead = run("rev-list", "--count", "@{u}..HEAD") or "0"
    return {
        "name": repo.name,
        "last": _ago(datetime.fromisoformat(iso)),
        "subject": run("log", "-1", "--format=%s"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": len(dirty),
        "unpushed": int(ahead) if ahead.isdigit() else 0,
    }


def _vault_status(project_dir: Path) -> str | None:
    """Pull the Current Status block out of a project's CLAUDE.md."""
    md = project_dir / "CLAUDE.md"
    if not md.exists():
        return None
    text = md.read_text(errors="ignore")
    m = re.search(r"##\s*Current Status\s*\n(.*?)(?=\n##\s|\Z)", text, re.S)
    if not m:
        return None
    body = re.sub(r"<!--.*?-->", "", m.group(1), flags=re.S)
    body = re.sub(r"^>\s*", "", body, flags=re.M)
    line = next((l for l in body.splitlines()
                 if l.strip() and "last updated" not in l.lower()), None)
    if not line:
        return None
    line = re.sub(r"\*\*", "", line)                    # drop bold markers
    line = re.sub(r"^\s*Status:\s*", "", line, flags=re.I)  # drop the label itself
    line = re.sub(r"\s+", " ", line).strip(" *_#-")
    return line.rstrip(".")[:200] or None


@tool(
    name="project_status",
    description=(
        "Report how DEVANSH'S OWN projects are going — recent commits, uncommitted "
        "work, and the status he wrote in his vault. Use for 'how are my projects "
        "going', 'status on everything', 'what did I work on recently', and for asking "
        "after one project by name: 'how's <project> coming along'. Never use this for "
        "questions about Claude."
    ),
    privacy=Privacy.LOCAL_ONLY,
    arg="query",
)
def project_status(query: str = "", **_) -> ToolResult:
    focus = (query or "").strip().lower()
    ignored = knowledge.ignored()

    repos: dict[str, dict] = {}
    for base in CODE_DIRS:
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if d.is_dir() and (info := _git(d)):
                repos[knowledge.canonical(d.name)] = info

    vaults: dict[str, str] = {}
    if VAULT.is_dir():
        for d in sorted(VAULT.iterdir()):
            if d.is_dir() and (st := _vault_status(d)):
                vaults[knowledge.canonical(d.name)] = st

    names = sorted((set(repos) | set(vaults)) - ignored)
    if focus and focus not in {"all", "everything", "my projects"}:
        hit = [n for n in names if focus.replace(" ", "_").lower() in n.lower()
               or n.lower() in focus.replace(" ", "_").lower()]
        names = hit or names

    if not names:
        return ToolResult("I couldn't find any projects to report on.",
                          Privacy.LOCAL_ONLY, ok=False)

    # Most recently touched first — that's what "how's it going" is really asking.
    order = {"minute": 0, "hour": 1, "day": 2, "week": 3, "month": 4}
    def recency(n):
        last = (repos.get(n) or {}).get("last", "99 months ago")
        unit = next((u for u in order if u in last), "month")
        return (order[unit], int(re.match(r"\d+", last).group()) if re.match(r"\d+", last) else 99)

    active = sorted([n for n in names if n in repos], key=recency)[:5]
    said = []
    for n in active:
        r = repos[n]
        if r.get("empty"):
            continue
        bits = [f"{n.replace('_', ' ')}, last commit {r['last']}"]
        if r["dirty"]:
            bits.append(f"{r['dirty']} uncommitted file{'s' if r['dirty'] != 1 else ''}")
        if r["unpushed"]:
            bits.append(f"{r['unpushed']} unpushed")
        if n in vaults:
            bits.append(vaults[n])
        said.append(". ".join(b.rstrip(".") for b in bits))

    quiet = [n for n in names if n in vaults and n not in repos]
    text = f"{len(active)} project{'s' if len(active) != 1 else ''} with recent code. " + " ".join(
        f"{i}. {s}." for i, s in enumerate(said, 1))
    if quiet and not focus:
        text += f" And {len(quiet)} with notes but no repo: {', '.join(q.replace('_', ' ') for q in quiet[:4])}."

    # Anything Bruno has never been told about is worth one question, not silence.
    ask = knowledge.unknowns([n for n in names if n in repos and n not in vaults])
    if ask and not focus:
        text += f" I don't know what {ask[0].replace('_', ' ')} maps to — tell me and I'll remember."

    return ToolResult(text, Privacy.LOCAL_ONLY,
                      data={"repos": repos, "vault": vaults, "unknown": ask})


def _tail(path: Path, n_bytes: int = 400_000) -> list[dict]:
    """Read only the end of a transcript. These reach 11MB and must never be
    loaded whole just to answer 'what have you been up to'."""
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - n_bytes))
        chunk = f.read().decode("utf-8", errors="ignore")
    out = []
    for line in chunk.splitlines()[1:]:          # first line is probably truncated
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _text_of(msg) -> str:
    content = (msg or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text")
    return ""


@tool(
    name="claude_sessions",
    description=(
        "Check on CLAUDE — the AI agent Devansh leaves running in terminals. Reports "
        "what each Claude session was doing and when it was last active. Use whenever "
        "the request mentions Claude, an agent, or an overnight/background run. This "
        "is about Claude's activity, not about Devansh's own code."
    ),
    privacy=Privacy.LOCAL_ONLY,
    arg="query",
)
def claude_sessions(query: str = "", **_) -> ToolResult:
    if not SESSIONS.is_dir():
        return ToolResult("I don't see any Claude Code history.", Privacy.LOCAL_ONLY, ok=False)

    focus = (query or "").strip().lower()
    found = []
    for d in SESSIONS.iterdir():
        for f in d.glob("*.jsonl"):
            found.append((f.stat().st_mtime, f, d.name))
    if not found:
        return ToolResult("No Claude sessions on disk yet.", Privacy.LOCAL_ONLY, ok=False)

    found.sort(reverse=True)
    if focus:
        hits = [x for x in found if focus.replace(" ", "-").lower() in x[2].lower()]
        found = hits or found
    found = found[:3]

    said, data = [], []
    for mtime, path, dirname in found:
        rows = _tail(path)
        title = next((r["aiTitle"] for r in reversed(rows) if r.get("type") == "ai-title"), None)
        last_asst = next((_text_of(r.get("message")) for r in reversed(rows)
                          if r.get("type") == "assistant" and _text_of(r.get("message"))), "")
        sid = next((r["sessionId"] for r in reversed(rows) if r.get("sessionId")), path.stem)
        cwd = next((r["cwd"] for r in reversed(rows) if r.get("cwd")), dirname)
        when = _ago(datetime.fromtimestamp(mtime, timezone.utc))
        project = Path(cwd).name
        summary = title or (last_asst[:140] + "..." if len(last_asst) > 140 else last_asst)
        said.append(f"{project}, last active {when}" + (f". {summary}" if summary else ""))
        data.append({"session_id": sid, "cwd": cwd, "project": project,
                     "last_active": when, "title": title})

    return ToolResult(". ".join(f"{i}. {s}" for i, s in enumerate(said, 1)) + ".",
                      Privacy.LOCAL_ONLY, data=data)


def ping_session(session_id: str, message: str, cwd: str, timeout: int = 300) -> ToolResult:
    """Send a message into an existing Claude Code session and return its reply.

    Not registered as a routed tool — it costs Pro quota and takes minutes, so it
    runs only behind an explicit confirmation from the claude_sessions flow.
    """
    try:
        r = subprocess.run(
            ["claude", "--resume", session_id, "-p", message],
            cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "CLAUDE_NO_COLOR": "1"},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(f"Claude didn't answer within {timeout // 60} minutes.",
                          Privacy.LOCAL_ONLY, ok=False)
    except FileNotFoundError:
        return ToolResult("The claude CLI isn't on the path.", Privacy.LOCAL_ONLY, ok=False)

    if r.returncode != 0:
        return ToolResult(f"Claude returned an error: {r.stderr.strip()[:160]}",
                          Privacy.LOCAL_ONLY, ok=False)
    reply = r.stdout.strip()
    return ToolResult(reply or "Claude replied with nothing.", Privacy.LOCAL_ONLY, data=reply)
