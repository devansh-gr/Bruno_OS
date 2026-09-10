"""What Bruno watches, and what he thinks is worth saying about it.

Each one wraps a tool that already exists, so the proactive path and the asked-for
path cannot disagree about what counts as a real email or a live session.

The bar is deliberately high. A proactive assistant that speaks too often gets
muted, and a muted assistant is worth less than no assistant.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

from bruno import config
from bruno.presence import Urgency
from bruno.proactive import Notice


def email() -> list[Notice]:
    """Mail from an actual human. Reuses the Sent-folder cross-reference, so this
    agrees with what "any email?" would have said."""
    from bruno import mail

    seen, items = set(), []
    for m in mail.worth_hearing(mail.fetch(per_account=10)):
        who = (m.sender_name or m.sender_addr).split("<")[0].strip()
        subject = (m.subject or "").strip()
        key = f"mail:{m.sender_addr}:{subject}"
        # Accounts overlap, so the same message arrives more than once per fetch.
        if key in seen:
            continue
        seen.add(key)
        items.append((key, who, subject))
    if not items:
        return []

    # One interruption, not one per message. He gets around a dozen real emails a
    # day; announcing each separately is a dozen interruptions and the predictable
    # response to that is muting him. The key covers every message in the batch, so
    # a later poll that sees the same set says nothing.
    if len(items) == 1:
        _, who, subject = items[0]
        text = f"{who} emailed you" + (f", about {subject}." if subject else ".")
    else:
        names = [w for _, w, _ in items[:3]]
        who = ", ".join(names[:-1]) + f" and {names[-1]}"
        more = f", and {len(items) - 3} more" if len(items) > 3 else ""
        text = f"You have {len(items)} emails worth reading. {who}{more}."
    return [Notice(key="mail:" + "|".join(sorted(k for k, _, _ in items)),
                   text=text, urgency=Urgency.NORMAL)]


# A session is working if its transcript was written to this recently, and has
# stopped if it then goes quiet for longer than a normal pause between turns.
WORKING_S = 10 * 60
STOPPED_S = 20 * 60
# On disk, because the transition is only visible by comparing polls. A restart
# with this in memory forgets every running session, so the overnight run that
# finished while Bruno was down is never mentioned -- which is the exact case he
# asked for.
_SEEN_PATH = config.ROOT / "sessions_seen.json"


def _load_last_seen() -> dict[str, float]:
    try:
        return {k: float(v) for k, v in json.loads(_SEEN_PATH.read_text()).items()}
    except Exception:
        return {}


def _save_last_seen(d: dict[str, float]) -> None:
    try:
        _SEEN_PATH.write_text(json.dumps(d))
    except Exception:
        pass


_last_seen: dict[str, float] = _load_last_seen()


def claude_sessions() -> list[Notice]:
    """A run that has *stopped*, which is the thing he actually asked about: 'ping
    claude and ask what its been up to since we left it running over night'.

    Not the session listing. Reading out which terminals exist, once a day, is a
    status report rather than news, and an assistant that reports status unprompted
    gets muted.
    """
    from bruno.tools.workspace import SESSIONS

    if not SESSIONS.is_dir():
        return []
    now = time.time()
    out = []
    changed = False
    for d in SESSIONS.iterdir():
        for f in d.glob("*.jsonl"):
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            idle = now - mtime
            was_working = _last_seen.get(f.stem, 0) > 0 and \
                now - _last_seen[f.stem] < WORKING_S * 3
            if idle < WORKING_S:
                _last_seen[f.stem] = now          # still going
                changed = True
            elif was_working and idle > STOPPED_S:
                _last_seen.pop(f.stem, None)
                changed = True
                project = d.name.split("-")[-1] or d.name
                out.append(Notice(
                    key=f"claude-stopped:{f.stem}:{int(mtime)}",
                    text=f"The Claude session in {project} has gone quiet. "
                         f"It ran for a while and stopped about "
                         f"{int(idle // 60)} minutes ago.",
                    urgency=Urgency.NORMAL))
    if changed:
        _save_last_seen(_last_seen)
    return out[:2]


def morning_briefing(hour: int = 8, until_hour: int = 12) -> list[Notice]:
    """Once a day, the first time he is around after `hour`.

    An exact hour match missed the whole day if Bruno was not running at 8am, which
    is most days: he starts it when he sits down. The window runs to midday and the
    key is the date, so it fires once whenever he first shows up inside it.
    """
    now = datetime.now()
    if not (hour <= now.hour < until_hour):
        return []
    return [Notice(key=f"briefing:{now:%Y-%m-%d}",
                   text="Morning. I've got a briefing when you want it.",
                   urgency=Urgency.CHATTER)]


def install(p, *, email_every=600.0, claude_every=900.0, briefing_every=1800.0) -> None:
    """Attach the standard set. Intervals are minutes, not seconds: none of this is
    urgent, and polling Mail.app over AppleScript is not free."""
    p.watch("email", email_every, email)
    p.watch("claude", claude_every, claude_sessions)
    p.watch("briefing", briefing_every, morning_briefing)
