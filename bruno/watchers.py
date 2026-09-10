"""What Bruno watches, and what he thinks is worth saying about it.

Each one wraps a tool that already exists, so the proactive path and the asked-for
path cannot disagree about what counts as a real email or a live session.

The bar is deliberately high. A proactive assistant that speaks too often gets
muted, and a muted assistant is worth less than no assistant.
"""

from __future__ import annotations

import time
from datetime import datetime

from bruno.presence import Urgency
from bruno.proactive import Notice


def email() -> list[Notice]:
    """Mail from an actual human. Reuses the Sent-folder cross-reference, so this
    agrees with what "any email?" would have said."""
    from bruno import mail

    out, keys = [], set()
    for m in mail.worth_hearing(mail.fetch(per_account=10)):
        who = (m.sender_name or m.sender_addr).split("<")[0].strip()
        subject = (m.subject or "").strip()
        key = f"mail:{m.sender_addr}:{subject}"
        # Accounts overlap, so the same message arrives more than once per fetch.
        # Deduping only at delivery would spend the cap below on copies.
        if key in keys:
            continue
        keys.add(key)
        out.append(Notice(
            key=key,
            text=f"{who} emailed you" + (f", about {subject}." if subject else "."),
            urgency=Urgency.NORMAL,
        ))
    return out[:3]      # three at once is a digest; more is a lecture


# A session is working if its transcript was written to this recently, and has
# stopped if it then goes quiet for longer than a normal pause between turns.
WORKING_S = 10 * 60
STOPPED_S = 20 * 60
_last_seen: dict[str, float] = {}


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
            elif was_working and idle > STOPPED_S:
                _last_seen.pop(f.stem, None)
                project = d.name.split("-")[-1] or d.name
                out.append(Notice(
                    key=f"claude-stopped:{f.stem}:{int(mtime)}",
                    text=f"The Claude session in {project} has gone quiet. "
                         f"It ran for a while and stopped about "
                         f"{int(idle // 60)} minutes ago.",
                    urgency=Urgency.NORMAL))
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
