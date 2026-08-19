"""Email through Mail.app: local, no OAuth, no API keys.

Mail.app is AppleScript-scriptable and already holds all four accounts, so Bruno
reads and writes mail without a single credential leaving the machine. That is
also the only way this could be LOCAL_ONLY honestly — a Gmail API integration
would mean tokens, a cloud round trip, and message bodies in transit.

The hard part is not fetching. It is that his inbox is mostly noise: of twelve
recent messages, two were Uber receipts, two were a jobs newsletter, one was a
gift-card promo. Reading the last five aloud would be actively annoying.

Header signals alone do not work. `List-Unsubscribe` flags a financial-aid notice
that genuinely matters, and misses `noreply@uber.com` entirely. Address shape
alone is no better: `appleid`, `rorecord` and `ceehub` all look exactly like a
person's username.

The signal that actually works is **has he ever written to this address**. Pulled
from his own Sent folder, it separates a PI replying to his outreach from a role
account that happens to be named like a human, and it needs no heuristics about
what a name looks like. It is also free: he built the list himself by sending
mail. Fetching headers turned out to cost 21 seconds for 39 messages and to add
nothing the correspondent list did not already give, so it is gone.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime

# Addresses that are never a person typing to him.
ROBOT_LOCAL = re.compile(
    r"^(no-?reply|do-?not-?reply|donotreply|notifications?|alerts?|mailer|"
    r"bounce|postmaster|automated|auto|system|support|info|hello|team|news|"
    r"updates?|digest|newsletter|marketing|promo)", re.I)

# Subject shapes that mark bulk mail regardless of sender.
BULK_SUBJECT = re.compile(
    r"(unsubscribe|newsletter|webinar|% off|sale|deal|save \d|"
    r"receipt|invoice|your .* trip|order (confirm|shipped)|verify your|"
    r"password reset|sign-?in|weekly digest|daily digest|chance to win|"
    r"gift card|survey|giveaway|raffle|announcements?:|reminder:|"
    r"your .* (id|account) .* updated|trusted device)", re.I)

# Words that mean an institution needs something from him.
ACTION_SUBJECT = re.compile(
    r"(cannot process|action required|overdue|deadline|expires?|urgent|"
    r"payment|owing|balance|suspend|hold on your|missing|incomplete|"
    r"interview|offer|accepted|rejected|decision)", re.I)


@dataclass
class Message:
    account: str
    sender_name: str
    sender_addr: str
    subject: str
    received: str
    bulk_header: bool
    bucket: str = "noise"
    score: float = 0.0
    why: list[str] = field(default_factory=list)

    def spoken(self) -> str:
        who = self.sender_name or self.sender_addr.split("@")[0]
        return f"{who}: {self.subject}"


def _osa(script: str, *args: str, timeout: int = 60) -> str:
    r = subprocess.run(["osascript", "-", *args], input=script,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:200])
    return r.stdout


SENT = r'''
on run argv
  set n to (item 1 of argv) as integer
  tell application "Mail"
    set out to ""
    repeat with a in accounts
      try
        set sb to first mailbox of a whose name is "Sent Items"
      on error
        try
          set sb to first mailbox of a whose name is "Sent Messages"
        on error
          try
            set sb to first mailbox of a whose name is "Sent"
          on error
            set sb to missing value
          end try
        end try
      end try
      if sb is not missing value then
        set total to count of messages of sb
        set lim to n
        if total < lim then set lim to total
        repeat with i from 1 to lim
          try
            set m to message i of sb
            repeat with r in to recipients of m
              set out to out & (address of r) & linefeed
            end repeat
          end try
        end repeat
      end if
    end repeat
    return out
  end tell
end run
'''

FETCH = r'''
on run argv
  set n to (item 1 of argv) as integer
  tell application "Mail"
    set out to ""
    repeat with a in accounts
      set acctName to name of a
      try
        set ib to first mailbox of a whose name is "Inbox"
      on error
        try
          set ib to first mailbox of a whose name is "INBOX"
        on error
          set ib to missing value
        end try
      end try
      if ib is not missing value then
        set total to count of messages of ib
        set lim to n
        if total < lim then set lim to total
        repeat with i from 1 to lim
          try
            set m to message i of ib
            set isBulk to "0"
            set sName to ""
            try
              set sName to extract name from sender of m
            end try
            set out to out & acctName & tab & sName & tab & (extract address from sender of m) & tab & isBulk & tab & ((date received of m) as string) & tab & (subject of m) & linefeed
          end try
        end repeat
      end if
    end repeat
    return out
  end tell
end run
'''


_CORRESPONDENTS: set[str] | None = None


def correspondents(scan: int = 120, refresh: bool = False) -> set[str]:
    """Everyone he has ever written to, from his own Sent folders.

    This is the load-bearing signal. Nothing about an address tells you whether a
    human is behind it -- `appleid`, `rorecord` and `ceehub` are shaped exactly
    like a person's username -- but a reply from someone he emailed first is
    unambiguous, and it is precisely his lab-outreach case.
    """
    global _CORRESPONDENTS
    if _CORRESPONDENTS is not None and not refresh:
        return _CORRESPONDENTS
    try:
        raw = _osa(SENT, str(scan), timeout=120)
    except (RuntimeError, subprocess.TimeoutExpired):
        _CORRESPONDENTS = set()
        return _CORRESPONDENTS
    _CORRESPONDENTS = {a.strip().lower() for a in raw.splitlines() if "@" in a}
    return _CORRESPONDENTS


def classify(m: Message, known: set[str] | None = None) -> Message:
    known = known if known is not None else set()
    addr, subj = m.sender_addr.lower(), m.subject or ""
    local = addr.split("@")[0]
    domain = addr.split("@")[-1]
    robot = bool(ROBOT_LOCAL.match(local))
    bulky = bool(BULK_SUBJECT.search(subj))
    is_reply = subj.lower().startswith(("re:", "re :", "fwd:"))
    he_wrote_first = addr in known

    # A known correspondent still sends bulk mail. He emailed a university hub
    # account once and now gets its gift-card promos; being in his Sent folder
    # does not make "Chance to win 1 of 3 $100 gift cards" worth hearing.
    if he_wrote_first and not robot and not bulky:
        m.bucket = "reply" if is_reply else "direct"
        m.score = 12.0 if is_reply else 9.0
        m.why.append("you emailed them first")
        if domain.endswith((".edu", ".ac.uk")):
            m.score += 2.0
            m.why.append("academic")
    elif is_reply and not robot and not bulky:
        # A reply from someone he never wrote to is still probably a human, just
        # a weaker signal -- an introduction, or a thread he was added to.
        m.bucket, m.score = "reply", 8.0
        m.why.append("reply, but you did not write first")
    elif robot or bulky or m.bulk_header:
        if ACTION_SUBJECT.search(subj):
            m.bucket, m.score = "action", 6.0
            m.why.append("automated but needs action")
        else:
            m.bucket, m.score = "noise", 0.0
            m.why.append("bulk or automated")
    elif ACTION_SUBJECT.search(subj):
        m.bucket, m.score = "action", 5.0
        m.why.append("looks like it needs a response")
    else:
        # Unknown sender, no reply marker, nothing urgent. Almost always a
        # mailing list or an org that got his address somewhere.
        m.bucket, m.score = "noise", 1.0
        m.why.append("unknown sender, nothing urgent")
    return m


def fetch(per_account: int = 15) -> list[Message]:
    known = correspondents()
    raw = _osa(FETCH, str(per_account))
    out = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        acct, name, addr, bulk, when, subject = parts[0], parts[1], parts[2], parts[3], parts[4], "\t".join(parts[5:])
        out.append(classify(Message(acct, name.strip(), addr.strip(), subject.strip(),
                                    when.strip(), bulk == "1"), known))
    return out


def worth_hearing(msgs: list[Message]) -> list[Message]:
    keep = [m for m in msgs if m.bucket != "noise"]
    return sorted(keep, key=lambda m: -m.score)


# --- composing -----------------------------------------------------------
#
# Undo is a delay, not a recall. Once SMTP has the message it is gone, and any
# product that claims otherwise is really just holding it back for a few seconds.
# So a "send" schedules the real send and returns immediately; "undo" cancels the
# timer. Honest, and it matches what he actually asked for.

import threading

UNDO_SECONDS = 20

_PENDING: dict = {}          # draft_id -> {"timer", "draft", "sent"}
_LOCK = threading.Lock()


@dataclass
class Draft:
    to: str
    subject: str
    body: str
    account: str = ""
    draft_id: str = ""

    def spoken(self) -> str:
        return (f"To {self.to}. Subject, {self.subject}. "
                f"It reads: {self.body}")


COMPOSE = r'''
on run argv
  set theTo to item 1 of argv
  set theSubject to item 2 of argv
  set theBody to item 3 of argv
  tell application "Mail"
    set msg to make new outgoing message with properties {subject:theSubject, content:theBody, visible:false}
    tell msg
      make new to recipient at end of to recipients with properties {address:theTo}
    end tell
    save msg
    return "ok"
  end tell
end run
'''

SEND_LAST = r'''
on run argv
  set theTo to item 1 of argv
  set theSubject to item 2 of argv
  tell application "Mail"
    repeat with m in (every outgoing message)
      if subject of m is theSubject then
        send m
        return "sent"
      end if
    end repeat
    return "notfound"
  end tell
end run
'''

DISCARD = r'''
on run argv
  set theSubject to item 1 of argv
  tell application "Mail"
    repeat with m in (every outgoing message)
      if subject of m is theSubject then
        close m saving no
        return "discarded"
      end if
    end repeat
    return "notfound"
  end tell
end run
'''


def compose(to: str, subject: str, body: str) -> Draft:
    """Create a draft in Mail.app. Nothing is sent."""
    _osa(COMPOSE, to, subject, body, timeout=30)
    d = Draft(to=to, subject=subject, body=body, draft_id=subject)
    with _LOCK:
        _PENDING[d.draft_id] = {"timer": None, "draft": d, "sent": False}
    return d


def send(draft_id: str, delay: int = UNDO_SECONDS) -> str:
    """Schedule the send. Returns immediately; the message leaves after `delay`."""
    with _LOCK:
        entry = _PENDING.get(draft_id)
        if entry is None:
            return "I don't have that draft any more."
        if entry["timer"] is not None:
            return "That one is already on its way."

        def _fire():
            with _LOCK:
                e = _PENDING.get(draft_id)
                if e is None or e["sent"]:
                    return
                e["sent"] = True
            try:
                _osa(SEND_LAST, entry["draft"].to, entry["draft"].subject, timeout=60)
            except (RuntimeError, subprocess.TimeoutExpired):
                pass

        t = threading.Timer(delay, _fire)
        t.daemon = True
        entry["timer"] = t
        t.start()
    return f"Sending in {delay} seconds. Say undo if you want it back."


def undo(draft_id: str) -> str:
    with _LOCK:
        entry = _PENDING.get(draft_id)
        if entry is None:
            return "I don't have that draft."
        if entry["sent"]:
            # Deliberately blunt. Pretending otherwise is the one thing that
            # would make this feature worse than not having it.
            return "Too late, that one already went out."
        if entry["timer"] is not None:
            entry["timer"].cancel()
            entry["timer"] = None
    return "Held it. The draft is still in your drafts folder."


def discard(draft_id: str) -> str:
    undo(draft_id)
    try:
        _osa(DISCARD, draft_id, timeout=30)
    except (RuntimeError, subprocess.TimeoutExpired):
        return "I couldn't delete the draft."
    with _LOCK:
        _PENDING.pop(draft_id, None)
    return "Deleted it."


def pending() -> Draft | None:
    with _LOCK:
        for e in reversed(list(_PENDING.values())):
            if not e["sent"]:
                return e["draft"]
    return None
