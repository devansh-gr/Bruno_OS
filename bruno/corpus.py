"""Devansh's own words, for the tone fine-tune.

Strictly LOCAL_ONLY. This reads a year of private conversation; nothing here is
eligible to reach a cloud model, and the extracted files are gitignored. The
fine-tune it feeds is for *register* -- word choice, length, how he opens and
closes -- never for facts. Training a model on new facts measurably increases
hallucination on facts it already knew (arxiv 2405.05904), which is the one
failure this project most needs to avoid.

Two registers, kept apart on purpose: iMessage is how he writes to people he knows,
sent mail is how he writes to people he does not. A model trained on the mixture
splits the difference and sounds like neither.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

CHAT_DB = Path.home() / "Library" / "Messages" / "chat.db"

# Apple typedstream: after the NSString marker comes b"\x84\x01+", then a length,
# then UTF-8. Lengths under 0x80 are a single byte; 0x81 means a 2-byte
# little-endian length follows. More than half of the messages since mid-2025 are
# stored only this way, with a NULL `text` column -- reading `text` alone silently
# drops the most recent and most representative half of the corpus.
_MARK = b"\x84\x01+"


def decode_attributed(blob: bytes | None) -> str:
    if not blob:
        return ""
    i = blob.find(_MARK)
    if i < 0:
        return ""
    j = i + len(_MARK)
    if j >= len(blob):
        return ""
    n = blob[j]
    j += 1
    if n == 0x81:
        n = int.from_bytes(blob[j:j + 2], "little")
        j += 2
    elif n == 0x82:
        n = int.from_bytes(blob[j:j + 4], "little")
        j += 4
    elif n >= 0x80:
        return ""
    return blob[j:j + n].decode("utf-8", errors="ignore")


# Things that are his keystrokes but carry no register: bare acknowledgements,
# a pasted link, a reaction. Keeping them teaches the model to answer "ok".
_URL = re.compile(r"https?://\S+")
_NOISE = re.compile(
    r"^(ok|okay|k|kk|yes|no|yeah|yep|nope|lol|lmao|haha+|hi|hey|hello|thanks|ty|"
    r"thank you|sure|nice|cool|got it|np|yw|bet|fr|ong|ight|aight|wdym|hm+|"
    r"\W*)$", re.I)


@dataclass
class Sample:
    text: str
    register: str          # "informal" | "formal"
    context: str = ""      # the message being replied to, when there is one


def _keep(t: str) -> bool:
    t = t.strip()
    if len(t) < 12 or len(t) > 2000:
        return False
    if _NOISE.match(t):
        return False
    if _URL.sub("", t).strip() == "":
        return False
    # Attachment placeholders and Apple's object-replacement character.
    return "￼" not in t and "�" not in t


def imessage(limit: int | None = None) -> list[Sample]:
    """His outgoing iMessages, with the preceding inbound message as context.

    Opened read-only. This is his real message history, not a copy, and a stray
    write would be unrecoverable.
    """
    if not CHAT_DB.exists():
        return []
    db = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
    db.text_factory = bytes
    rows = db.execute("""
        select m.is_from_me, m.text, m.attributedBody, m.date, m.cache_roomnames
        from message m
        where m.associated_message_type = 0
        order by m.date
    """).fetchall()

    out: list[Sample] = []
    prev_in = ""
    for is_me, text, blob, _date, room in rows:
        body = (text.decode("utf-8", "ignore") if text else "") or decode_attributed(blob)
        body = body.strip()
        if not body:
            continue
        if not is_me:
            prev_in = body
            continue
        if _keep(body):
            # Group chats are excluded: he writes differently to a room than to a
            # person, and the reply-to pairing is meaningless with many speakers.
            if not room:
                out.append(Sample(body, "informal", prev_in[:400]))
        prev_in = ""
        if limit and len(out) >= limit:
            break
    return out


def sent_mail(limit: int = 400) -> list[Sample]:
    """His sent email. Formal register, and a much smaller corpus than iMessage."""
    from bruno import mail

    out: list[Sample] = []
    for m in mail.sent(limit=limit):
        body = _strip_quoted(m.get("body", ""))
        if _keep(body):
            out.append(Sample(body, "formal", (m.get("subject") or "")[:200]))
    return out


_QUOTE = re.compile(
    r"(^>.*$|^On .{0,80}wrote:.*|^-{2,}\s*Original Message.*|^From:.*$)",
    re.M | re.S)


def _strip_quoted(body: str) -> str:
    """Drop quoted replies and signatures. Quoted text is somebody else's register,
    and it is the majority of the bytes in most replies."""
    body = _QUOTE.split(body)[0] if _QUOTE.search(body) else body
    body = re.split(r"\n--\s*\n", body)[0]
    return re.sub(r"\n{3,}", "\n\n", body).strip()
