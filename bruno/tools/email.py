"""Email tools. All LOCAL_ONLY: bodies and senders never enter a cloud prompt.

Drafting composes with the local model deliberately, not only for privacy. The
whole point of the tone fine-tune is that Bruno writes in Devansh's register, and
that model will live locally. Routing drafting to Claude would produce better
prose in the wrong voice.
"""

from __future__ import annotations

from bruno import mail
from bruno.tools.base import Privacy, ToolResult, tool

_LAST_LIST: list = []


@tool(
    name="check_email",
    description=(
        "Check for email actually worth hearing about: replies from real people, "
        "and automated mail that needs action. Newsletters, receipts and promos are "
        "filtered out. Use for 'any email', 'check my inbox', 'anything important', "
        "'did anyone reply'."
    ),
    privacy=Privacy.LOCAL_ONLY,
    arg="query",
)
def check_email(query: str = "", **_) -> ToolResult:
    global _LAST_LIST
    try:
        msgs = mail.fetch(per_account=12)
    except Exception as e:  # noqa: BLE001: Mail.app raises a variety
        return ToolResult(f"I couldn't reach Mail: {e}", Privacy.LOCAL_ONLY, ok=False)

    keep = mail.worth_hearing(msgs)
    _LAST_LIST = keep
    if not keep:
        return ToolResult(f"Nothing worth reading. {len(msgs)} messages, all noise.",
                          Privacy.LOCAL_ONLY)

    replies = [m for m in keep if m.bucket == "reply"]
    action = [m for m in keep if m.bucket == "action"]
    lines = [m.spoken() for m in keep[:5]]
    head = f"{len(keep)} worth your time out of {len(msgs)}"
    if replies:
        head += f", {len(replies)} of them replies"
    if action:
        head += f", {len(action)} needing action"
    return ToolResult(f"{head}. " + ". ".join(lines) + ".", Privacy.LOCAL_ONLY,
                      data=[{"from": m.sender_addr, "subject": m.subject,
                             "bucket": m.bucket} for m in keep])


@tool(
    name="draft_email",
    description=(
        "Write an email draft. Use for 'draft an email to X about Y', 'write back "
        "to X', 'reply to that saying Y'. The draft is saved but NEVER sent: "
        "Devansh has to hear it and say send."
    ),
    privacy=Privacy.LOCAL_ONLY,
    arg="query",
    confirm=False,
)
def draft_email(query: str = "", **_) -> ToolResult:
    from bruno.backends.llm_llamacpp import LlamaCppLLM
    from bruno import config

    if not (query or "").strip():
        return ToolResult("Who am I writing to, and about what?",
                          Privacy.LOCAL_ONLY, ok=False)

    # Resolve "reply to that" against whatever check_email just read out.
    recipient = ""
    for m in _LAST_LIST:
        if any(w in query.lower() for w in
               (m.sender_addr.split("@")[0].lower(), (m.sender_name or "").lower())
               if w and len(w) > 3):
            recipient = m.sender_addr
            break

    llm = LlamaCppLLM()
    llm.start()
    system = (
        "You write email as Devansh, a biomedical engineering student. "
        "Output ONLY the email body: no subject line, no greeting boilerplate "
        "like 'I hope this finds you well', no sign-off beyond his first name. "
        "Direct, short, technical. Three or four sentences at most."
    )
    body = "".join(llm.stream(f"Write this email: {query}", system=system)).strip()

    subj_sys = ("Write ONLY a short email subject line for this request. "
                "No quotes, no prefix, under eight words.")
    subject = "".join(llm.stream(query, system=subj_sys)).strip().strip('"')[:90]

    if not recipient:
        return ToolResult(
            f"I've written it but I don't know who it goes to. "
            f"Subject would be: {subject}. It reads: {body}",
            Privacy.LOCAL_ONLY, ok=False,
            data={"subject": subject, "body": body, "needs_recipient": True})

    d = mail.compose(recipient, subject, body)
    return ToolResult(
        f"Drafted to {recipient}. Subject, {subject}. It reads: {body}. "
        f"Say send, or tell me what to change.",
        Privacy.LOCAL_ONLY,
        data={"to": recipient, "subject": subject, "body": body})


@tool(
    name="read_draft",
    description=(
        "Read the current email draft back out loud. Use for 'read it back', "
        "'what does it say', 'read me the draft'."
    ),
    privacy=Privacy.LOCAL_ONLY,
    arg="query",
)
def read_draft(query: str = "", **_) -> ToolResult:
    d = mail.pending()
    if d is None:
        return ToolResult("There's no draft waiting.", Privacy.LOCAL_ONLY, ok=False)
    return ToolResult(d.spoken(), Privacy.LOCAL_ONLY,
                      data={"to": d.to, "subject": d.subject})


@tool(
    name="send_email",
    description=(
        "Send the draft Devansh just heard. Use for 'send it', 'send that', "
        "'go ahead and send'. There is a short window to undo afterwards."
    ),
    privacy=Privacy.LOCAL_ONLY,
    arg="query",
    confirm=True,
)
def send_email(query: str = "", **_) -> ToolResult:
    d = mail.pending()
    if d is None:
        return ToolResult("There's no draft to send.", Privacy.LOCAL_ONLY, ok=False)
    return ToolResult(mail.send(d.draft_id), Privacy.LOCAL_ONLY,
                      data={"to": d.to, "subject": d.subject})


@tool(
    name="undo_send",
    description=(
        "Stop an email that was just sent, within the undo window. Use for 'undo', "
        "'stop', 'cancel that', 'don't send it', 'wait no'."
    ),
    privacy=Privacy.LOCAL_ONLY,
    arg="query",
)
def undo_send(query: str = "", **_) -> ToolResult:
    d = mail.pending()
    if d is None:
        return ToolResult("Nothing to undo.", Privacy.LOCAL_ONLY, ok=False)
    return ToolResult(mail.undo(d.draft_id), Privacy.LOCAL_ONLY)
