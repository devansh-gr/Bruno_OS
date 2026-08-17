"""The briefing tools: what's happening, tell me more, pull that up.

Transcripts are fetched lazily and never during the briefing itself. Pulling one
costs ~3 seconds; doing it for a dozen episodes up front would make "what's
happening" a forty-second wait for material he mostly won't ask about. So the
briefing reads titles, and the deep fetch happens only for the one item he picks.

LOCAL_ONLY throughout. The briefing is ranked against his projects and his
interests, which makes the *selection* personal even though every source is
public — a cloud model seeing what Bruno chose to surface learns what he's
working on.
"""

from __future__ import annotations

from bruno import briefing as _briefing
from bruno.tools.base import Privacy, ToolResult, tool

# The last briefing, so a follow-up has something to match a description against.
_LAST: _briefing.Briefing | None = None


def last() -> _briefing.Briefing | None:
    return _LAST


@tool(
    name="briefing",
    description=(
        "Gather what is happening right now across news, markets, podcasts and X, "
        "ranked by relevance to Devansh, and read it as a list. Use for 'what's "
        "going on', 'catch me up', 'anything interesting today', 'what's new in AI', "
        "'give me the rundown'. Every item is fetched, never recalled."
    ),
    privacy=Privacy.LOCAL_ONLY,
    arg="query",
)
def build_briefing(query: str = "", **_) -> ToolResult:
    global _LAST
    want_pods = "podcast" in (query or "").lower() or not query
    try:
        _LAST = _briefing.build(limit=8, include_podcasts=want_pods)
    except Exception as e:  # noqa: BLE001 — one dead source shouldn't kill a briefing
        return ToolResult(f"I couldn't pull the briefing together: {e}",
                          Privacy.LOCAL_ONLY, ok=False)
    if not _LAST.items:
        return ToolResult("Nothing came back from any source.",
                          Privacy.LOCAL_ONLY, ok=False)
    return ToolResult(_briefing.speak(_LAST), Privacy.LOCAL_ONLY,
                      data=[{"title": i.title, "source": i.source, "url": i.url}
                            for i in _LAST.items])


@tool(
    name="tell_me_more",
    description=(
        "Go deeper on something from the briefing Devansh just heard, named by what "
        "it was about rather than a number — 'tell me more about the Nvidia one', "
        "'what was that Anthropic thing', 'expand on the podcast one'. For a podcast "
        "this pulls the actual transcript and reads back only the parts relevant to him."
    ),
    privacy=Privacy.LOCAL_ONLY,
    arg="query",
)
def tell_me_more(query: str = "", **_) -> ToolResult:
    if _LAST is None or not _LAST.items:
        return ToolResult("I haven't given you a briefing yet.",
                          Privacy.LOCAL_ONLY, ok=False)
    item = _LAST.find(query)
    if item is None:
        return ToolResult(f"I can't tell which one you mean by {query}.",
                          Privacy.LOCAL_ONLY, ok=False)

    if item.kind == "podcast":
        from bruno import transcript
        show = item.source.split(":", 1)[-1]
        got = transcript.fetch(item.title, show, interests=_briefing._interests())
        if got and got["passages"]:
            body = " ... ".join(p[:500] for p in got["passages"][:2])
            return ToolResult(
                f"From the episode itself: {body}", Privacy.LOCAL_ONLY,
                data={"url": got["source"], "chars": got["chars"],
                      "passages": got["passages"]})
        if got:
            return ToolResult(
                f"I got the transcript but nothing in it touched your areas. "
                f"The episode is about: {item.text[:280]}",
                Privacy.LOCAL_ONLY, data={"url": got["source"]})

    return ToolResult(item.text[:600] or item.title, Privacy.LOCAL_ONLY,
                      data={"url": item.url, "source": item.source})


@tool(
    name="pull_up",
    description=(
        "Open something from the briefing in the browser so Devansh can read it "
        "himself — 'pull up that article', 'open the Nvidia one', 'show me that "
        "podcast'. Refers to items by what they were about, not by number."
    ),
    privacy=Privacy.LOCAL_ONLY,
    arg="query",
)
def pull_up(query: str = "", **_) -> ToolResult:
    if _LAST is None or not _LAST.items:
        return ToolResult("I haven't given you a briefing yet.",
                          Privacy.LOCAL_ONLY, ok=False)
    item = _LAST.find(query)
    if item is None:
        return ToolResult(f"I can't tell which one you mean by {query}.",
                          Privacy.LOCAL_ONLY, ok=False)
    if not item.url:
        return ToolResult(f"I don't have a link for {item.title[:60]}.",
                          Privacy.LOCAL_ONLY, ok=False)

    from bruno.osctl import get_control
    res = get_control().open_uri(item.url)
    where = item.source.split(":")[-1]
    return ToolResult(
        f"Opening the {where} one." if res.ok else "I couldn't open that.",
        Privacy.LOCAL_ONLY, ok=res.ok, data={"url": item.url})
