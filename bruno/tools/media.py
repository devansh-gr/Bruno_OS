"""Media control: play a named song, and transport controls.

Two deliberately separate paths, because they have very different costs.

`play_song` needs to *find* a track, which means the Spotify Web API: but only
the search half, which runs on free client-credentials auth with no user login and
no Premium. Playback then happens by handing the resulting `spotify:track:...` URI
to the OS, and URL schemes behave identically on macOS and Windows. So the
expensive part (OAuth, Premium, the Web Playback SDK) is avoided entirely.

`media_control` is play/pause/skip and touches no network at all.
"""

from __future__ import annotations

import os
import time

import requests

from bruno.osctl import get_control
from bruno.tools.base import Privacy, ToolResult, tool

_TOKEN: dict = {"value": None, "expires": 0.0}


def _client_token() -> str | None:
    """Client-credentials token: search only, no user account, no Premium.
    Needs SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET from a free developer app."""
    cid, secret = os.environ.get("SPOTIFY_CLIENT_ID"), os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not (cid and secret):
        return None
    if _TOKEN["value"] and time.time() < _TOKEN["expires"] - 60:
        return _TOKEN["value"]
    try:
        r = requests.post("https://accounts.spotify.com/api/token",
                          data={"grant_type": "client_credentials"},
                          auth=(cid, secret), timeout=12)
        r.raise_for_status()
        j = r.json()
    except (requests.RequestException, ValueError):
        return None
    _TOKEN["value"] = j.get("access_token")
    _TOKEN["expires"] = time.time() + j.get("expires_in", 3600)
    return _TOKEN["value"]


@tool(
    name="play_song",
    description=(
        "Play AUDIO on Spotify: a specific song, album, or artist by name. Use for "
        "'play <song>', 'put on <artist>', 'play that <band> album'. This is music, "
        "not video: if the request is about listening, it is this tool."
    ),
    privacy=Privacy.CLOUD_OK,
    arg="query",
)
def play_song(query: str, **_) -> ToolResult:
    q = (query or "").strip()
    if not q:
        return ToolResult("What do you want me to play?", Privacy.CLOUD_OK, ok=False)

    token = _client_token()
    if not token:
        # No credentials: fall back to opening a search in the app rather than failing.
        get_control().open_uri(f"spotify:search:{requests.utils.quote(q)}")
        return ToolResult(
            f"I opened a search for {q}. I need Spotify API keys to play it directly.",
            Privacy.CLOUD_OK, ok=False,
        )

    try:
        r = requests.get("https://api.spotify.com/v1/search",
                         headers={"Authorization": f"Bearer {token}"},
                         params={"q": q, "type": "track", "limit": 1}, timeout=12)
        r.raise_for_status()
        items = r.json().get("tracks", {}).get("items", [])
    except (requests.RequestException, ValueError, KeyError) as e:
        return ToolResult(f"Spotify search failed: {e}", Privacy.CLOUD_OK, ok=False)

    if not items:
        return ToolResult(f"Couldn't find {q} on Spotify.", Privacy.CLOUD_OK, ok=False)

    track = items[0]
    who = ", ".join(a["name"] for a in track.get("artists", []))
    res = get_control().open_uri(track["uri"])
    if not res.ok:
        return ToolResult(f"Found {track['name']} but couldn't hand it to Spotify.",
                          Privacy.CLOUD_OK, ok=False, data=res.detail)
    return ToolResult(f"Playing {track['name']}{f' by {who}' if who else ''}.",
                      Privacy.CLOUD_OK,
                      data={"uri": track["uri"], "name": track["name"], "artists": who})


@tool(
    name="media_control",
    description=(
        "Pause, resume, or skip whatever is currently playing. Use for 'pause', "
        "'play', 'resume', 'skip', 'next track', 'go back'. Touches no network."
    ),
    privacy=Privacy.LOCAL_ONLY,
    arg="query",
)
def media_control(query: str = "playpause", **_) -> ToolResult:
    q = (query or "").lower()
    if any(w in q for w in ("skip", "next", "forward")):
        key = "next"
    elif any(w in q for w in ("back", "previous", "last")):
        key = "previous"
    else:
        key = "playpause"

    res = get_control().media(key)
    if not res.ok:
        return ToolResult("I couldn't reach the media controls.", Privacy.LOCAL_ONLY,
                          ok=False, data=res.detail)
    # Deliberately near-silent: narrating "pausing the music" over music you just
    # paused is noise. The action is its own feedback.
    return ToolResult({"next": "Skipped.", "previous": "Went back."}.get(key, ""),
                      Privacy.LOCAL_ONLY, data=key)
