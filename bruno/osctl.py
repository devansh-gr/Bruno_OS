"""Seam six: OS control.

Scripted actions and URL schemes only. No pixel-clicking: a voice assistant
almost never needs it, and coordinate output is exactly where a local model is
weakest. Where an app exposes a URL scheme, a script API, or an HTTP API, use
that; where it doesn't, we go without rather than guessing at screen positions.

The per-OS delta is deliberately tiny. Everything Bruno actually does reduces to
"open this URI", "run this app", or "send this media key", and URL schemes work
identically on macOS and Windows: `spotify:track:abc` opens Spotify on both.

The Windows half is written but UNTESTED; there is no Windows machine yet. Every
Windows path is marked so a failure there is read as "never verified" rather than
"regression".
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from bruno import config


class ActionRefused(RuntimeError):
    """A destructive action was attempted without confirmation."""


@dataclass
class ActionResult:
    ok: bool
    said: str          # what Bruno should say about it
    detail: str = ""


# Actions that change or destroy state need a spoken yes first. Play/pause/open
# do not: asking permission to unpause music is worse than useless.
DESTRUCTIVE = {"quit_app", "close_window", "delete"}


class MacControl:
    name = "macos"
    tested = True

    def _osa(self, script: str) -> subprocess.CompletedProcess:
        return subprocess.run(["osascript", "-e", script], capture_output=True,
                              text=True, timeout=15)

    def open_uri(self, uri: str) -> ActionResult:
        r = subprocess.run(["open", uri], capture_output=True, text=True, timeout=15)
        return ActionResult(r.returncode == 0,
                            "Opening it." if r.returncode == 0 else "That didn't open.",
                            r.stderr.strip())

    def launch_app(self, app: str) -> ActionResult:
        r = subprocess.run(["open", "-a", app], capture_output=True, text=True, timeout=20)
        return ActionResult(r.returncode == 0,
                            f"{app} is opening." if r.returncode == 0 else f"Couldn't find {app}.",
                            r.stderr.strip())

    def media(self, key: str) -> ActionResult:
        """playpause | next | previous. Key codes 16/17/18 are the media keys."""
        codes = {"playpause": 16, "next": 17, "previous": 18}
        if key not in codes:
            return ActionResult(False, f"I don't know the {key} control.")
        # System Events needs Accessibility permission; fall back to Music/Spotify
        # scripting if it's not granted.
        r = self._osa(f'tell application "System Events" to key code {codes[key]}')
        if r.returncode != 0:
            for app in ("Spotify", "Music"):
                verb = {"playpause": "playpause", "next": "next track",
                        "previous": "previous track"}[key]
                r2 = self._osa(f'tell application "{app}" to {verb}')
                if r2.returncode == 0:
                    return ActionResult(True, "", f"via {app}")
            return ActionResult(False, "I couldn't reach the media controls.", r.stderr.strip())
        return ActionResult(True, "", "via System Events")

    def quit_app(self, app: str) -> ActionResult:
        r = self._osa(f'tell application "{app}" to quit')
        return ActionResult(r.returncode == 0, f"Closed {app}.", r.stderr.strip())

    def notify(self, text: str) -> ActionResult:
        safe = text.replace('"', "'")
        r = self._osa(f'display notification "{safe}" with title "Bruno"')
        return ActionResult(r.returncode == 0, "", r.stderr.strip())


class WindowsControl:
    """UNTESTED: written from documentation, no Windows machine to verify against.
    Treat every failure here as unverified rather than as a regression."""

    name = "windows"
    tested = False

    def _ps(self, script: str) -> subprocess.CompletedProcess:
        return subprocess.run(["powershell", "-NoProfile", "-Command", script],
                              capture_output=True, text=True, timeout=20)

    def open_uri(self, uri: str) -> ActionResult:
        r = self._ps(f"Start-Process '{uri.replace(chr(39), chr(39) * 2)}'")
        return ActionResult(r.returncode == 0,
                            "Opening it." if r.returncode == 0 else "That didn't open.",
                            r.stderr.strip())

    def launch_app(self, app: str) -> ActionResult:
        r = self._ps(f"Start-Process '{app.replace(chr(39), chr(39) * 2)}'")
        return ActionResult(r.returncode == 0,
                            f"{app} is opening." if r.returncode == 0 else f"Couldn't find {app}.",
                            r.stderr.strip())

    def media(self, key: str) -> ActionResult:
        # Virtual key codes: 0xB3 play/pause, 0xB0 next, 0xB1 previous.
        codes = {"playpause": "0xB3", "next": "0xB0", "previous": "0xB1"}
        if key not in codes:
            return ActionResult(False, f"I don't know the {key} control.")
        r = self._ps(
            "$sig='[DllImport(\"user32.dll\")]public static extern void keybd_event"
            "(byte b,byte s,int f,int e);';"
            "$k=Add-Type -MemberDefinition $sig -Name K -Namespace W -PassThru;"
            f"$k::keybd_event({codes[key]},0,0,0); $k::keybd_event({codes[key]},0,2,0)"
        )
        return ActionResult(r.returncode == 0, "", r.stderr.strip())

    def quit_app(self, app: str) -> ActionResult:
        r = self._ps(f"Stop-Process -Name '{app}' -ErrorAction Stop")
        return ActionResult(r.returncode == 0, f"Closed {app}.", r.stderr.strip())

    def notify(self, text: str) -> ActionResult:
        # BurntToast is not installed by default; degrade to console rather than fail.
        r = self._ps(
            "if (Get-Module -ListAvailable -Name BurntToast) { Import-Module BurntToast; "
            f"New-BurntToastNotification -Text 'Bruno','{text.replace(chr(39), chr(39) * 2)}' }} "
            "else { exit 3 }"
        )
        if r.returncode == 3:
            return ActionResult(False, "", "BurntToast not installed")
        return ActionResult(r.returncode == 0, "", r.stderr.strip())


def get_control():
    if config.IS_MAC:
        return MacControl()
    if config.IS_WINDOWS:
        return WindowsControl()
    raise NotImplementedError("no OS control for this platform")


def guard(action: str, confirmed: bool) -> None:
    if action in DESTRUCTIVE and not confirmed:
        raise ActionRefused(f"{action} needs confirmation first")
