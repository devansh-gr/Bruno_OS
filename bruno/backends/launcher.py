"""Seam four: spawn the Claude Code CLI in a visible terminal, in a target directory.

Not wired into the voice loop yet: M1 only needs the seam to exist so the Windows
port stays additive.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from bruno import config


def _check(cwd: str) -> str:
    p = Path(cwd).expanduser().resolve()
    if not p.is_dir():
        raise NotADirectoryError(f"target directory does not exist: {p}")
    return str(p)


class MacLauncher:
    name = "applescript"

    def launch(self, cwd: str, prompt: str) -> None:
        cwd = _check(cwd)
        cmd = f"cd {shlex.quote(cwd)} && claude {shlex.quote(prompt)}"
        # Escape for the AppleScript string literal, not for the shell.
        script = (
            'tell application "Terminal"\n'
            f'  do script "{cmd.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"\n'
            "  activate\n"
            "end tell"
        )
        subprocess.run(["osascript", "-e", script], check=True)


class WindowsLauncher:
    name = "powershell"

    def launch(self, cwd: str, prompt: str) -> None:
        cwd = _check(cwd)
        # Untested: no Windows machine yet. Written so the port is additive.
        inner = f"Set-Location -LiteralPath '{cwd}'; claude '{prompt.replace(chr(39), chr(39) * 2)}'"
        subprocess.Popen(
            ["powershell", "-NoExit", "-NoProfile", "-Command", inner],
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )


def get_launcher():
    if config.IS_MAC:
        return MacLauncher()
    if config.IS_WINDOWS:
        return WindowsLauncher()
    raise NotImplementedError(f"no launcher for this platform")
