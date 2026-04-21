"""Cross-platform app launcher and focus helpers.

Two operations are exposed:

    open_app(name)       - launch an app by name, or no-op if it's already
                           running.
    activate_app(name)   - bring an already-running app to the foreground.

The implementations branch on `sys.platform`. macOS uses `open -a` plus
AppleScript; Windows uses `start` plus a tiny PowerShell helper to focus
the window. We deliberately keep this file very small — anything more
complex than "launch + focus" belongs elsewhere.
"""

from __future__ import annotations

import subprocess
import sys


def _mac_open_app(app_name: str) -> None:
    """`open -a "<name>"` on macOS. Returns immediately; the app may still
    be launching when this returns, so callers usually sleep briefly after."""
    subprocess.run(["open", "-a", app_name], capture_output=True)


def _mac_activate_app(app_name: str) -> None:
    """Bring an already-launched macOS app to the front via AppleScript."""
    subprocess.run(
        ["osascript", "-e", f'tell application "{app_name}" to activate'],
        capture_output=True,
    )


def _windows_open_app(app_name: str) -> None:
    """Launch an app on Windows.

    Approach:
        - `start "" "<app>"` works for things on PATH or registered apps
          (e.g. "chrome", "notepad", "explorer").
        - We pass shell=True because `start` is a cmd.exe builtin, not a
          standalone executable.
        - The empty string "" is the (ignored) window title `start` requires
          when the first quoted arg is the program path.
    """
    subprocess.Popen(
        f'start "" "{app_name}"',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _windows_activate_app(app_name: str) -> None:
    """Bring a Windows app's main window to the foreground.

    We use a small PowerShell snippet because Python's stdlib doesn't expose
    SetForegroundWindow, and we don't want to require pywin32 for this single
    call (pywinauto is already an optional install).
    """
    ps = (
        f'$p = Get-Process | Where-Object {{ $_.MainWindowTitle -like "*{app_name}*" }} '
        f'| Select-Object -First 1; '
        f'if ($p) {{ '
        f'  Add-Type -TypeDefinition "using System; using System.Runtime.InteropServices; '
        f'public class W {{ [DllImport(\\"user32.dll\\")] '
        f'public static extern bool SetForegroundWindow(IntPtr h); }}"; '
        f'  [W]::SetForegroundWindow($p.MainWindowHandle); '
        f'}}'
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API (platform-dispatching wrappers)
# ─────────────────────────────────────────────────────────────────────────────

def open_app(app_name: str) -> None:
    """Launch the named app on whatever OS we're on.
    No-op (silently) on platforms we don't support yet."""
    if sys.platform == "darwin":
        _mac_open_app(app_name)
    elif sys.platform == "win32":
        _windows_open_app(app_name)


def activate_app(app_name: str) -> None:
    """Bring the named app to the foreground, focusing its main window."""
    if sys.platform == "darwin":
        _mac_activate_app(app_name)
    elif sys.platform == "win32":
        _windows_activate_app(app_name)
