"""Cross-platform mouse + keyboard control.

Mouse:
    pyautogui works the same on Windows and Mac, so the mouse helpers are
    one-liners that just delegate to it. We use a "move-then-click" pattern
    everywhere to avoid race conditions with hover-state UIs.

Keyboard:
    On macOS we route through AppleScript (osascript). pyautogui's keyboard
    path needs Accessibility permission for the terminal app, and silently
    fails when it isn't granted, which is a really painful debugging
    experience. AppleScript talks to "System Events" through a separate
    permission path (Automation) which is more reliable in practice.

    On Windows (and any non-Mac), we fall back to pyautogui — it works fine
    out of the box without extra permissions.

Sound:
    Tiny audio cues after each step / at the end of a run. macOS uses afplay
    on the bundled system sounds; Windows uses winsound (stdlib).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pyautogui


# ─────────────────────────────────────────────────────────────────────────────
# AppleScript key tables (macOS only)
# ─────────────────────────────────────────────────────────────────────────────
# Some keys (Enter, Escape, arrows...) can't be sent via `keystroke` and need
# AppleScript's `key code N` syntax with a numeric ID. These tables map the
# friendly names we use in plans/actions ("enter", "escape") to those IDs.

_APPLESCRIPT_KEY_CODES = {
    "enter": 36, "return": 36,
    "escape": 53, "esc": 53,
    "tab": 48,
    "space": 49,
    "delete": 51, "backspace": 51,
    "up": 126, "down": 125, "left": 123, "right": 124,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118,
    "f5": 96, "f6": 97, "f7": 98, "f8": 100,
}

# Modifier-key alias table. We accept several common spellings of each modifier
# (cmd / command / ⌘ are all "command down" to AppleScript).
_APPLESCRIPT_MODIFIERS = {
    "command": "command down", "cmd": "command down",
    "shift": "shift down",
    "option": "option down", "alt": "option down", "opt": "option down",
    "control": "control down", "ctrl": "control down",
}


def _is_mac() -> bool:
    """True if we're running on macOS — used to pick the keyboard backend."""
    return sys.platform == "darwin"


def _run_applescript(script: str) -> None:
    """Execute a single line of AppleScript via osascript.
    Errors are swallowed so a flaky permission state can't crash the agent."""
    subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# Mouse
# ─────────────────────────────────────────────────────────────────────────────

def click(x: int, y: int) -> None:
    """Move the cursor to (x, y) over a brief animation, then left-click.

    Why move-then-click instead of pyautogui.click(x, y)?
        Many web UIs render hover styles or load lazy elements when the
        cursor enters their bounding box. Clicking instantly at a fresh
        coordinate sometimes misses the real target, especially on YouTube,
        Gmail, etc. The 150ms animation gives the UI time to react.
    """
    pyautogui.moveTo(x, y, duration=0.15)
    time.sleep(0.1)
    pyautogui.click()


def double_click(x: int, y: int) -> None:
    """Move-then-double-click. See `click` for the rationale."""
    pyautogui.moveTo(x, y, duration=0.15)
    time.sleep(0.1)
    pyautogui.doubleClick()


def right_click(x: int, y: int) -> None:
    """Move-then-right-click. See `click` for the rationale."""
    pyautogui.moveTo(x, y, duration=0.15)
    time.sleep(0.1)
    pyautogui.rightClick()


def move_to(x: int, y: int, duration: float = 0.3) -> None:
    """Move the cursor to (x, y) with a short animation. No click."""
    pyautogui.moveTo(x, y, duration=duration)


def scroll_screen(direction: str, amount: int = 3) -> None:
    """Scroll the active window. `direction` is 'up' or 'down'; `amount` is
    in pyautogui's logical click units (each click ~one notch on the wheel)."""
    clicks = amount if direction == "up" else -amount
    pyautogui.scroll(clicks)


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard
# ─────────────────────────────────────────────────────────────────────────────

def hotkey(*keys: str) -> None:
    """Press a key combination such as `hotkey("command", "l")`.

    On macOS:
        Builds an AppleScript `keystroke "<k>" using {<modifiers>}` line and
        sends it through System Events. Special keys (Enter, Tab, etc.) use
        `key code N` instead.
    Elsewhere:
        Delegates to pyautogui.hotkey, which works on Windows and Linux.
    """
    if not _is_mac():
        pyautogui.hotkey(*keys)
        return

    cleaned = [k.lower().strip() for k in keys]
    modifiers: list[str] = []
    main_key: str | None = None
    for k in cleaned:
        if k in _APPLESCRIPT_MODIFIERS:
            modifiers.append(_APPLESCRIPT_MODIFIERS[k])
        else:
            main_key = k

    if main_key is None:
        return

    modifier_str = ", ".join(modifiers)
    if main_key in _APPLESCRIPT_KEY_CODES:
        code = _APPLESCRIPT_KEY_CODES[main_key]
        if modifier_str:
            script = (f'tell application "System Events" to '
                      f'key code {code} using {{{modifier_str}}}')
        else:
            script = f'tell application "System Events" to key code {code}'
    else:
        if modifier_str:
            script = (f'tell application "System Events" to '
                      f'keystroke "{main_key}" using {{{modifier_str}}}')
        else:
            script = f'tell application "System Events" to keystroke "{main_key}"'

    _run_applescript(script)


def press_key(key: str) -> None:
    """Press a single key with no modifier (Enter, Escape, Tab, "/", etc.).

    Uses the AppleScript key-code table on macOS for special keys, and falls
    back to a `keystroke "<k>"` for normal printable characters."""
    if not _is_mac():
        pyautogui.press(key)
        return

    k = key.lower().strip()
    if k in _APPLESCRIPT_KEY_CODES:
        code = _APPLESCRIPT_KEY_CODES[k]
        _run_applescript(f'tell application "System Events" to key code {code}')
    else:
        _run_applescript(f'tell application "System Events" to keystroke "{k}"')


def type_text(text: str) -> None:
    """Type the given string of text at the current focus.

    macOS uses AppleScript's `keystroke` with the text as a single string
    (after escaping backslashes and double quotes for AppleScript's quoting
    rules). Other platforms use pyautogui.write with a small inter-key delay
    so apps with throttled input handlers (e.g. some text editors) don't drop
    characters.
    """
    if not _is_mac():
        pyautogui.write(text, interval=0.03)
        return

    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    _run_applescript(f'tell application "System Events" to keystroke "{escaped}"')
    # Brief settle so the keystrokes finish before the next action runs.
    time.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Audio feedback
# ─────────────────────────────────────────────────────────────────────────────

def _play_mac_sound(sound_name: str) -> None:
    """Play one of macOS's bundled .aiff system sounds via afplay (non-blocking)."""
    sound_path = f"/System/Library/Sounds/{sound_name}.aiff"
    if not os.path.exists(sound_path):
        return
    subprocess.Popen(
        ["afplay", sound_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _play_windows_sound(alias: str) -> None:
    """Play a Windows system sound alias via the stdlib winsound module.

    Common aliases: 'SystemAsterisk', 'SystemExclamation', 'SystemHand',
    'SystemQuestion', 'SystemDefault'. We don't pick anything obnoxious."""
    try:
        import winsound  # type: ignore

        winsound.PlaySound(alias, winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception:
        pass


def play_step_sound() -> None:
    """Short sound after each step completes."""
    if _is_mac():
        _play_mac_sound("Tink")
    elif sys.platform == "win32":
        _play_windows_sound("SystemDefault")


def play_done_sound() -> None:
    """Slightly more celebratory sound when the entire run completes."""
    if _is_mac():
        _play_mac_sound("Glass")
    elif sys.platform == "win32":
        _play_windows_sound("SystemAsterisk")
