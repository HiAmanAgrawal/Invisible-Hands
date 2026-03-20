"""
Device controller — handles all interaction with the Mac desktop.

This module provides functions to:
  - Take screenshots (using Quartz CoreGraphics)
  - Open and focus apps (using macOS `open -a` command)
  - Send keyboard input (using AppleScript via osascript — more reliable than pyautogui)
  - Control the mouse (using pyautogui — clicks, scrolls, etc.)
  - Play system sounds (using macOS `afplay` command)

WHY AppleScript instead of pyautogui for keyboard?
  pyautogui uses CGEventCreateKeyboardEvent which requires Accessibility permission.
  AppleScript talks to "System Events" which is a separate, more reliable path.
  This is the same approach used in test.py (send_keystroke function).
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import time

import pyautogui
from PIL import Image

# FAILSAFE: move mouse to top-left corner to abort any pyautogui action
pyautogui.FAILSAFE = True
# Small pause between pyautogui actions to avoid racing
pyautogui.PAUSE = 0.3


# ────────────────────────────────────────────────────────────
# AppleScript key code mappings
# ────────────────────────────────────────────────────────────
# AppleScript uses numeric key codes for special keys (Enter, Escape, etc.)
# Normal letter keys can be sent as strings via `keystroke "a"`
# but special keys need `key code 36` style syntax.

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

# Maps common modifier names to AppleScript modifier syntax
# e.g. "command" → "command down" (used in `using {command down}`)
_APPLESCRIPT_MODIFIERS = {
    "command": "command down", "cmd": "command down",
    "shift": "shift down",
    "option": "option down", "alt": "option down", "opt": "option down",
    "control": "control down", "ctrl": "control down",
}


# ────────────────────────────────────────────────────────────
# Screen capture
# ────────────────────────────────────────────────────────────

def get_screen_size():
    """Return the logical screen size (width, height) in points.
    On Retina displays, the actual pixel count is 2x this, but
    pyautogui coordinates use logical points, so we use these."""
    return pyautogui.size()


def take_screenshot() -> Image.Image:
    """Capture the full screen and return as a PIL Image.

    On macOS:
      1. Tries Quartz CoreGraphics API (fastest, no temp file)
      2. Falls back to `screencapture` CLI tool
      3. Resizes to logical resolution so coordinates match pyautogui

    On other platforms: uses pyautogui.screenshot() directly.
    """
    if sys.platform == "darwin":
        img = _screenshot_macos()

        # Retina displays capture at 2x resolution (e.g. 3456x2234)
        # but pyautogui works in logical points (e.g. 1728x1117)
        # so we resize the screenshot to match pyautogui coordinates
        logical_w, logical_h = get_screen_size()
        if img.size != (logical_w, logical_h):
            img = img.resize((logical_w, logical_h), Image.LANCZOS)
        return img

    return pyautogui.screenshot()


def _screenshot_macos() -> Image.Image:
    """Try multiple capture methods on macOS.
    Raises PermissionError with fix instructions if all methods fail."""

    # Method 1: Quartz CoreGraphics — fastest, no temp file needed
    try:
        return _capture_quartz()
    except Exception:
        pass

    # Method 2: screencapture CLI — works as fallback
    try:
        return _capture_screencapture()
    except Exception:
        pass

    # Both failed — most likely Screen Recording permission is missing
    raise PermissionError(
        "Screen capture FAILED — Screen Recording permission not granted.\n"
        "  Fix: System Settings → Privacy & Security → Screen Recording\n"
        "        → enable your terminal app, then RESTART it."
    )


def _capture_quartz() -> Image.Image:
    """Capture screen using Quartz CoreGraphics API directly.
    This avoids creating temp files and is the fastest method."""
    import Quartz.CoreGraphics as CG

    # CGWindowListCreateImage captures all visible windows into one image
    image_ref = CG.CGWindowListCreateImage(
        CG.CGRectInfinite,                    # capture full screen
        CG.kCGWindowListOptionOnScreenOnly,   # only visible windows
        CG.kCGNullWindowID,                   # all windows (not a specific one)
        CG.kCGWindowImageDefault,             # default rendering
    )

    if image_ref is None:
        raise RuntimeError("CGWindowListCreateImage returned None")

    width = CG.CGImageGetWidth(image_ref)
    height = CG.CGImageGetHeight(image_ref)

    # A 1x1 image means permission was denied (macOS returns a blank pixel)
    if width <= 1 or height <= 1:
        raise RuntimeError("Capture returned 1x1 — permission denied")

    # Convert the raw pixel data into a PIL Image
    # macOS uses BGRA byte order, so we convert from that
    bytes_per_row = CG.CGImageGetBytesPerRow(image_ref)
    data_provider = CG.CGImageGetDataProvider(image_ref)
    raw_data = CG.CGDataProviderCopyData(data_provider)

    return Image.frombuffer(
        "RGBA", (width, height), raw_data, "raw", "BGRA", bytes_per_row, 1
    )


def _capture_screencapture() -> Image.Image:
    """Capture screen using macOS `screencapture` CLI tool.
    The -x flag suppresses the screenshot sound."""
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        subprocess.run(
            ["screencapture", "-x", tmp_path],
            capture_output=True, timeout=10,
        )
        # Verify the file isn't empty/corrupt (should be way more than 100 bytes)
        if os.path.getsize(tmp_path) < 100:
            raise RuntimeError("screencapture produced empty file")
        img = Image.open(tmp_path)
        img.load()  # force PIL to read the file before we delete it
        return img
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def screenshot_to_bytes(image: Image.Image) -> bytes:
    """Convert a PIL Image to raw PNG bytes.
    The ollama library accepts raw bytes for image input
    (base64 strings cause a 'file name too long' error)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ────────────────────────────────────────────────────────────
# App control (macOS)
# ────────────────────────────────────────────────────────────

def open_app(app_name: str):
    """Launch an app using macOS `open -a` command.
    This is more reliable than Spotlight (Command+Space → type → Enter).
    Same approach as test.py: subprocess.run(["open", "-a", "Spotify"])"""
    subprocess.run(["open", "-a", app_name], capture_output=True)


def activate_app(app_name: str):
    """Bring an app to the foreground using AppleScript.
    Same approach as test.py: 'tell application "Spotify" to activate'
    This ensures the app window is focused and ready for keyboard input."""
    subprocess.run([
        "osascript", "-e",
        f'tell application "{app_name}" to activate'
    ], capture_output=True)


# ────────────────────────────────────────────────────────────
# Keyboard control via AppleScript
# ────────────────────────────────────────────────────────────
# Why AppleScript? Because pyautogui.hotkey() requires macOS Accessibility
# permission for the terminal app, and silently fails without it.
# AppleScript goes through "System Events" which is a separate, more
# reliable permission path. This is the same approach as test.py's
# send_keystroke() function.

def _run_applescript(script: str):
    """Run an AppleScript command via osascript.
    All keyboard functions below call this helper."""
    subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)


def hotkey(*keys: str):
    """Press a key combination via AppleScript.

    Examples:
      hotkey("command", "space")  → Cmd+Space (Spotlight)
      hotkey("command", "l")     → Cmd+L (focus address bar)
      hotkey("command", "k")     → Cmd+K (Spotify search)

    How it works:
      1. Separates modifier keys (command, shift, etc.) from the main key
      2. If the main key is a special key (Enter, Escape), uses `key code`
      3. Otherwise uses `keystroke "k" using {command down}`

    On non-macOS: falls back to pyautogui.hotkey().
    """
    if sys.platform != "darwin":
        pyautogui.hotkey(*keys)
        return

    keys = [k.lower().strip() for k in keys]

    # Separate modifiers (command, shift, etc.) from the main key
    modifiers = []
    main_key = None
    for k in keys:
        if k in _APPLESCRIPT_MODIFIERS:
            modifiers.append(_APPLESCRIPT_MODIFIERS[k])
        else:
            main_key = k

    if main_key is None:
        return

    # Build the AppleScript modifier string: {command down, shift down}
    modifier_str = ", ".join(modifiers) if modifiers else ""

    # Special keys (Enter, Escape, Tab) need `key code N`
    # Normal keys (letters, numbers) use `keystroke "k"`
    if main_key in _APPLESCRIPT_KEY_CODES:
        code = _APPLESCRIPT_KEY_CODES[main_key]
        if modifier_str:
            script = f'tell application "System Events" to key code {code} using {{{modifier_str}}}'
        else:
            script = f'tell application "System Events" to key code {code}'
    else:
        if modifier_str:
            script = f'tell application "System Events" to keystroke "{main_key}" using {{{modifier_str}}}'
        else:
            script = f'tell application "System Events" to keystroke "{main_key}"'

    _run_applescript(script)


def press_key(key: str):
    """Press a single key via AppleScript (no modifier).

    Examples:
      press_key("enter")   → press Enter key
      press_key("escape")  → press Escape key
      press_key("tab")     → press Tab key

    Uses key codes for special keys, keystroke for letters.
    """
    if sys.platform != "darwin":
        pyautogui.press(key)
        return

    k = key.lower().strip()
    if k in _APPLESCRIPT_KEY_CODES:
        code = _APPLESCRIPT_KEY_CODES[k]
        _run_applescript(f'tell application "System Events" to key code {code}')
    else:
        _run_applescript(f'tell application "System Events" to keystroke "{k}"')


def type_text(text: str):
    """Type a string of text via AppleScript keystroke.

    Uses AppleScript's `keystroke` command which types the whole string.
    Special characters are escaped for AppleScript string syntax.

    On non-macOS: falls back to pyautogui.write() with a small interval.
    """
    if sys.platform != "darwin":
        pyautogui.write(text, interval=0.03)
        return

    # Escape backslashes and double-quotes for AppleScript string
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    _run_applescript(f'tell application "System Events" to keystroke "{escaped}"')
    time.sleep(0.3)


# ────────────────────────────────────────────────────────────
# Mouse control (via pyautogui)
# ────────────────────────────────────────────────────────────
# Mouse events still use pyautogui — AppleScript can't easily
# click at arbitrary pixel coordinates.

def click(x: int, y: int):
    """Click at the given (x, y) pixel coordinates."""
    pyautogui.click(x, y)


def double_click(x: int, y: int):
    """Double-click at the given (x, y) pixel coordinates."""
    pyautogui.doubleClick(x, y)


def right_click(x: int, y: int):
    """Right-click at the given (x, y) pixel coordinates."""
    pyautogui.rightClick(x, y)


def scroll_screen(direction: str, amount: int = 3):
    """Scroll up or down. Positive = up, negative = down."""
    clicks = amount if direction == "up" else -amount
    pyautogui.scroll(clicks)


def move_to(x: int, y: int, duration: float = 0.3):
    """Move the mouse cursor to (x, y) over `duration` seconds."""
    pyautogui.moveTo(x, y, duration=duration)


# ────────────────────────────────────────────────────────────
# Sound feedback
# ────────────────────────────────────────────────────────────
# Uses macOS built-in system sounds via `afplay` command.
# These are short .aiff files in /System/Library/Sounds/

def play_sound(sound_name: str):
    """Play a macOS system sound in the background (non-blocking).

    Available sounds: Tink, Glass, Hero, Pop, Ping, Purr, etc.
    Full list: ls /System/Library/Sounds/
    """
    if sys.platform != "darwin":
        return
    sound_path = f"/System/Library/Sounds/{sound_name}.aiff"
    if os.path.exists(sound_path):
        # Run afplay in background so it doesn't block the agent
        subprocess.Popen(
            ["afplay", sound_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def play_step_sound():
    """Short subtle sound when a step completes (Tink)."""
    play_sound("Tink")


def play_done_sound():
    """Celebratory sound when the entire flow completes (Glass)."""
    play_sound("Glass")


# ────────────────────────────────────────────────────────────
# Pre-flight checks
# ────────────────────────────────────────────────────────────

def preflight_check() -> list[str]:
    """Run pre-flight checks before the agent starts.

    Validates:
      1. Screen capture works (Screen Recording permission)
      2. AppleScript can talk to System Events (Automation permission)
      3. Ollama is running and has the required models

    Returns a list of error strings. Empty list = all good.
    """
    errors = []

    # 1. Screen capture — need Screen Recording permission
    try:
        img = take_screenshot()
        w, h = img.size
        if w <= 1 or h <= 1:
            errors.append(
                "Screen capture returned a blank image — "
                "Screen Recording permission likely missing."
            )
    except PermissionError as e:
        errors.append(str(e).strip())
    except Exception as e:
        errors.append(f"Screen capture failed: {e}")

    # 2. AppleScript System Events — need Automation permission
    # We test by asking System Events for the name of the first process
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to return name of first process'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            errors.append(
                "AppleScript System Events access failed.\n"
                "  You may need to allow your terminal in:\n"
                "  System Settings → Privacy & Security → Automation"
            )
    except Exception as e:
        errors.append(f"AppleScript check failed: {e}")

    # 3. Ollama connectivity + required models
    try:
        import ollama as _ollama_mod
        models = _ollama_mod.list()
        model_names = [m.model for m in models.models]

        if not any("llama3.2-vision" in n for n in model_names):
            errors.append(
                f"Model 'llama3.2-vision' not found. Available: {model_names}\n"
                f"  Run: ollama pull llama3.2-vision"
            )
        if not any("llama3" in n for n in model_names):
            errors.append(
                f"Model 'llama3' not found. Available: {model_names}\n"
                f"  Run: ollama pull llama3"
            )
    except Exception as e:
        errors.append(
            f"Cannot connect to Ollama: {e}\n"
            f"  Make sure Ollama is running."
        )

    return errors
