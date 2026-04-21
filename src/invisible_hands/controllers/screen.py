"""Screen capture, screen size, the coordinate-grid overlay, and start-up
permission checks.

This module is intentionally cross-platform: it uses the `mss` library as the
primary capture method (works on Mac, Windows, Linux) with a pyautogui fallback,
and it never hardcodes paths or commands that only exist on macOS.

The grid overlay (`annotate_screenshot_with_axes`) is a critical helper for
the vision-model clicker — small VLMs are bad at guessing raw pixel values, so
we draw labelled axes along the top and left edges. The model just reads the
nearest number off the grid instead of doing free-form pixel arithmetic.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import pyautogui
from PIL import Image, ImageDraw

from invisible_hands import config


# Move the cursor to the top-left corner to abort any in-flight pyautogui call.
# This is one of pyautogui's built-in safety features and is the user's
# emergency stop.
pyautogui.FAILSAFE = True

# Small delay between pyautogui actions to avoid race conditions on slow UIs.
pyautogui.PAUSE = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Screen size
# ─────────────────────────────────────────────────────────────────────────────

def get_screen_size() -> tuple[int, int]:
    """Return the LOGICAL screen size in (width, height).

    On Retina/HiDPI displays the physical pixel count is 2x or 3x the logical
    point count, but the OS-level mouse APIs all use logical points. We always
    work in logical points so coordinates come out clickable.
    """
    return tuple(pyautogui.size())


# ─────────────────────────────────────────────────────────────────────────────
# Screenshot capture
# ─────────────────────────────────────────────────────────────────────────────

def take_screenshot() -> Image.Image:
    """Capture the full primary display and return as a PIL Image.

    Capture method preference, in order:
        1. mss              - cross-platform, no temp files, very fast
        2. pyautogui        - cross-platform fallback (uses mss internally on
                              modern versions, but works without it too)
        3. screencapture    - macOS-only CLI fallback for permission edge cases

    The image is always resized to LOGICAL screen coordinates so click points
    match what the mouse APIs expect, regardless of pixel density.
    """
    img = _capture_with_mss() or _capture_with_pyautogui() or _capture_with_screencapture()
    if img is None:
        raise PermissionError(
            "Screen capture FAILED.\n"
            "  macOS: System Settings -> Privacy & Security -> Screen Recording,\n"
            "         enable your terminal app, then RESTART it.\n"
            "  Windows: usually no permission setup is needed; check that no\n"
            "         security tool is blocking screen capture for Python."
        )

    logical_w, logical_h = get_screen_size()
    if img.size != (logical_w, logical_h):
        img = img.resize((logical_w, logical_h), Image.LANCZOS)
    return img


def _capture_with_mss() -> Image.Image | None:
    """Use the `mss` library to grab the primary monitor.
    Returns None if mss isn't installed or the grab fails."""
    try:
        import mss

        with mss.mss() as sct:
            # monitors[0] is "all monitors combined"; monitors[1] is the
            # primary display. We want just the primary so coordinates align
            # with pyautogui.size().
            monitor = sct.monitors[1]
            raw = sct.grab(monitor)
            return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    except Exception:
        return None


def _capture_with_pyautogui() -> Image.Image | None:
    """Fall back to pyautogui's built-in screenshot."""
    try:
        return pyautogui.screenshot()
    except Exception:
        return None


def _capture_with_screencapture() -> Image.Image | None:
    """macOS last-ditch fallback using the `screencapture` CLI tool."""
    if sys.platform != "darwin":
        return None

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        subprocess.run(
            ["screencapture", "-x", tmp_path],
            capture_output=True, timeout=10,
        )
        if os.path.getsize(tmp_path) < 100:
            return None
        img = Image.open(tmp_path)
        img.load()
        return img
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def screenshot_to_bytes(image: Image.Image) -> bytes:
    """Encode a PIL Image as raw PNG bytes (used when sending to the LLM)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate grid overlay (helps the vision model read pixel positions)
# ─────────────────────────────────────────────────────────────────────────────

def _get_axis_font(size: int, bold: bool = False):
    """Load a TrueType font for the axis labels.

    Tries several common system font paths (Mac, Windows, Linux) in order,
    falling back to PIL's bitmap default if nothing else loads. We prefer a
    bold font for major-tick labels so they stay legible against busy UIs.
    """
    from PIL import ImageFont

    bold_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",                 # macOS
        "/Library/Fonts/Arial Bold.ttf",                       # macOS (alt)
        "C:\\Windows\\Fonts\\arialbd.ttf",                     # Windows bold
        "C:\\Windows\\Fonts\\segoeuib.ttf",                    # Windows Segoe UI
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    ]
    regular_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    candidates = bold_candidates if bold else regular_candidates
    for path in candidates:
        try:
            # index=1 in .ttc collections is typically the Bold variant.
            if bold and path.endswith(".ttc"):
                return ImageFont.truetype(path, size, index=1)
            return ImageFont.truetype(path, size)
        except Exception:
            continue

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Older Pillow: load_default() doesn't take a size argument.
        return ImageFont.load_default()


def annotate_screenshot_with_axes(
    image: Image.Image,
    tick_spacing: int = 50,
) -> Image.Image:
    """Overlay a labelled coordinate grid on a screenshot.

    The result is a graph-paper-style image: a dark band along the top with
    X coordinates, a dark band along the left with Y coordinates, and faint
    gridlines crossing the screenshot. Every 100 pixels gets a thicker red
    line with a bold yellow number; smaller ticks get a faint yellow line
    with a smaller label.

    Why this exists:
        Small vision models are unreliable at guessing exact pixel
        coordinates from a raw screenshot. With this overlay the model just
        reads the nearest grid number above/beside the target element,
        which is a much easier task.

    Args:
        image: source screenshot.
        tick_spacing: distance between minor ticks. Use 100 for the resized
            image we send to the model (cleaner), 50 for full-resolution
            debug overlays (more precise).
    """
    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bold_font = _get_axis_font(16, bold=True)
    small_font = _get_axis_font(12, bold=True)
    axis_band = 22

    # Solid dark bands along the top and left so the labels stay readable
    # against bright UIs (white pages, light editor themes, etc.).
    draw.rectangle([0, 0, w, axis_band], fill=(0, 0, 0, 230))
    draw.rectangle([0, 0, axis_band + 14, h], fill=(0, 0, 0, 230))

    # X-axis: vertical gridlines + labels.
    for x in range(0, w, tick_spacing):
        is_major = (x % 100 == 0)
        line_color = (255, 0, 0, 140) if is_major else (255, 255, 0, 90)
        line_width = 2 if is_major else 1
        draw.line([(x, axis_band), (x, h)], fill=line_color, width=line_width)
        draw.line(
            [(x, axis_band - 4), (x, axis_band)],
            fill=(255, 255, 255, 255), width=line_width,
        )
        font = bold_font if is_major else small_font
        text_color = (255, 255, 0, 255) if is_major else (200, 200, 200, 255)
        draw.text((x + 2, 2 if is_major else 5), str(x), fill=text_color, font=font)

    # Y-axis: horizontal gridlines + labels.
    for y in range(0, h, tick_spacing):
        is_major = (y % 100 == 0)
        line_color = (255, 0, 0, 140) if is_major else (255, 255, 0, 90)
        line_width = 2 if is_major else 1
        draw.line([(axis_band + 14, y), (w, y)], fill=line_color, width=line_width)
        draw.line(
            [(axis_band + 10, y), (axis_band + 14, y)],
            fill=(255, 255, 255, 255), width=line_width,
        )
        font = bold_font if is_major else small_font
        text_color = (255, 255, 0, 255) if is_major else (200, 200, 200, 255)
        draw.text((1, y + 2), str(y), fill=text_color, font=font)

    base = image.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

def preflight_check() -> list[str]:
    """Run a sequence of start-up sanity checks. Returns a list of human-readable
    error strings; an empty list means everything is good.

    Checks performed:
        1. Screen capture works (Screen Recording / equivalent permission).
        2. Keyboard automation works (macOS Automation permission for System
           Events; not needed on Windows).
        3. LM Studio is reachable AND has the configured models loaded.

    Returning errors (instead of raising) lets the caller print a nice
    summary with all problems at once.
    """
    errors: list[str] = []

    # --- 1. Screen capture --------------------------------------------------
    try:
        img = take_screenshot()
        w, h = img.size
        if w <= 1 or h <= 1:
            errors.append(
                "Screen capture returned a 1x1 image (permission likely missing)."
            )
    except PermissionError as e:
        errors.append(str(e).strip())
    except Exception as e:
        errors.append(f"Screen capture failed: {e}")

    # --- 2. AppleScript / Automation permission (macOS only) ----------------
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to return name of first process'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                errors.append(
                    "AppleScript System Events access failed.\n"
                    "  Allow your terminal in:\n"
                    "  System Settings -> Privacy & Security -> Automation"
                )
        except Exception as e:
            errors.append(f"AppleScript check failed: {e}")

    # --- 3. LM Studio reachable + required models loaded --------------------
    try:
        req = urllib.request.Request(f"{config.LM_STUDIO_BASE}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json as _json
            data = _json.loads(resp.read())
        loaded_models = [m["id"] for m in data.get("data", [])]

        if config.VISION_MODEL not in loaded_models:
            errors.append(
                f"Vision model '{config.VISION_MODEL}' is not loaded in LM Studio.\n"
                f"  Loaded: {loaded_models}\n"
                f"  Open LM Studio's model manager and load it, or override with\n"
                f"  INVISIBLE_VISION_MODEL=<id-from-loaded-list>."
            )
        if config.THINKING_MODEL not in loaded_models:
            errors.append(
                f"Thinking model '{config.THINKING_MODEL}' is not loaded in LM Studio.\n"
                f"  Loaded: {loaded_models}\n"
                f"  Load it via LM Studio's model manager, or override with\n"
                f"  INVISIBLE_THINKING_MODEL=<id-from-loaded-list>."
            )
    except Exception as e:
        errors.append(
            f"Cannot connect to LM Studio at {config.LM_STUDIO_BASE}: {e}\n"
            f"  Start LM Studio with the local server enabled (port 1234 by default)."
        )

    return errors
