#!/usr/bin/env python3
"""Click-accuracy benchmark for the full clicker chain.

What it does:
    1. Spins up a tiny local HTTP server that serves a single HTML page.
    2. Opens the page in Chrome with a bright yellow CLICK ME button placed
       at a random screen position (controlled via URL query string).
    3. Takes a real screenshot, finds the actual on-screen button position
       by colour clustering (so we know exactly where the truth is even
       after Chrome's chrome offsets the page).
    4. Asks the FULL clicker chain ("Click the yellow CLICK ME button") to
       locate it. The chain might pick OCR, native UI, or vision depending
       on what's installed and what wins first.
    5. Compares the clicker's coordinates against the true button bounds
       and prints PASS / FAIL plus distance + which strategy won.

Useful for:
    - Sanity-checking that a clicker change still hits the button.
    - Comparing strategies: turn each off in turn (`--no-ocr`, `--no-native`,
      `--no-vision`) to benchmark them in isolation.
    - Seeing real numbers for how often vision wins vs OCR.

Usage:
    python scripts/test_click.py                    # 1 trial
    python scripts/test_click.py --rounds 5         # 5 trials in a row
    python scripts/test_click.py --click            # actually click (visible test)
    python scripts/test_click.py --save             # save annotated screenshots
    python scripts/test_click.py --no-ocr           # skip the OCR clicker
    python scripts/test_click.py --no-native        # skip the native UI clicker
    python scripts/test_click.py --no-vision        # skip the vision clicker
"""

from __future__ import annotations

import argparse
import http.server
import math
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image

# Make the in-tree package importable when running as a plain script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from invisible_hands.clickers.base import ClickRequest
from invisible_hands.clickers.chain import ClickerChain
from invisible_hands.clickers.native import NativeUIClicker
from invisible_hands.clickers.ocr import TesseractClicker
from invisible_hands.clickers.vision import VisionClicker
from invisible_hands.controllers.input import click as do_click_action
from invisible_hands.controllers.screen import (
    annotate_screenshot_with_axes,
    get_screen_size,
    take_screenshot,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixture (HTML page with a single yellow button)
# ─────────────────────────────────────────────────────────────────────────────

BUTTON_WIDTH = 180
BUTTON_HEIGHT = 70
BUTTON_TEXT = "CLICK ME"

# A "hit" is a click within this pixel distance of the button's bounds.
TOLERANCE_PX = 50

SERVER_PORT = 9877

# A bare-bones HTML page. The button is positioned via URL query params
# (?x=...&y=...) so we can move it between trials without restarting Chrome.
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Click Test</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    width: 100vw; height: 100vh;
    background: white;
    overflow: hidden;
    position: relative;
    font-family: -apple-system, Helvetica, Arial, sans-serif;
  }
  #target-btn {
    position: absolute;
    width: BUTTON_WIDTHpx;
    height: BUTTON_HEIGHTpx;
    background: #FFD700;
    color: black;
    font-size: 22px;
    font-weight: bold;
    border: 3px solid #B8960F;
    border-radius: 12px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    user-select: none;
  }
  #target-btn:hover { background: #FFC000; }
  #target-btn:active { background: #E6B800; transform: scale(0.97); }
  #status {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 18px;
    color: #888;
    pointer-events: none;
  }
  #flash {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 999;
    pointer-events: none;
  }
</style>
</head>
<body>
  <button id="target-btn">BUTTON_TEXT</button>
  <div id="status">Click Test - waiting for AI...</div>
  <div id="flash"></div>
  <script>
    const params = new URLSearchParams(window.location.search);
    const bx = parseInt(params.get('x') || '400');
    const by = parseInt(params.get('y') || '300');
    const btn = document.getElementById('target-btn');
    btn.style.left = bx + 'px';
    btn.style.top = by + 'px';

    btn.addEventListener('click', () => {
      const flash = document.getElementById('flash');
      flash.style.display = 'block';
      flash.style.background = 'rgba(0, 200, 0, 0.3)';
      document.getElementById('status').textContent = 'HIT! Button was clicked.';
      document.getElementById('status').style.color = 'green';
      document.title = 'CLICKED';
      setTimeout(() => flash.style.display = 'none', 500);
    });

    document.body.addEventListener('click', (e) => {
      if (e.target === btn) return;
      const flash = document.getElementById('flash');
      flash.style.display = 'block';
      flash.style.background = 'rgba(200, 0, 0, 0.2)';
      document.getElementById('status').textContent =
        'MISS - clicked at (' + e.clientX + ', ' + e.clientY + ')';
      document.getElementById('status').style.color = 'red';
      setTimeout(() => flash.style.display = 'none', 500);
    });
  </script>
</body>
</html>""".replace("BUTTON_WIDTH", str(BUTTON_WIDTH)).replace(
    "BUTTON_HEIGHT", str(BUTTON_HEIGHT)
).replace("BUTTON_TEXT", BUTTON_TEXT)


# ANSI colours — same palette as the agent for visual continuity.
GREEN, RED, YELLOW, CYAN = "\033[92m", "\033[91m", "\033[93m", "\033[96m"
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# HTTP server
# ─────────────────────────────────────────────────────────────────────────────

class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the HTML page and silences the default access-log spam."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def log_message(self, *args):
        pass


def _start_server() -> http.server.HTTPServer:
    """Start the HTTP server on SERVER_PORT in a daemon thread."""
    server = http.server.HTTPServer(("127.0.0.1", SERVER_PORT), _QuietHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ─────────────────────────────────────────────────────────────────────────────
# Button placement + ground-truth detection
# ─────────────────────────────────────────────────────────────────────────────

def _random_button_position(screen_w: int, screen_h: int) -> tuple[int, int]:
    """Pick a CSS position for the button inside the browser viewport.

    We leave wide margins so the button never lands behind Chrome's chrome
    or off-screen — this isn't a placement test, it's a clicker test."""
    margin = 200
    max_x = screen_w - margin - BUTTON_WIDTH
    max_y = screen_h - margin - BUTTON_HEIGHT
    css_x = random.randint(margin, max(margin + 1, max_x))
    css_y = random.randint(100, max(101, max_y - 100))
    return css_x, css_y


def _find_yellow_button(screenshot: Image.Image) -> dict | None:
    """Locate the actual gold button in the screenshot by colour clustering.

    Why we need this:
        We told Chrome to put the button at CSS (x, y), but Chrome adds
        its own chrome (title bar, bookmarks bar, scrollbar) so the
        physical screen pixel where the button ends up isn't the same as
        the CSS coordinate. This function reads the truth from the
        screenshot itself.

    Algorithm:
        1. Mask all pixels that match the gold colour (#FFD700-ish).
        2. Find the densest contiguous block of those pixels (the button).
        3. Reject anything wildly the wrong size (favicons, scrollbars).
        4. Return its bounding box + center.
    """
    arr = np.array(screenshot)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (r >= 220) & (g >= 190) & (b < 80)

    coords = np.argwhere(mask)
    if len(coords) < 200:
        return None

    ys = coords[:, 0]
    xs = coords[:, 1]

    y_hist = np.bincount(ys, minlength=arr.shape[0])
    active_rows = np.where(y_hist >= 40)[0]
    if len(active_rows) < 10:
        return None

    diffs = np.diff(active_rows)
    splits = np.where(diffs > 3)[0]
    segments = np.split(active_rows, splits + 1)
    best_seg = max(segments, key=len)

    y_min, y_max = int(best_seg[0]), int(best_seg[-1])
    row_mask = (ys >= y_min) & (ys <= y_max)
    button_xs = xs[row_mask]
    x_min, x_max = int(button_xs.min()), int(button_xs.max())

    w = x_max - x_min
    h = y_max - y_min
    if w < 50 or w > 500 or h < 20 or h > 200:
        return None

    cx = x_min + w // 2
    cy = y_min + h // 2
    return {
        "x": int(x_min), "y": int(y_min),
        "w": int(w), "h": int(h),
        "cx": int(cx), "cy": int(cy),
    }


def _open_test_page(css_x: int, css_y: int) -> None:
    """Tell the OS to open the test page in Chrome at the given button pos."""
    url = f"http://127.0.0.1:{SERVER_PORT}/?x={css_x}&y={css_y}"
    if sys.platform == "darwin":
        subprocess.run(["open", "-a", "Google Chrome", url], capture_output=True)
    elif sys.platform == "win32":
        subprocess.Popen(
            f'start chrome "{url}"', shell=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(["xdg-open", url], capture_output=True)


# ─────────────────────────────────────────────────────────────────────────────
# Trial loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(
    chain: ClickerChain,
    trial_num: int,
    screen_w: int,
    screen_h: int,
    do_click: bool,
    save_dir: str | None,
) -> dict:
    """Run one round: place button, screenshot, ask chain, score."""
    print(f"\n  {CYAN}[Trial {trial_num}]{RESET}")

    css_x, css_y = _random_button_position(screen_w, screen_h)
    print(f"  {DIM}Button CSS pos: ({css_x}, {css_y}){RESET}")

    _open_test_page(css_x, css_y)
    time.sleep(2.5)  # let Chrome render

    screenshot = take_screenshot()

    detected = _find_yellow_button(screenshot)
    if not detected:
        print(f"  {RED}Could not detect yellow button - skipping trial{RESET}")
        return {
            "trial": trial_num, "hit": False, "distance_px": 9999,
            "time_s": 0, "winner": "none",
            "model_x": -1, "model_y": -1,
            "button_screen": (0, 0), "button_center": (0, 0),
            "reason": "button not detected",
        }

    btn_x, btn_y, btn_w, btn_h = detected["x"], detected["y"], detected["w"], detected["h"]
    btn_cx, btn_cy = detected["cx"], detected["cy"]
    print(f"  {DIM}Detected button: ({btn_x}, {btn_y}) size {btn_w}x{btn_h} - "
          f"center ({btn_cx}, {btn_cy}){RESET}")

    if save_dir:
        screenshot.save(os.path.join(save_dir, f"trial_{trial_num:02d}_raw.png"))
        annotated = annotate_screenshot_with_axes(screenshot)
        annotated.save(os.path.join(save_dir, f"trial_{trial_num:02d}_axes.png"))

    step = f'Click the yellow "{BUTTON_TEXT}" button'
    print(f"  {DIM}Step: \"{step}\"{RESET}")
    print(f"  {DIM}Trying clickers: {' -> '.join(chain.names) or '(none)'}{RESET}")

    request = ClickRequest(
        step=step,
        screenshot=screenshot,
        screen_size=(screen_w, screen_h),
    )

    t0 = time.time()
    result, attempts = chain.find(request)
    elapsed = time.time() - t0

    for att in attempts:
        if att.get("found"):
            print(f"    {GREEN}{att['name'].upper()}{RESET} -> "
                  f"({att['x']}, {att['y']}) "
                  f"conf={att['confidence']:.2f} in {att['duration_s']}s")
        elif "error" in att:
            print(f"    {RED}{att['name'].upper()} ERROR{RESET}: "
                  f"{att['error']} ({att['duration_s']}s)")
        else:
            print(f"    {DIM}{att['name'].upper()} miss{RESET} "
                  f"({att['duration_s']}s)")

    model_x, model_y = result.x, result.y

    hit_x = (btn_x - TOLERANCE_PX) <= model_x <= (btn_x + btn_w + TOLERANCE_PX)
    hit_y = (btn_y - TOLERANCE_PX) <= model_y <= (btn_y + btn_h + TOLERANCE_PX)
    hit = result.found and hit_x and hit_y
    distance = math.sqrt((model_x - btn_cx) ** 2 + (model_y - btn_cy) ** 2)

    if not result.found:
        print(f"\n  {RED}{BOLD}FAIL{RESET} - all clickers missed")
    elif hit:
        print(f"\n  {GREEN}{BOLD}PASS{RESET} - {result.source} click "
              f"({model_x}, {model_y}) hits the button "
              f"(distance: {distance:.0f}px from center)")
    else:
        print(f"\n  {RED}{BOLD}FAIL{RESET} - {result.source} click "
              f"({model_x}, {model_y}) misses button bounds "
              f"({btn_x}..{btn_x + btn_w}, {btn_y}..{btn_y + btn_h}) "
              f"(distance: {distance:.0f}px from center)")

    if do_click and result.found:
        print(f"  {YELLOW}Clicking at ({model_x}, {model_y})...{RESET}")
        time.sleep(0.5)
        do_click_action(model_x, model_y)
        time.sleep(1.0)

    return {
        "trial": trial_num,
        "winner": result.source,
        "button_screen": (btn_x, btn_y),
        "button_size": (btn_w, btn_h),
        "button_center": (btn_cx, btn_cy),
        "model_x": model_x,
        "model_y": model_y,
        "hit": hit,
        "distance_px": round(distance, 1),
        "time_s": round(elapsed, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark the click chain.")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--click", action="store_true",
                   help="Actually click after each trial (visible test).")
    p.add_argument("--save", action="store_true",
                   help="Save raw + annotated screenshots into test_results/.")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--no-native", action="store_true")
    p.add_argument("--no-vision", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    save_dir = None
    if args.save:
        save_dir = os.path.join(_REPO_ROOT, "test_results")
        os.makedirs(save_dir, exist_ok=True)

    chain = ClickerChain([
        TesseractClicker() if not args.no_ocr else None,
        NativeUIClicker() if not args.no_native else None,
        VisionClicker() if not args.no_vision else None,
    ])

    screen_w, screen_h = get_screen_size()

    print()
    print(f"  {BOLD}Click Accuracy Test{RESET}")
    print(f"  {DIM}{'-' * 40}{RESET}")
    print(f"  Screen:    {screen_w}x{screen_h}")
    print(f"  Rounds:    {args.rounds}")
    print(f"  Tolerance: {TOLERANCE_PX}px")
    print(f"  Click:     {'yes' if args.click else 'no (coordinate check only)'}")
    print(f"  Save:      {save_dir or 'no'}")
    print(f"  Chain:     {' -> '.join(chain.names) or '(none enabled!)'}")

    print(f"  {DIM}Starting test server on port {SERVER_PORT}...{RESET}")
    server = _start_server()

    print(f"\n  {YELLOW}Starting in...{RESET}")
    for sec in range(3, 0, -1):
        print(f"  {BOLD}{sec}{RESET}")
        time.sleep(1)

    results = []
    for r in range(1, args.rounds + 1):
        results.append(
            _run_trial(chain, r, screen_w, screen_h, args.click, save_dir)
        )
        if r < args.rounds:
            time.sleep(1.5)

    passed = sum(1 for r in results if r["hit"])
    total = len(results)
    distances = [r["distance_px"] for r in results]
    avg_dist = sum(distances) / len(distances) if distances else 0
    avg_time = sum(r["time_s"] for r in results) / total if total else 0

    print(f"\n  {BOLD}{'-' * 40}{RESET}")
    color = GREEN if passed == total else (YELLOW if passed > 0 else RED)
    print(f"  {color}{BOLD}Results: {passed}/{total} passed{RESET}")
    print(f"  Avg distance from center: {avg_dist:.0f}px")
    print(f"  Avg total time: {avg_time:.1f}s")

    by_winner: dict[str, int] = {}
    for r in results:
        by_winner[r["winner"]] = by_winner.get(r["winner"], 0) + 1
    if by_winner:
        winners = ", ".join(f"{k}={v}" for k, v in by_winner.items())
        print(f"  Winners: {winners}")

    if total > 1:
        print(f"\n  {DIM}Per-trial:{RESET}")
        for r in results:
            icon = f"{GREEN}PASS{RESET}" if r["hit"] else f"{RED}FAIL{RESET}"
            print(f"    Trial {r['trial']}: {icon}  "
                  f"winner={r['winner']:<10}  "
                  f"model=({r['model_x']},{r['model_y']})  "
                  f"target=({r['button_center'][0]},{r['button_center'][1]})  "
                  f"dist={r['distance_px']}px  time={r['time_s']}s")

    print()
    if save_dir:
        print(f"  {DIM}Screenshots saved to: {save_dir}{RESET}\n")

    server.shutdown()
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
