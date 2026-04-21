"""Pretty terminal output + per-run JSON report writer.

This is purely presentation: nothing in this file makes decisions about
what the agent should do — it only prints things and saves files. Keeping
the agent loop free of formatting noise (long f-strings, ANSI escape
sequences, JSON munging) makes it MUCH easier to read.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

from invisible_hands import config


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colors
# ─────────────────────────────────────────────────────────────────────────────
# Standard 8-color ANSI codes — work in every modern terminal without any
# extra dependencies (no termcolor / colorama needed on modern Windows
# terminals, which support ANSI by default).

COLORS = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "reset": "\033[0m",
}


def c(text: str, color: str) -> str:
    """Wrap `text` in the given ANSI color code. Unknown colors pass through."""
    code = COLORS.get(color)
    if not code:
        return text
    return f"{code}{text}{COLORS['reset']}"


# ─────────────────────────────────────────────────────────────────────────────
# Report directories + files
# ─────────────────────────────────────────────────────────────────────────────

def make_report_dir(task: str) -> tuple[str, str]:
    """Create a timestamped directory for this run's artefacts.

    Returns (report_dir, screenshots_dir). The structure looks like:

        reports/
          2026-04-14_15-30-45_open-youtube/
            report.json
            screenshots/
              step_01_before.png
              step_01_axes.png
              step_01_after.png
    """
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower().strip())[:40].strip("-")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{timestamp}_{slug}" if slug else timestamp

    report_dir = os.path.join(config.REPORTS_DIR, dir_name)
    screenshots_dir = os.path.join(report_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    return report_dir, screenshots_dir


def save_report(report_dir: str, report: dict) -> str:
    """Pretty-print the report dict as report.json and return its path."""
    report_file = os.path.join(report_dir, "report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report_file


# ─────────────────────────────────────────────────────────────────────────────
# Action formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_action(action: dict) -> str:
    """Render an action dict as a short, terminal-friendly one-liner.

    Examples:
        {"action": "click", "x": 800, "y": 400}      -> 'click (800, 400)'
        {"action": "type",  "text": "youtube.com"}   -> 'type "youtube.com"'
        {"action": "hotkey","keys": ["command","l"]} -> 'hotkey command+l'
    """
    a = action.get("action", "?")
    if a == "open_app":
        return f'open_app "{action.get("app", "?")}"'
    if a == "activate_app":
        return f'activate_app "{action.get("app", "?")}"'
    if a in ("click", "double_click", "right_click", "move"):
        return f"{a} ({action.get('x')}, {action.get('y')})"
    if a == "type":
        return f'type "{action.get("text", "")[:50]}"'
    if a == "hotkey":
        return f"hotkey {'+'.join(action.get('keys', []))}"
    if a == "press":
        return f"press {action.get('key', '?')}"
    if a == "scroll":
        return f"scroll {action.get('direction', '?')} x{action.get('amount', 3)}"
    if a == "wait":
        return f"wait {action.get('seconds', 2)}s"
    return a


# ─────────────────────────────────────────────────────────────────────────────
# LLM call logging
# ─────────────────────────────────────────────────────────────────────────────

def log_llm_call(label: str, result: dict) -> None:
    """Print a one-line LLM call summary, plus a thinking preview if present.

    Output looks like:
        [Planner] model=qwen/qwen3.5-9b | time=2.31s | tokens: 540->180 (total 720)
        Thinking:
          The user wants to open youtube...
          I'll need to open chrome first...
    """
    model = result.get("model", "?")
    duration = result.get("duration_s", 0)
    usage = result.get("usage")

    tokens_str = ""
    if usage:
        tokens_str = (
            f" | tokens: {usage.get('prompt_tokens', '?')}->"
            f"{usage.get('completion_tokens', '?')} "
            f"(total {usage.get('total_tokens', '?')})"
        )

    print(
        f"         {c(f'[{label}]', 'dim')} model={model} | "
        f"time={duration}s{tokens_str}"
    )

    thinking = result.get("thinking")
    if thinking:
        lines = thinking.strip().split("\n")
        preview = lines[:3]
        print(f"         {c('Thinking:', 'magenta')}")
        for line in preview:
            truncated = line[:100] + ("..." if len(line) > 100 else "")
            print(f"           {c(truncated, 'dim')}")
        if len(lines) > 3:
            print(f"           {c(f'... ({len(lines) - 3} more lines)', 'dim')}")
