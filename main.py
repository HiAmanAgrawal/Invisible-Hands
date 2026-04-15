#!/usr/bin/env python3
"""
AI Desktop Agent — give it a task and watch it work.

How it works:
  1. You type a task like "open youtube in chrome"
  2. The text model (llama3) breaks it into steps:
     ["Open Google Chrome", "Wait for Chrome", "Press Cmd+L", "Type youtube.com", ...]
  3. For each step:
     - Simple steps (Open, Type, Press, Wait) are handled directly via AppleScript
     - Visual steps (Click something) use the vision model to find coordinates
  4. Sound plays after each step (Tink) and after all steps (Glass)
  5. A detailed JSON report + screenshots are saved to reports/ folder

Safety: move your mouse to the top-left corner to abort at any time.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime

from brain import create_plan, decide_action_for_step, verify_step_completion
from device_controller import (
    activate_app,
    annotate_screenshot_with_axes,
    click,
    double_click,
    get_screen_size,
    hotkey,
    move_to,
    open_app,
    play_done_sound,
    play_step_sound,
    preflight_check,
    press_key,
    right_click,
    screenshot_to_bytes,
    scroll_screen,
    take_screenshot,
    type_text,
)

# How many times to retry a step if the vision model fails
MAX_RETRIES_PER_STEP = 3

# Seconds to wait after each action (let the UI update)
DELAY_AFTER_ACTION = 2.0

# Countdown before starting (so you can switch windows)
COUNTDOWN_SECONDS = 3

# Folder where reports are saved
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# ── Terminal colors for pretty output ──
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


def c(text, color):
    """Wrap text in ANSI color codes for terminal output."""
    return f"{COLORS[color]}{text}{COLORS['reset']}"


# ────────────────────────────────────────────────────────────
# Report helper
# ────────────────────────────────────────────────────────────

def _make_report_dir(task: str) -> str:
    """Create a timestamped report directory for this run.

    Folder structure:
      reports/
        2026-03-20_15-30-45_search-bairan-youtube/
          report.json        ← full structured report
          screenshots/
            step_07_before.png   ← raw screenshot
            step_07_grid.png     ← screenshot with grid overlay
    """
    # Sanitize task into a short folder-safe slug
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower().strip())[:40].strip("-")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{timestamp}_{slug}"

    report_path = os.path.join(REPORTS_DIR, dir_name)
    screenshots_path = os.path.join(report_path, "screenshots")
    os.makedirs(screenshots_path, exist_ok=True)

    return report_path


def _save_report(report_path: str, report: dict):
    """Write the report dict as a pretty-printed JSON file."""
    report_file = os.path.join(report_path, "report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report_file


# ────────────────────────────────────────────────────────────
# Simple step detection
# ────────────────────────────────────────────────────────────

def _is_simple_step(step: str) -> dict | None:
    """Check if a step can be handled directly WITHOUT the vision model.

    Many steps are unambiguous keyboard actions that don't need
    a screenshot to figure out — we can just execute them directly.
    This avoids hallucinations from the vision model.

    Returns an action dict if simple, None if it needs the vision model.

    Examples of simple steps:
      "Open Google Chrome"     → {"action": "open_app", "app": "Google Chrome"}
      "Wait for Chrome"        → {"action": "wait", "seconds": 3}
      "Type youtube.com"       → {"action": "type", "text": "youtube.com"}
      "Press Command+L"        → {"action": "hotkey", "keys": ["command", "l"]}
      "Press Enter"            → {"action": "press", "key": "enter"}
      "Press /"                → {"action": "press", "key": "/"}
      "Press k"                → {"action": "press", "key": "k"}
      "Scroll down"            → {"action": "scroll", "direction": "down"}
      "Activate Chrome"        → {"action": "activate_app", "app": "Chrome"}

    Steps that need the vision model (returns None):
      "Click the search box"   → need to find it on screen
      "Click the play button"  → need coordinates from screenshot
    """
    s = step.lower().strip()

    # ── "Open <AppName>" → launch app via `open -a` ──
    if s.startswith("open ") and "open the " not in s and "open a " not in s:
        app_name = step[5:].strip().strip('"\'')
        if app_name and "search" not in s and "click" not in s:
            return {"action": "open_app", "app": app_name, "reason": step}

    # ── "Launch <AppName>" → same as Open ──
    if s.startswith("launch "):
        app_name = step[7:].strip().strip('"\'')
        if app_name:
            return {"action": "open_app", "app": app_name, "reason": step}

    # ── "Activate <App>" / "Focus <App>" → bring app to foreground ──
    if s.startswith("activate ") or s.startswith("focus "):
        prefix_len = 9 if s.startswith("activate") else 6
        app_name = step[prefix_len:].strip().strip('"\'')
        if app_name:
            return {"action": "activate_app", "app": app_name, "reason": step}

    # ── "Wait for X to load" / "Wait N seconds" ──
    if s.startswith("wait"):
        seconds_match = re.search(r"(\d+)\s*second", s)
        seconds = int(seconds_match.group(1)) if seconds_match else 3
        return {"action": "wait", "seconds": min(seconds, 10), "reason": step}

    # ── "Type X" → type text via AppleScript keystroke ──
    if s.startswith("type "):
        text = step[5:]
        for marker in ['" in ', "' in ", '" into ', "' into ",
                       '\u201d in ', '\u2019 in ']:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
                break
        text = text.strip().strip("\"'\u201c\u201d\u2018\u2019")
        if text:
            return {"action": "type", "text": text, "reason": step}

    # ── "Scroll down" / "Scroll up" ──
    if s.startswith("scroll "):
        direction = "up" if "up" in s else "down"
        amount_match = re.search(r"(\d+)", s)
        amount = int(amount_match.group(1)) if amount_match else 3
        return {"action": "scroll", "direction": direction,
                "amount": amount, "reason": step}

    # ── "Press Command+L" / "Press Command+K" → hotkey combo ──
    hotkey_match = re.search(r"press\s+([\w]+)\s*\+\s*([\w]+)", s)
    if hotkey_match:
        key1, key2 = hotkey_match.group(1), hotkey_match.group(2)
        return {"action": "hotkey", "keys": [key1, key2], "reason": step}

    # ── "Press Enter" / "Press Escape" → named key ──
    # ── "Press /" / "Press k" → single character key (app shortcuts) ──
    if s.startswith("press "):
        raw = step[6:].strip()
        key = raw.split()[0] if raw.split() else raw
        if len(key) == 1:
            return {"action": "press", "key": key, "reason": step}
        return {"action": "press", "key": key.lower(), "reason": step}

    return None


# ────────────────────────────────────────────────────────────
# Coordinate validation
# ────────────────────────────────────────────────────────────

def _validate_click_coords(action: dict, screen_w: int, screen_h: int) -> dict:
    """Validate and clamp click coordinates to screen bounds.

    Rejects clearly invalid coordinates (negative, zero, or far outside screen).
    Clamps borderline coordinates to safe screen area.
    Returns the (possibly adjusted) action, or an error action if unsalvageable.
    """
    x, y = action.get("x", -1), action.get("y", -1)

    if x <= 0 or y <= 0:
        return {"action": "error",
                "reason": f"Invalid coordinates ({x}, {y}) — zero or negative"}

    if x > screen_w * 1.1 or y > screen_h * 1.1:
        return {"action": "error",
                "reason": f"Coordinates ({x}, {y}) far outside screen {screen_w}x{screen_h}"}

    action["x"] = max(5, min(int(x), screen_w - 5))
    action["y"] = max(5, min(int(y), screen_h - 5))
    return action


# ────────────────────────────────────────────────────────────
# Action execution
# ────────────────────────────────────────────────────────────

def execute_action(action: dict) -> bool:
    """Execute a single action on the desktop.

    Dispatches to the appropriate device_controller function
    based on the action type.

    Returns True on success, False on error.
    Special: returns True for "done" (step already complete).
    """
    action_type = action.get("action")

    if action_type == "open_app":
        app = action["app"]
        open_app(app)
        time.sleep(3)
        activate_app(app)

    elif action_type == "activate_app":
        activate_app(action["app"])

    elif action_type == "click":
        click(action["x"], action["y"])

    elif action_type == "double_click":
        double_click(action["x"], action["y"])

    elif action_type == "right_click":
        right_click(action["x"], action["y"])

    elif action_type == "type":
        type_text(action["text"])

    elif action_type == "press":
        press_key(action["key"])

    elif action_type == "hotkey":
        hotkey(*action["keys"])

    elif action_type == "scroll":
        scroll_screen(
            action.get("direction", "down"),
            action.get("amount", 3),
        )

    elif action_type == "move":
        move_to(action["x"], action["y"])

    elif action_type == "wait":
        time.sleep(min(action.get("seconds", 2), 10))

    elif action_type == "done":
        return True

    elif action_type == "error":
        print(f"         {c('Parse error:', 'red')} {action.get('reason', '?')}")
        return False

    else:
        print(f"         {c('Unknown action:', 'red')} {action_type}")
        return False

    return True


def _format_action(action: dict) -> str:
    """Format an action dict into a readable one-line string for the terminal."""
    action_type = action.get("action", "?")

    if action_type == "open_app":
        return f'open_app "{action.get("app", "?")}"'
    elif action_type == "activate_app":
        return f'activate_app "{action.get("app", "?")}"'
    elif action_type in ("click", "double_click", "right_click", "move"):
        return f"{action_type} ({action.get('x')}, {action.get('y')})"
    elif action_type == "type":
        return f'type "{action.get("text", "")[:50]}"'
    elif action_type == "hotkey":
        return f"hotkey {'+'.join(action.get('keys', []))}"
    elif action_type == "press":
        return f"press {action.get('key', '?')}"
    elif action_type == "scroll":
        return f"scroll {action.get('direction', '?')} x{action.get('amount', 3)}"
    elif action_type == "wait":
        return f"wait {action.get('seconds', 2)}s"
    return action_type


# ────────────────────────────────────────────────────────────
# Agent loop
# ────────────────────────────────────────────────────────────

def _log_llm_call(label: str, result: dict):
    """Print detailed LLM call information to the terminal."""
    model = result.get("model", "?")
    duration = result.get("duration_s", 0)
    usage = result.get("usage")

    tokens_str = ""
    if usage:
        tokens_str = (f" | tokens: {usage.get('prompt_tokens', '?')}→"
                      f"{usage.get('completion_tokens', '?')} "
                      f"(total {usage.get('total_tokens', '?')})")

    print(f"         {c(f'[{label}]', 'dim')} model={model} | "
          f"time={duration}s{tokens_str}")

    thinking = result.get("thinking")
    if thinking:
        # Show first 3 lines of thinking, truncated
        lines = thinking.strip().split("\n")
        preview = lines[:3]
        print(f"         {c('Thinking:', 'magenta')}")
        for line in preview:
            truncated = line[:100] + ("..." if len(line) > 100 else "")
            print(f"           {c(truncated, 'dim')}")
        if len(lines) > 3:
            print(f"           {c(f'... ({len(lines) - 3} more lines)', 'dim')}")


def _verify_and_log(
    step: str,
    action: dict,
    step_idx: int,
    screenshots_dir: str,
    step_report: dict,
):
    """Take a screenshot after an action, run verification, and log results.
    Returns the verification result dict."""
    time.sleep(DELAY_AFTER_ACTION)

    try:
        after_screenshot = take_screenshot()
        after_path = os.path.join(screenshots_dir, f"step_{step_idx:02d}_after.png")
        after_screenshot.save(after_path)
        step_report["screenshot_after"] = f"screenshots/step_{step_idx:02d}_after.png"

        # Ask vision model to verify the step
        after_bytes = screenshot_to_bytes(after_screenshot)
        action_summary = _format_action(action)

        print(f"         {c('Verifying...', 'cyan')}")
        verify = verify_step_completion(step, action_summary, after_bytes)

        vr = verify["result"]
        verified = vr.get("verified", True)
        confidence = vr.get("confidence", "?")
        observation = vr.get("observation", "")

        if verified:
            status_icon = c("PASS", "green")
        else:
            status_icon = c("FAIL", "red")

        print(f"         {c('Verify:', 'bold')} {status_icon} "
              f"({confidence} confidence) | {verify['duration_s']}s")
        if observation:
            print(f"         {c('Observed:', 'dim')} {observation[:120]}")

        suggestion = vr.get("suggestion")
        if suggestion and not verified:
            print(f"         {c('Suggestion:', 'yellow')} {suggestion[:120]}")

        step_report["verification"] = {
            "verified": verified,
            "confidence": confidence,
            "observation": observation,
            "suggestion": suggestion,
            "duration_s": verify["duration_s"],
            "raw_response": verify["raw_response"],
        }

        return vr

    except Exception as e:
        print(f"         {c('Verify error:', 'red')} {e}")
        step_report["verification"] = {"verified": True, "error": str(e)}
        return {"verified": True}


def run_agent(task: str):
    """Run the full agent loop for a given task.

    Phase 1 — Planning:
      Uses the thinking model to create a step-by-step plan.
      Displays thinking process, timing, and token usage.

    Phase 2 — Executing + Verifying:
      For each step:
        - If it's a simple step (Type, Press, Open, Wait): execute directly
        - If it's a visual step (Click something): take screenshot,
          send to vision model, get coordinates, click
        - After each step: vision model verifies if it completed correctly
        - If verification fails: retry the step
        - Detailed logs shown for every LLM call
    """
    run_start = time.time()
    width, height = get_screen_size()

    report_path = _make_report_dir(task)
    screenshots_dir = os.path.join(report_path, "screenshots")

    report = {
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "screen": {"width": width, "height": height},
        "plan": [],
        "planning": {},
        "steps": [],
        "status": "running",
        "total_duration_s": 0,
    }

    print(f"\n  {c('Task:', 'bold')}   {task}")
    print(f"  {c('Screen:', 'dim')} {width}x{height}")
    print(f"  {c('Safety:', 'dim')} Move mouse to top-left corner to abort\n")

    # ── Phase 1: Plan ──
    print(f"  {c('Phase 1: Planning...', 'magenta')}")
    print(f"         {c('Prompt:', 'dim')} \"Task: {task}\"")

    plan_result = create_plan(task)
    plan = plan_result["steps"]

    # Log planning details
    _log_llm_call("Planner", plan_result)

    report["plan"] = plan
    report["planning"] = {
        "model": plan_result["model"],
        "duration_s": plan_result["duration_s"],
        "usage": plan_result["usage"],
        "thinking": plan_result["thinking"],
        "raw_response": plan_result["raw_response"],
    }

    print(f"\n  {c('Plan:', 'bold')} ({len(plan)} steps, "
          f"{plan_result['duration_s']}s)")
    for i, step in enumerate(plan, 1):
        print(f"    {c(f'{i}.', 'cyan')} {step}")
    print()

    # ── Phase 2: Execute + Verify each step ──
    print(f"  {c('Phase 2: Executing + Verifying...', 'magenta')}\n")

    for step_idx, step in enumerate(plan, 1):
        step_start = time.time()
        print(f"  {c(f'[Step {step_idx}/{len(plan)}]', 'cyan')} {step}")

        step_report = {
            "step_number": step_idx,
            "description": step,
            "method": None,
            "action": None,
            "status": "pending",
            "retries": 0,
            "duration_s": 0,
            "llm_calls": [],
            "verification": None,
        }

        # ── Simple steps (no vision needed) ──
        simple = _is_simple_step(step)
        if simple:
            action = simple
            step_report["method"] = "direct"
            step_report["action"] = action

            print(f"         {c('Action:', 'bold')} "
                  f"{c(_format_action(action), 'yellow')} (direct)")
            success = execute_action(action)
            step_report["status"] = "success" if success else "error"

            # Verify via vision after non-wait/non-trivial direct steps
            if action.get("action") not in ("wait",):
                vr = _verify_and_log(step, action, step_idx,
                                     screenshots_dir, step_report)
                if not vr.get("verified", True):
                    step_report["status"] = "verify_failed"
            else:
                time.sleep(DELAY_AFTER_ACTION)

            step_report["duration_s"] = round(time.time() - step_start, 2)
            print(f"         {c('Step time:', 'dim')} {step_report['duration_s']}s")
            play_step_sound()
            report["steps"].append(step_report)
            print()
            continue

        # ── Visual steps (clicks) — use vision model ──
        step_report["method"] = "vision_model"
        retries = 0
        while retries < MAX_RETRIES_PER_STEP:
            print(f"         Capturing screen...")
            screenshot = take_screenshot()

            raw_path = os.path.join(screenshots_dir, f"step_{step_idx:02d}_before.png")
            screenshot.save(raw_path)
            step_report["screenshot_before"] = f"screenshots/step_{step_idx:02d}_before.png"

            annotated = annotate_screenshot_with_axes(screenshot)
            img_bytes = screenshot_to_bytes(annotated)

            grid_path = os.path.join(screenshots_dir, f"step_{step_idx:02d}_axes.png")
            annotated.save(grid_path)
            step_report["screenshot_axes"] = f"screenshots/step_{step_idx:02d}_axes.png"

            print(f"         Thinking...")
            vision_result = decide_action_for_step(step, img_bytes, width, height)
            action = vision_result["action"]
            action_type = action.get("action", "?")

            # Log the vision model call details
            _log_llm_call("Vision", vision_result)
            step_report["llm_calls"].append({
                "type": "vision",
                "model": vision_result["model"],
                "duration_s": vision_result["duration_s"],
                "usage": vision_result["usage"],
                "thinking": vision_result["thinking"],
                "raw_response": vision_result["raw_response"],
                "prompt_sent": vision_result["prompt_sent"],
            })

            # Validate coordinates for click-type actions
            if action_type in ("click", "double_click", "right_click"):
                action = _validate_click_coords(action, width, height)
                action_type = action.get("action", "?")

            step_report["action"] = action

            print(f"         {c('Action:', 'bold')} "
                  f"{c(_format_action(action), 'yellow')}")
            reason = action.get("reason", "")
            if reason:
                print(f"         {c('Reason:', 'dim')} {reason}")

            if action_type == "done":
                print(f"         {c('(step already done, skipping)', 'green')}")
                step_report["status"] = "skipped"
                break

            if action_type == "error":
                print(f"         {c('(retrying...)', 'red')}")
                retries += 1
                step_report["retries"] = retries
                continue

            # Execute and verify
            success = execute_action(action)
            step_report["status"] = "success" if success else "error"

            vr = _verify_and_log(step, action, step_idx,
                                 screenshots_dir, step_report)

            if not vr.get("verified", True) and retries < MAX_RETRIES_PER_STEP - 1:
                print(f"         {c('Verification failed — retrying step...', 'red')}")
                retries += 1
                step_report["retries"] = retries
                continue

            break

        if retries >= MAX_RETRIES_PER_STEP:
            step_report["status"] = "failed_max_retries"

        step_report["duration_s"] = round(time.time() - step_start, 2)
        print(f"         {c('Step time:', 'dim')} {step_report['duration_s']}s")
        play_step_sound()
        report["steps"].append(step_report)
        print()

    # ── Save final report ──
    report["status"] = "completed"
    report["total_duration_s"] = round(time.time() - run_start, 2)
    report_file = _save_report(report_path, report)

    play_done_sound()
    print(f"  {c('All steps completed!', 'green')}")
    print(f"  {c('Total time:', 'bold')} {report['total_duration_s']}s")
    print(f"  {c('Report saved:', 'dim')} {report_file}\n")


# ────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────

def main():
    """Start the agent CLI.

    1. Show banner
    2. Run pre-flight checks (screen capture, AppleScript, Ollama)
    3. Enter interactive loop: read task → plan → execute → repeat
    """
    print()
    print(f"  {c('AI Desktop Agent', 'bold')}")
    print(f"  {c('────────────────────────────────────', 'dim')}")
    print(f"  Powered by LM Studio (qwen3.5 thinking + qwen2.5-vl vision)")
    print(f"  Uses AppleScript for keyboard, open -a for apps")
    print()

    # Run pre-flight checks before accepting tasks
    print(f"  {c('Running pre-flight checks...', 'dim')}")
    errors = preflight_check()
    if errors:
        print(f"  {c('Pre-flight checks FAILED:', 'red')}")
        for err in errors:
            for line in err.split("\n"):
                print(f"    {c('✗', 'red')} {line}")
        print()
        print(f"  Fix the issues above and try again.")
        sys.exit(1)

    # All checks passed — show green checkmarks
    print(f"  {c('✓ Screen capture OK', 'green')}")
    print(f"  {c('✓ AppleScript OK', 'green')} (System Events accessible)")
    print(f"  {c('✓ LM Studio + models OK', 'green')}")
    print()

    print(f"  Type a task and press Enter. Type {c('quit', 'yellow')} to exit.")
    print(f"  {c('Safety:', 'red')} Move mouse to top-left corner to emergency stop.")
    print()

    # Interactive loop — read tasks from the user
    while True:
        try:
            task = input(f"  {c('>', 'cyan')} ").strip()

            if not task:
                continue
            if task.lower() in ("quit", "exit", "q"):
                print(f"  {c('Goodbye!', 'dim')}")
                break

            # Countdown so the user can switch to the target window
            print(f"\n  {c('Starting in...', 'yellow')}")
            for i in range(COUNTDOWN_SECONDS, 0, -1):
                print(f"  {c(str(i), 'bold')}")
                time.sleep(1)

            # Run the agent for this task
            run_agent(task)

        except KeyboardInterrupt:
            print(f"\n\n  {c('Interrupted. Goodbye!', 'dim')}")
            sys.exit(0)


if __name__ == "__main__":
    main()
