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

Safety: move your mouse to the top-left corner to abort at any time.
"""

from __future__ import annotations

import re
import sys
import time

from brain import create_plan, decide_action_for_step
from device_controller import (
    activate_app,
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

    Steps that need the vision model (returns None):
      "Click the search box"   → need to find it on screen
      "Click the play button"  → need coordinates from screenshot
    """
    s = step.lower().strip()

    # ── "Open <AppName>" → launch app via `open -a` ──
    # Matches: "Open Google Chrome", "Open Spotify"
    # Skips: "Open the search box" (that's a click action)
    if s.startswith("open ") and "open the " not in s and "open a " not in s:
        app_name = step[5:].strip().strip('"\'')
        if app_name and "search" not in s and "click" not in s:
            return {"action": "open_app", "app": app_name, "reason": step}

    # ── "Launch <AppName>" → same as Open ──
    if s.startswith("launch "):
        app_name = step[7:].strip().strip('"\'')
        if app_name:
            return {"action": "open_app", "app": app_name, "reason": step}

    # ── "Wait for X to load" / "Wait N seconds" ──
    if s.startswith("wait"):
        seconds_match = re.search(r"(\d+)\s*second", s)
        seconds = int(seconds_match.group(1)) if seconds_match else 3
        return {"action": "wait", "seconds": min(seconds, 10), "reason": step}

    # ── "Type X" → type text via AppleScript keystroke ──
    # Matches: 'Type "youtube.com" in the address bar'
    # Extracts just "youtube.com" (strips the "in the address bar" part)
    if s.startswith("type "):
        text = step[5:]  # strip "Type "
        # Remove trailing context like 'in the search box', 'in the address bar'
        for marker in ['" in ', "' in ", '" into ', "' into ",
                       '\u201d in ', '\u2019 in ']:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
                break
        # Remove surrounding quotes
        text = text.strip().strip("\"'\u201c\u201d\u2018\u2019")
        if text:
            return {"action": "type", "text": text, "reason": step}

    # ── "Press Command+L" / "Press Command+K" → hotkey combo ──
    # Uses regex to match "press <mod>+<key>" patterns
    hotkey_match = re.search(r"press\s+([\w]+)\s*\+\s*([\w]+)", s)
    if hotkey_match:
        key1, key2 = hotkey_match.group(1), hotkey_match.group(2)
        return {"action": "hotkey", "keys": [key1, key2], "reason": step}

    # ── "Press Enter" / "Press Escape" / "Press Tab" → single key ──
    if s.startswith("press "):
        key = step[6:].strip().split()[0].lower()
        return {"action": "press", "key": key, "reason": step}

    # Not a simple step — needs the vision model
    return None


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
        # Step 1: Launch the app using `open -a`
        app = action["app"]
        open_app(app)
        # Step 2: Wait for it to start up
        time.sleep(3)
        # Step 3: Bring it to the foreground (like test.py does with activate)
        activate_app(app)

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

def run_agent(task: str):
    """Run the full agent loop for a given task.

    Phase 1 — Planning:
      Sends the task to llama3 which returns a numbered list of steps.

    Phase 2 — Executing:
      For each step:
        - If it's a simple step (Type, Press, Open, Wait): execute directly
        - If it's a visual step (Click something): take screenshot,
          send to vision model, get coordinates, click
        - Play a sound after each completed step
      Play a different sound when all steps are done.
    """
    width, height = get_screen_size()

    print(f"\n  {c('Task:', 'bold')}   {task}")
    print(f"  {c('Screen:', 'dim')} {width}x{height}")
    print(f"  {c('Safety:', 'dim')} Move mouse to top-left corner to abort\n")

    # ── Phase 1: Plan ──
    # Send task to text model, get back a list of steps
    print(f"  {c('Phase 1: Planning...', 'magenta')}")
    plan = create_plan(task)
    print(f"  {c('Plan:', 'bold')} ({len(plan)} steps)")
    for i, step in enumerate(plan, 1):
        print(f"    {c(f'{i}.', 'cyan')} {step}")
    print()

    # ── Phase 2: Execute each step ──
    print(f"  {c('Phase 2: Executing...', 'magenta')}\n")

    for step_idx, step in enumerate(plan, 1):
        print(f"  {c(f'[Step {step_idx}/{len(plan)}]', 'cyan')} {step}")

        # Check if this step can be handled directly (no vision model needed)
        simple = _is_simple_step(step)
        if simple:
            action = simple
            print(f"         {c('Action:', 'bold')} "
                  f"{c(_format_action(action), 'yellow')} (direct)")
            execute_action(action)
            time.sleep(DELAY_AFTER_ACTION)
            # Play step-complete sound
            play_step_sound()
            print()
            continue

        # For visual steps (clicks), use the vision model to find coordinates
        retries = 0
        while retries < MAX_RETRIES_PER_STEP:
            # Take a screenshot of the current screen
            print(f"         Capturing screen...")
            screenshot = take_screenshot()
            img_bytes = screenshot_to_bytes(screenshot)

            # Send screenshot to vision model and ask what to do
            print(f"         Thinking...")
            action = decide_action_for_step(step, img_bytes, width, height)
            action_type = action.get("action", "?")

            # Display the action the model chose
            print(f"         {c('Action:', 'bold')} "
                  f"{c(_format_action(action), 'yellow')}")
            reason = action.get("reason", "")
            if reason:
                print(f"         {c('Reason:', 'dim')} {reason}")

            # If the model says "done", the step is already complete
            if action_type == "done":
                print(f"         {c('(step already done, skipping)', 'green')}")
                break

            # If parsing failed, retry (up to MAX_RETRIES_PER_STEP)
            if action_type == "error":
                print(f"         {c('(retrying...)', 'red')}")
                retries += 1
                continue

            # Execute the action and move on
            execute_action(action)
            time.sleep(DELAY_AFTER_ACTION)
            break

        # Play step-complete sound
        play_step_sound()
        print()

    # Play flow-complete sound (different from step sound)
    play_done_sound()
    print(f"  {c('All steps completed!', 'green')}\n")


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
    print(f"  Powered by Ollama (llama3.2-vision + llama3)")
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
    print(f"  {c('✓ Ollama + models OK', 'green')}")
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
