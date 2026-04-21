"""Decide whether a plan step is a "simple" deterministic action.

Many plan steps are unambiguous keyboard or app actions that don't need any
visual reasoning to execute:

    "Open Google Chrome"      -> open the app
    "Press Command+L"         -> hotkey
    "Type youtube.com"        -> type the text literally
    "Wait for page to load"   -> sleep a few seconds
    "Scroll down"             -> wheel scroll

Recognising these up front lets us skip the (slow, sometimes inaccurate)
vision model entirely. Everything that *isn't* a simple step is a "click"
step that goes through the clicker chain.
"""

from __future__ import annotations

import re


def is_simple_step(step: str) -> dict | None:
    """Try to convert `step` into a deterministic action dict.

    Returns:
        A ready-to-execute action dict if the step is simple, otherwise None.

    The action shape matches what `agent.execute_action` understands:
        {"action": "open_app", "app": "Google Chrome", "reason": <step>}
        {"action": "wait",     "seconds": 3,           "reason": <step>}
        {"action": "type",     "text": "youtube.com",  "reason": <step>}
        {"action": "press",    "key": "enter",         "reason": <step>}
        {"action": "hotkey",   "keys": ["command","l"], "reason": <step>}
        {"action": "scroll",   "direction": "down", "amount": 3, "reason": <step>}
    """
    s = step.lower().strip()

    # ---- Open <app> -----------------------------------------------------
    # Avoid matching "open the X" or "open a X" (those are usually clicks
    # like "open the Files menu" rather than "launch the Files app").
    if s.startswith("open ") and "open the " not in s and "open a " not in s:
        app_name = step[5:].strip().strip('"\'')
        if app_name and "search" not in s and "click" not in s:
            return {"action": "open_app", "app": app_name, "reason": step}

    # ---- Launch <app> ---------------------------------------------------
    if s.startswith("launch "):
        app_name = step[7:].strip().strip('"\'')
        if app_name:
            return {"action": "open_app", "app": app_name, "reason": step}

    # ---- Activate / Focus <app> ----------------------------------------
    if s.startswith("activate ") or s.startswith("focus "):
        prefix_len = 9 if s.startswith("activate") else 6
        app_name = step[prefix_len:].strip().strip('"\'')
        if app_name:
            return {"action": "activate_app", "app": app_name, "reason": step}

    # ---- Wait [N seconds | for X to load] ------------------------------
    if s.startswith("wait"):
        seconds_match = re.search(r"(\d+)\s*second", s)
        seconds = int(seconds_match.group(1)) if seconds_match else 3
        return {"action": "wait", "seconds": min(seconds, 10), "reason": step}

    # ---- Type "..." ----------------------------------------------------
    # Strips trailing 'in <field>' / 'into <field>' phrases that the planner
    # sometimes appends, plus surrounding quotes (including curly quotes).
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

    # ---- Scroll up / down ----------------------------------------------
    if s.startswith("scroll "):
        direction = "up" if "up" in s else "down"
        amount_match = re.search(r"(\d+)", s)
        amount = int(amount_match.group(1)) if amount_match else 3
        return {"action": "scroll", "direction": direction,
                "amount": amount, "reason": step}

    # ---- Press <Modifier>+<Key> ----------------------------------------
    hotkey_match = re.search(r"press\s+([\w]+)\s*\+\s*([\w]+)", s)
    if hotkey_match:
        key1, key2 = hotkey_match.group(1), hotkey_match.group(2)
        return {"action": "hotkey", "keys": [key1, key2], "reason": step}

    # ---- Press <key> ---------------------------------------------------
    if s.startswith("press "):
        raw = step[6:].strip()
        key = raw.split()[0] if raw.split() else raw
        if len(key) == 1:
            return {"action": "press", "key": key, "reason": step}
        return {"action": "press", "key": key.lower(), "reason": step}

    return None
