"""
Brain module — the "thinking" part of the AI agent.

Connects to Ollama (local LLM server) and uses two models:
  1. llama3 (text-only) — fast model for PLANNING
     "Break this task into steps: open youtube in chrome"
     → ["Open Google Chrome", "Wait for Chrome to load", ...]

  2. llama3.2-vision (vision model) — for EXECUTING visual steps
     Given a screenshot + step like "Click the search box",
     it returns {"action": "click", "x": 500, "y": 300}

This two-phase approach works better than asking the vision model
to do everything, because small vision models tend to hallucinate
coordinates or repeat the task description as text.
"""

from __future__ import annotations

import json
import re
import ollama

# Vision model — looks at screenshots and decides where to click
MODEL = "llama3.2-vision"

# Text model — plans the task into steps (no vision needed, much faster)
TEXT_MODEL = "llama3"

# ────────────────────────────────────────────────────────────
# System prompts
# ────────────────────────────────────────────────────────────

# PLANNER_PROMPT tells the text model how to break a task into steps.
# Key design decisions:
#   - "Open <AppName>" instead of Spotlight (we use `open -a` under the hood)
#   - "Press Command+L" to focus browser address bar (reliable keyboard shortcut)
#   - Each step is ONE action (no combining "open app and type url")
PLANNER_PROMPT = """You are a macOS desktop automation planner.
Given a task, output a numbered list of concrete low-level steps to accomplish it.

Rules:
- Each step must be a single physical action.
- To open/launch an app: write "Open <AppName>" (e.g. "Open Google Chrome", "Open Spotify").
- After opening an app, add "Wait for <AppName> to load" as the next step.
- To focus the browser address bar: write "Press Command+L" (this works in all browsers).
- To navigate to a URL: first "Press Command+L", then type the URL, then press Enter.
- To search on YouTube: go to youtube.com first, then click the search box, type the query, press Enter.
- To use a keyboard shortcut in an app: write "Press Command+K" or similar.
- Do NOT use Spotlight. Always use "Open <AppName>" directly.
- Do NOT combine multiple actions into one step.
- Output ONLY the numbered list, nothing else. No introduction or summary.

Example task: "open youtube in chrome"
1. Open Google Chrome
2. Wait for Google Chrome to load
3. Press Command+L to focus the address bar
4. Type "youtube.com" in the address bar
5. Press Enter to go to YouTube
6. Wait for YouTube to load

Example task: "play bairan on spotify"
1. Open Spotify
2. Wait for Spotify to load
3. Press Command+K to open Spotify search
4. Type "bairan" in the search box
5. Press Enter to search"""

# EXECUTOR_PROMPT tells the vision model how to look at a screenshot
# and decide what action to take for a specific step.
# It only handles visual tasks (clicking UI elements).
# Keyboard actions are handled directly without the vision model.
EXECUTOR_PROMPT = """You are a macOS desktop automation agent. You see a screenshot of the screen and must perform ONE specific step.

Your current step: {current_step}

Look at the screenshot carefully. Based on what you see, choose the RIGHT action to accomplish this step.

Available actions — respond with ONLY a valid JSON object:

{{"action": "click", "x": <int>, "y": <int>, "reason": "<why>"}}
{{"action": "double_click", "x": <int>, "y": <int>, "reason": "<why>"}}
{{"action": "type", "text": "<text to type>", "reason": "<why>"}}
{{"action": "hotkey", "keys": ["key1", "key2"], "reason": "<why>"}}
{{"action": "press", "key": "<key name>", "reason": "<why>"}}
{{"action": "scroll", "direction": "up|down", "amount": <1-10>, "reason": "<why>"}}
{{"action": "wait", "seconds": <1-5>, "reason": "<why>"}}
{{"action": "done", "reason": "<step is already complete based on what I see>"}}

CRITICAL rules:
1. Output ONLY valid JSON. No markdown, no extra text.
2. Coordinates are pixels from the top-left corner. The screen is {width}x{height}.
3. If the step says "Press Command+K", use {{"action": "hotkey", "keys": ["command", "k"]}}.
4. If the step says "Type X", use {{"action": "type", "text": "X"}}. Type ONLY what the step says, not the task description.
5. If the step says "Press Enter", use {{"action": "press", "key": "enter"}}.
6. If the step says "Click X", find X on the screenshot and click its coordinates.
7. If the step appears already done (e.g. the app is already open), use "done"."""


# ────────────────────────────────────────────────────────────
# Phase 1: Planning
# ────────────────────────────────────────────────────────────

def create_plan(task: str) -> list[str]:
    """Ask the text model to break the task into concrete steps.

    How it works:
      1. Sends the task to llama3 with the PLANNER_PROMPT
      2. Parses the numbered list from the response
      3. Filters out preamble lines like "Here is the list..."
      4. Returns a clean list of step strings

    Example:
      create_plan("open youtube in chrome")
      → ["Open Google Chrome", "Wait for Google Chrome to load",
         "Press Command+L to focus the address bar",
         "Type \"youtube.com\"", "Press Enter"]
    """
    # Send the task to the text model with low temperature (deterministic)
    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": f"Task: {task}"},
        ],
        options={"temperature": 0.1},
    )

    raw = response["message"]["content"].strip()

    # Parse the numbered list from the model's response
    steps = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Strip the leading number + punctuation: "1. Open Chrome" → "Open Chrome"
        # Matches patterns like "1.", "1)", "1:", "1 -", etc.
        for i, ch in enumerate(line):
            if not ch.isdigit() and ch not in ".):- ":
                line = line[i:].strip()
                break

        if not line:
            continue

        # Filter out preamble/filler lines the LLM sometimes adds
        # e.g. "Here is the numbered list of steps to accomplish the task:"
        if _is_preamble(line):
            continue

        steps.append(line)

    return steps if steps else [f"Complete the task: {task}"]


def _is_preamble(line: str) -> bool:
    """Check if a line is LLM preamble/filler rather than an actual step.

    The LLM sometimes adds introductory text like:
      "Here is the numbered list of concrete low-level steps:"
      "I'd be happy to help! Here are the steps:"

    Real steps start with action verbs: Open, Click, Type, Press, Wait, etc.
    """
    s = line.lower()

    # Known preamble patterns
    preamble_phrases = [
        "here is", "here are",
        "i'd be happy", "i'm happy",
        "the following", "below are",
        "numbered list", "low-level steps",
        "to accomplish", "could you please",
    ]
    if any(phrase in s for phrase in preamble_phrases):
        return True

    # Real steps typically start with these action verbs
    action_starts = [
        "open", "click", "type", "press", "wait", "launch",
        "scroll", "go ", "navigate", "search", "select",
        "close", "switch", "focus", "double",
    ]
    if any(s.startswith(verb) for verb in action_starts):
        return False

    # If it's a very long line (> 80 chars) and doesn't start with a verb,
    # it's probably a preamble sentence
    if len(line) > 80:
        return True

    return False


# ────────────────────────────────────────────────────────────
# Phase 2: Execution (vision model)
# ────────────────────────────────────────────────────────────

def decide_action_for_step(
    step: str,
    screenshot_bytes: bytes,
    screen_width: int,
    screen_height: int,
) -> dict:
    """Given a step and a screenshot, ask the vision model what to do.

    This is only called for "visual" steps like "Click the search box"
    where we need the model to look at the screenshot and find coordinates.

    Simple keyboard steps (Type, Press, Open) are handled directly in
    main.py without calling this function.

    Args:
      step: The current step text, e.g. "Click the YouTube search box"
      screenshot_bytes: Raw PNG bytes of the current screen
      screen_width: Logical screen width (for coordinate context)
      screen_height: Logical screen height

    Returns:
      A dict like {"action": "click", "x": 500, "y": 300, "reason": "..."}
    """
    # Build the prompt with the current step and screen dimensions
    prompt = EXECUTOR_PROMPT.format(
        current_step=step,
        width=screen_width,
        height=screen_height,
    )

    # Send screenshot + prompt to the vision model
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f'Look at the screenshot. Perform this step: "{step}"\n'
                    f'Respond with ONLY a JSON action.'
                ),
                "images": [screenshot_bytes],
            },
        ],
        options={"temperature": 0.1},
    )

    # Parse the JSON action from the model's text response
    raw = response["message"]["content"]
    return _parse_action(raw)


# ────────────────────────────────────────────────────────────
# Response parsing
# ────────────────────────────────────────────────────────────

def _parse_action(text: str) -> dict:
    """Extract a JSON action object from the model's response text.

    The model should return pure JSON like:
      {"action": "click", "x": 500, "y": 300, "reason": "search box"}

    But sometimes it wraps it in markdown code blocks or adds extra text.
    This function handles those cases gracefully.
    """
    text = text.strip()

    # Strip markdown code block wrappers if present
    # e.g. ```json\n{...}\n```
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try parsing as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting the first JSON object from the text
    # (model sometimes adds text before/after the JSON)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Nothing worked — return an error action
    return {"action": "error", "reason": f"Could not parse: {text[:200]}"}
