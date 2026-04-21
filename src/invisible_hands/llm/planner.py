"""Phase 1 of every run: turn the user's task into a numbered step plan.

The planner uses the larger thinking model (qwen3.5-9b by default). We
explicitly disable its `<think>` reasoning channel — the planner prompt is
strict enough that chain-of-thought hurts more than it helps, and the
reasoning tokens were eating up the entire output budget on the first pass.

Output of this module is a list of strings like:
    ["Open Google Chrome",
     "Wait for Google Chrome to load",
     "Press Command+L",
     "Type youtube.com",
     "Press Enter",
     "Wait for page to load",
     "Click the first video thumbnail"]
"""

from __future__ import annotations

from invisible_hands import config
from invisible_hands.llm import client
from invisible_hands.parsing.actions import extract_thinking, strip_thinking


PLANNER_PROMPT = """You are a desktop automation planner.
Given a task, output ONLY a numbered list of steps. No text before or after. No explanations.

STRICT RULES — follow exactly:
1. Each step is ONE single physical action. Never combine two actions into one step.
2. To open an app: "Open <AppName>". Always follow it immediately with: "Wait for <AppName> to load".
3. To navigate in a browser: "Press Command+L", then "Type <url>", then "Press Enter", then "Wait for page to load".
4. ALWAYS prefer keyboard shortcuts over clicking. Only use "Click" when there is no keyboard shortcut available.
5. After every "Type <query>" step, ALWAYS add: "Press Enter".
6. After every "Press Enter" that loads a new page, ALWAYS add: "Wait for page to load".
7. Never skip steps. Never assume an element is already focused.

KEYBOARD SHORTCUTS — use these INSTEAD of clicking:
Chrome / Browser:
  - Command+L → focus address bar (NEVER click the address bar)
  - Command+T → new tab
  - Command+W → close tab
  - Tab → move focus to next element
YouTube (after page loads):
  - k → play/pause video
  - f → fullscreen
  - j → rewind 10s, l → forward 10s
Spotify:
  - Command+K → open search
General:
  - Enter → confirm / select focused item
  - Escape → close dialog / cancel
  - Tab → move between UI elements

ALLOWED step formats (use ONLY these):
- Open <AppName>
- Wait for <AppName> to load
- Wait for page to load
- Press <key or shortcut>
- Click <element description>
- Type <text>
- Scroll down

WRONG (never write these):
- "Navigate to X" → use: Press Command+L, Type "X", Press Enter, Wait for page to load
- "Search for X on YouTube" → use: Press /, Type "X", Press Enter
- "Click the address bar" → use: Press Command+L
- "Play the video" when a video page is open → use: Press k

Example task: "search for lo-fi music on YouTube and play a video"
1. Open Google Chrome
2. Wait for Google Chrome to load
3. Press Command+L
4. Type youtube.com/results?search_query=lo-fi+music
5. Press Enter
6. Wait for page to load
7. Click the first video thumbnail"""


def _is_preamble(line: str) -> bool:
    """True if the line looks like LLM intro text rather than an actual step.

    Models occasionally insert "Here is the numbered list of steps:" or
    "I'd be happy to help!" before the actual list. We detect those by:
        - Known intro phrases.
        - Lines that don't start with one of the action verbs the prompt
          tells the model to use AND are unusually long (> 80 chars).
    """
    s = line.lower()

    preamble_phrases = (
        "here is", "here are",
        "i'd be happy", "i'm happy",
        "the following", "below are",
        "numbered list", "low-level steps",
        "to accomplish", "could you please",
    )
    if any(phrase in s for phrase in preamble_phrases):
        return True

    action_starts = (
        "open", "click", "type", "press", "wait", "launch",
        "scroll", "go ", "navigate", "search", "select",
        "close", "switch", "focus", "double",
    )
    if any(s.startswith(verb) for verb in action_starts):
        return False

    return len(line) > 80


def create_plan(task: str) -> dict:
    """Ask the thinking model to break `task` into a numbered list of steps.

    Returns:
        {
            "steps":       [...],     # parsed step strings (always at least 1)
            "thinking":    "..." | None,
            "raw_response": "...",   # full text the model returned
            "prompt_sent":  "...",   # what we sent as the user message
            "model":        "...",
            "duration_s":   1.23,
            "usage":        {...} | None,
        }
    """
    user_msg = f"Task: {task}"

    response = client.chat(
        model=config.THINKING_MODEL,
        system_prompt=PLANNER_PROMPT,
        user_input=user_msg,
        temperature=0.1,
        reasoning="off",        # Planner's prompt is strict — no need to think aloud.
        max_output_tokens=400,  # Plans should be short; cap so a runaway model can't hang us.
    )

    raw = response["text"]
    thinking = extract_thinking(raw)
    cleaned = strip_thinking(raw)

    steps: list[str] = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Strip leading list markers: "1.", "1)", "- ", etc.
        for i, ch in enumerate(line):
            if not ch.isdigit() and ch not in ".):- ":
                line = line[i:].strip()
                break

        if not line or _is_preamble(line):
            continue

        steps.append(line)

    if not steps:
        # Defensive fallback: never let the rest of the agent see an empty plan.
        steps = [f"Complete the task: {task}"]

    return {
        "steps": steps,
        "thinking": thinking,
        "raw_response": raw,
        "prompt_sent": user_msg,
        "model": config.THINKING_MODEL,
        "duration_s": response["duration_s"],
        "usage": response["usage"],
    }
