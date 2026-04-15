"""
Brain module — the "thinking" part of the AI agent.

Connects to LM Studio (local LLM server with OpenAI-compatible API) and uses two models:
  1. qwen/qwen3.5-9b (thinking model) — plans tasks into steps with chain-of-thought
     "Break this task into steps: open youtube in chrome"
     → ["Open Google Chrome", "Wait for Chrome to load", ...]

  2. qwen2.5-vl-3b-instruct (vision model) — for EXECUTING visual steps
     Given a screenshot + step like "Click the first video thumbnail",
     it returns {"action": "click", "x": 500, "y": 300}

The thinking model reasons through the plan step-by-step, while the
vision model handles the only thing that truly needs vision: finding
click targets on screen. Keyboard shortcuts are preferred everywhere else.
"""

from __future__ import annotations

import base64
import json
import re
import time

from openai import OpenAI

# LM Studio runs an OpenAI-compatible server on localhost:1234
_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Vision model — looks at screenshots and decides where to click
VISION_MODEL = "qwen2.5-vl-3b-instruct"

# Thinking model — plans the task into steps (chain-of-thought reasoning)
THINKING_MODEL = "qwen/qwen3.5-9b"

# ────────────────────────────────────────────────────────────
# System prompts
# ────────────────────────────────────────────────────────────

# PLANNER_PROMPT tells the thinking model how to break a task into steps.
# Key design decisions:
#   - "Open <AppName>" instead of Spotlight (we use `open -a` under the hood)
#   - Keyboard shortcuts preferred over clicking (e.g. "/" on YouTube, Cmd+L in Chrome)
#   - Each step is ONE action (no combining "open app and type url")
# ────────────────────────────────────────────────────────────
# System prompts (optimized for LM Studio models)
# ────────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are a macOS desktop automation planner.
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
- "Play the video" when a video page is open → use: Pr/ess k

Example task: "search for lo-fi music on YouTube and play a video"
1. Open Google Chrome
2. Wait for Google Chrome to load
3. Press Command+L
4. Type youtube.com/results?search_query=lo-fi+music
5. Press Enter
6. Wait for page to load
7. Click the first video thumbnail"""




# EXECUTOR_PROMPT tells the vision model how to look at a screenshot
# and decide what action to take for a specific step.
#
# The screenshot has COORDINATE AXES drawn on it — numbers along the
# top (X-axis) and left (Y-axis) edges, with light gridlines.
# The model reads these axis labels to estimate x,y coordinates.
EXECUTOR_PROMPT = """You are a desktop automation agent controlling a macOS computer.
You will see a screenshot with a coordinate grid. Your job is to perform ONE specific step.

Current step: {current_step}
Screen size: {width} x {height} pixels.

The screenshot has a coordinate grid overlaid on it:
- Numbers along the TOP edge are X coordinates (left=0, right={width})
- Numbers along the LEFT edge are Y coordinates (top=0, bottom={height})
- Green grid lines mark every 100 pixels

THINK STEP BY STEP before outputting coordinates:
1. SCAN: What major UI elements do you see? (browser window, search bar, buttons, videos, etc.)
2. LOCATE: Which specific element matches the current step?
3. MEASURE: Look at the grid lines closest to that element.
   - Find the X number on the TOP edge directly above the element's center.
   - Find the Y number on the LEFT edge directly beside the element's center.
   - If between two gridlines, interpolate (e.g., 1/3 of the way from 300 to 400 = ~333).
4. VERIFY: Is the coordinate within the element's visible bounds? Adjust if needed.

OUTPUT FORMAT — CRITICAL:
Your response must contain ONLY a JSON object on the LAST line.
Before the JSON, you may write brief reasoning (2-3 lines max).
The JSON must NOT be wrapped in code blocks or markdown.

DECISION RULES:
- "Click <something>": find that element, read its coordinates, output a click action.
- "Type <text>": output a type action with that exact text.
- "Press <key>": output a press action with that key.
- "Press <key1>+<key2>": output a hotkey action.
- "Wait": output a wait action.
- "Scroll down": output a scroll action.
- If the step is already visibly complete on screen: output a done action.

AVAILABLE ACTIONS (output exactly one):
{{"action": "click", "x": <int>, "y": <int>, "reason": "<what you see at that location>"}}
{{"action": "double_click", "x": <int>, "y": <int>, "reason": "<why>"}}
{{"action": "type", "text": "<exact text>", "reason": "<why>"}}
{{"action": "press", "key": "<key>", "reason": "<why>"}}
{{"action": "hotkey", "keys": ["<key1>", "<key2>"], "reason": "<why>"}}
{{"action": "scroll", "direction": "up|down", "amount": <1-5>, "reason": "<why>"}}
{{"action": "wait", "seconds": <1-3>, "reason": "<why>"}}
{{"action": "done", "reason": "<why step is complete>"}}

COORDINATE CONSTRAINTS:
- x must be between 30 and {width} (avoid the left axis overlay)
- y must be between 20 and {height} (avoid the top axis overlay)
- Click the CENTER of the target element, not its edge

EXAMPLE — step: "Click the first video thumbnail"
I see a grid of video thumbnails. The first one starts around x=200 and spans to x=500. Its vertical center is near the 300 gridline.
{{"action": "click", "x": 350, "y": 300, "reason": "First video thumbnail in the results grid"}}"""


# ────────────────────────────────────────────────────────────
# Thinking tag handling
# ────────────────────────────────────────────────────────────

def _extract_thinking(text: str) -> str | None:
    """Extract the content inside <think>...</think> tags.
    Returns None if no thinking tags are present."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks that thinking models emit.
    Returns only the content after the closing </think> tag,
    or the original text if no thinking tags are present."""
    pattern = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
    stripped = pattern.sub("", text).strip()
    return stripped if stripped else text


# ────────────────────────────────────────────────────────────
# Phase 1: Planning
# ────────────────────────────────────────────────────────────

def create_plan(task: str) -> dict:
    """Ask the thinking model to break the task into concrete steps.

    Returns a dict with:
      steps:        list[str]  — the parsed step list
      thinking:     str|None   — the model's <think> reasoning (if any)
      raw_response: str        — the full raw model output
      prompt_sent:  str        — the user message sent to the model
      model:        str        — model name used
      duration_s:   float      — wall-clock seconds for the LLM call
      usage:        dict|None  — token usage from the API (if available)
    """
    user_msg = f"Task: {task}"

    t0 = time.time()
    response = _client.chat.completions.create(
        model=THINKING_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    duration = time.time() - t0

    raw = response.choices[0].message.content.strip()

    # Extract thinking content before stripping it
    thinking = _extract_thinking(raw)
    cleaned = _strip_thinking_tags(raw)

    # Parse token usage if available
    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    # Parse the numbered list from the model's response
    steps = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line:
            continue

        for i, ch in enumerate(line):
            if not ch.isdigit() and ch not in ".):- ":
                line = line[i:].strip()
                break

        if not line:
            continue

        if _is_preamble(line):
            continue

        steps.append(line)

    if not steps:
        steps = [f"Complete the task: {task}"]

    return {
        "steps": steps,
        "thinking": thinking,
        "raw_response": raw,
        "prompt_sent": user_msg,
        "model": THINKING_MODEL,
        "duration_s": round(duration, 2),
        "usage": usage,
    }


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
    """Given a step and an annotated screenshot, ask the vision model what to do.

    Returns a dict with:
      action:        dict       — the parsed action (click, type, etc.)
      thinking:      str|None   — model's reasoning text before the JSON
      raw_response:  str        — full raw model output
      prompt_sent:   str        — the text prompt sent (image excluded)
      model:         str        — model name used
      duration_s:    float      — wall-clock seconds for the LLM call
      usage:         dict|None  — token usage
    """
    system_prompt = EXECUTOR_PROMPT.format(
        current_step=step,
        width=screen_width,
        height=screen_height,
    )

    user_text = (
        f'Look at the screenshot with coordinate axes.\n'
        f'Step to perform: "{step}"\n\n'
        f'Think step by step:\n'
        f'1. What UI elements do you see?\n'
        f'2. Which element matches this step?\n'
        f'3. What are the X,Y grid numbers nearest to that element?\n'
        f'Then output the JSON action on the last line.'
    )

    img_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    t0 = time.time()
    response = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            },
        ],
        temperature=0.1,
    )
    duration = time.time() - t0

    raw = response.choices[0].message.content
    thinking = _extract_thinking(raw)
    action = _parse_action(raw)

    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "action": action,
        "thinking": thinking,
        "raw_response": raw,
        "prompt_sent": user_text,
        "model": VISION_MODEL,
        "duration_s": round(duration, 2),
        "usage": usage,
    }


# ────────────────────────────────────────────────────────────
# Phase 3: Verification (vision model)
# ────────────────────────────────────────────────────────────

VERIFY_PROMPT = """You are verifying whether a desktop automation step was executed correctly.
You will see a screenshot taken AFTER the step was attempted.

Step that was attempted: {step}
Action that was executed: {action_summary}

Look at the screenshot and determine if the step was completed successfully.

Respond with ONLY a JSON object:
{{"verified": true, "confidence": "high|medium|low", "observation": "<what you see that confirms or denies success>"}}

OR if the step clearly failed:
{{"verified": false, "confidence": "high|medium|low", "observation": "<what went wrong>", "suggestion": "<what to try instead>"}}"""


def verify_step_completion(
    step: str,
    action_summary: str,
    screenshot_bytes: bytes,
) -> dict:
    """Ask the vision model to verify whether a step was completed successfully.

    Returns a dict with:
      result:       dict    — {verified, confidence, observation, suggestion?}
      raw_response: str     — full raw model output
      model:        str     — model name used
      duration_s:   float   — wall-clock seconds
    """
    prompt = VERIFY_PROMPT.format(step=step, action_summary=action_summary)
    img_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    t0 = time.time()
    response = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Look at this screenshot taken after the step. Was it completed successfully?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            },
        ],
        temperature=0.1,
    )
    duration = time.time() - t0

    raw = response.choices[0].message.content
    cleaned = _strip_thinking_tags(raw).strip()

    # Parse the verification JSON
    result = {"verified": True, "confidence": "low", "observation": "Could not parse verification"}
    try:
        parsed = json.loads(cleaned)
        result = parsed
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                result = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

    return {
        "result": result,
        "raw_response": raw,
        "model": VISION_MODEL,
        "duration_s": round(duration, 2),
    }


# ────────────────────────────────────────────────────────────
# Response parsing
# ────────────────────────────────────────────────────────────

def _parse_action(text: str) -> dict:
    """Extract a JSON action object from the model's response text.

    The model is instructed to output reasoning text followed by a JSON object
    on the last line. This parser handles multiple formats gracefully:
      - Pure JSON
      - JSON preceded by thinking/reasoning text or <think> blocks
      - JSON wrapped in markdown code blocks
      - JSON embedded anywhere in the response
    """
    text = _strip_thinking_tags(text).strip()

    # Strip markdown code block wrappers if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try parsing as-is first (pure JSON response)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try the last line first (thinking models put JSON at the end)
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    # Try extracting the last JSON object from the text
    last_brace = text.rfind("}")
    if last_brace != -1:
        search_region = text[:last_brace + 1]
        depth = 0
        start = -1
        for i in range(last_brace, -1, -1):
            if search_region[i] == "}":
                depth += 1
            elif search_region[i] == "{":
                depth -= 1
                if depth == 0:
                    start = i
                    break
        if start != -1:
            try:
                return json.loads(search_region[start:last_brace + 1])
            except json.JSONDecodeError:
                pass

    # Fallback: find first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {"action": "error", "reason": f"Could not parse: {text[:200]}"}
