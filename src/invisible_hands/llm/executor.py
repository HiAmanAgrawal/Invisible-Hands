"""Phase 2 of every run: ask the vision model what to DO for one plan step.

This is the heart of the vision-based clicker. The flow is:

    1. Take the raw screenshot (full resolution).
    2. Resize it down to `VISION_MAX_WIDTH` so the grid we overlay is large
       and legible to a small model.
    3. Draw a labelled coordinate grid (red major lines every 100px, yellow
       numbers along the top + left edges).
    4. Send the resized + annotated image to the vision model along with the
       step text and a strict prompt.
    5. Parse a JSON action out of the model's reply.
    6. If the action is a click, scale its coordinates back up to the
       original screen resolution before returning.

The result of step 6 is what makes the whole resize trick safe: the model
operates on small grid numbers, but the agent acts on real screen pixels.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from invisible_hands import config
from invisible_hands.controllers.screen import annotate_screenshot_with_axes
from invisible_hands.llm import client
from invisible_hands.parsing.actions import extract_thinking, parse_action


EXECUTOR_PROMPT = """You are a desktop automation agent controlling a computer.
You see a screenshot with a coordinate grid overlay. Perform ONE specific step.

Current step: {current_step}
Screen size: {width} x {height} pixels.

COORDINATE GRID:
- TOP EDGE: yellow numbers = X coordinates (left=0 → right={width})
- LEFT EDGE: yellow numbers = Y coordinates (top=0 → bottom={height})
- RED lines with yellow labels appear every 100px

HOW TO READ COORDINATES:
1. Find the target element on screen.
2. Look STRAIGHT UP from its center to the top edge → read the nearest X number.
3. Look STRAIGHT LEFT from its center to the left edge → read the nearest Y number.
4. If between two red lines, estimate. Example: 1/3 between 200 and 300 = ~233.

OUTPUT FORMAT:
Write 1-2 lines of reasoning, then a single JSON object on the LAST line.
No markdown, no code blocks, no extra text after the JSON.

DECISION RULES:
- "Click <element>": locate it, read coordinates, output click.
- "Type <text>": output type action.
- "Press <key>": output press action.
- "Press <key1>+<key2>": output hotkey action.
- "Wait": output wait action.
- "Scroll down/up": output scroll action.
- If step is already complete on screen: output done action.

ACTIONS:
{{"action": "click", "x": <int>, "y": <int>, "reason": "<what is at that location>"}}
{{"action": "double_click", "x": <int>, "y": <int>, "reason": "<why>"}}
{{"action": "type", "text": "<exact text>", "reason": "<why>"}}
{{"action": "press", "key": "<key>", "reason": "<why>"}}
{{"action": "hotkey", "keys": ["<key1>", "<key2>"], "reason": "<why>"}}
{{"action": "scroll", "direction": "up|down", "amount": <1-5>, "reason": "<why>"}}
{{"action": "wait", "seconds": <1-3>, "reason": "<why>"}}
{{"action": "done", "reason": "<why step is already complete>"}}

CONSTRAINTS:
- x must be between 40 and {width}
- y must be between 25 and {height}
- Always click the CENTER of the element, not its edge or corner
- For video thumbnails: click the middle of the thumbnail image itself"""


def decide_action_for_step(
    step: str,
    screenshot_bytes: bytes,
    screen_width: int,
    screen_height: int,
) -> dict:
    """Ask the vision model what action to take for one step.

    Args:
        step:             The plan step text, e.g. 'Click the first video thumbnail'.
        screenshot_bytes: Raw PNG bytes of the FULL-resolution screenshot.
                          We resize internally; the caller doesn't have to.
        screen_width:     Logical screen width (matches pyautogui.size()).
        screen_height:    Logical screen height.

    Returns:
        {
            "action":       parsed action dict (click/type/press/...),
            "thinking":     model chain-of-thought, if any,
            "raw_response": full model reply text,
            "prompt_sent":  the user-side text we sent (image not included),
            "model":        the LM Studio model id used,
            "duration_s":   seconds for the chat call,
            "usage":        token usage stats or None,
        }
    """
    original = Image.open(BytesIO(screenshot_bytes))

    # Resize so grid numbers are big enough for small VLMs to read.
    scale = 1.0
    if original.width > config.VISION_MAX_WIDTH:
        scale = original.width / config.VISION_MAX_WIDTH
        new_h = int(original.height / scale)
        resized = original.resize((config.VISION_MAX_WIDTH, new_h), Image.LANCZOS)
    else:
        resized = original

    # 100px ticks make labels uncluttered on the small image; we'll scale the
    # final coordinates back up before clicking.
    annotated = annotate_screenshot_with_axes(resized, tick_spacing=100)
    img_w, img_h = annotated.size

    system_prompt = EXECUTOR_PROMPT.format(
        current_step=step,
        width=img_w,
        height=img_h,
    )

    user_text = (
        f'Step to perform: "{step}"\n'
        f'The image is {img_w}x{img_h}. Read coordinates from the grid overlay. '
        f'Output JSON with x,y values matching the grid numbers in the image.'
    )

    buf = BytesIO()
    annotated.save(buf, format="PNG")
    data_url = client.encode_image_data_url(buf.getvalue())

    response = client.chat(
        model=config.VISION_MODEL,
        system_prompt=system_prompt,
        user_input=[
            {"type": "text", "content": user_text},
            {"type": "image", "data_url": data_url},
        ],
        temperature=0.1,
    )

    raw = response["text"]
    action = parse_action(raw)

    # Scale model coordinates (in resized-image space) back to real screen pixels.
    if scale > 1.0 and action.get("action") in ("click", "double_click", "right_click"):
        if "x" in action:
            action["x"] = int(action["x"] * scale)
        if "y" in action:
            action["y"] = int(action["y"] * scale)

    return {
        "action": action,
        "thinking": extract_thinking(raw),
        "raw_response": raw,
        "prompt_sent": user_text,
        "model": config.VISION_MODEL,
        "duration_s": response["duration_s"],
        "usage": response["usage"],
    }
