"""Extract a structured action dict from a vision-model's free-form reply.

The vision model is asked to write a short reasoning sentence followed by a
single JSON object on the last line, e.g.

    The Sign In button is at the top right around (1240, 80).
    {"action": "click", "x": 1240, "y": 80, "reason": "Sign In button"}

In practice models do creative things: wrap the JSON in markdown fences, add
trailing commentary, drop the quotes around keys, embed the JSON mid-paragraph,
or emit a `<think>...</think>` reasoning block before everything else.

This module makes a best-effort attempt to recover a valid action dict from
all of those failure modes. If it really can't parse anything we return an
explicit `{"action": "error", "reason": "..."}` so the caller can retry.
"""

from __future__ import annotations

import json
import re


# Reasoning-tagged thinking models (Qwen, DeepSeek, etc.) emit chain-of-thought
# inside `<think>...</think>` blocks before their actual answer. We strip those
# out so they don't confuse the JSON extractor.
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_BLOCK_STRIP_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def extract_thinking(text: str) -> str | None:
    """Return the contents of the model's <think>...</think> block, or None."""
    match = _THINK_BLOCK_RE.search(text)
    return match.group(1).strip() if match else None


def strip_thinking(text: str) -> str:
    """Return `text` with any <think>...</think> blocks removed.

    If the result is empty (the model's whole reply was thinking with no
    answer) we return the original text so the caller has something to look
    at, even if it's just chain-of-thought."""
    stripped = _THINK_BLOCK_STRIP_RE.sub("", text).strip()
    return stripped if stripped else text


def _fix_unquoted_keys(s: str) -> str:
    """Add missing quotes around object keys: `{action: ...}` -> `{"action": ...}`.

    Some smaller models output JSON-ish text where the keys aren't quoted,
    which is technically not JSON. The regex matches a word followed by a
    colon, immediately after `{` or `,`, and wraps it in double quotes.
    """
    return re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', s)


def _try_parse_json(s: str) -> dict | None:
    """Parse `s` as JSON. If that fails, retry after fixing unquoted keys."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_fix_unquoted_keys(s))
    except json.JSONDecodeError:
        return None


def parse_action(text: str) -> dict:
    """Pull a JSON action object out of `text`. Always returns a dict.

    Strategy (each step falls through to the next on failure):
        1. Strip any <think>...</think> blocks.
        2. Drop a markdown ```...``` fence if the whole reply is wrapped in one.
        3. Try to parse the cleaned text directly as JSON.
        4. Walk the lines from the bottom up, parsing each `{...}` line.
        5. From the LAST closing brace, walk back to find a balanced `{...}`.
        6. As a last resort, take everything between the first `{` and the
           last `}` and try again.
        7. Give up and return an `error` action with a snippet for debugging.
    """
    text = strip_thinking(text).strip()

    if text.startswith("```"):
        cleaned_lines = [
            line for line in text.split("\n")
            if not line.strip().startswith("```")
        ]
        text = "\n".join(cleaned_lines).strip()

    result = _try_parse_json(text)
    if result is not None:
        return result

    for line in reversed(text.split("\n")):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            result = _try_parse_json(line)
            if result is not None:
                return result

    last_brace = text.rfind("}")
    if last_brace != -1:
        depth = 0
        start = -1
        for i in range(last_brace, -1, -1):
            ch = text[i]
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    start = i
                    break
        if start != -1:
            result = _try_parse_json(text[start:last_brace + 1])
            if result is not None:
                return result

    first_brace = text.find("{")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        result = _try_parse_json(text[first_brace:last_brace + 1])
        if result is not None:
            return result

    return {"action": "error", "reason": f"Could not parse: {text[:200]}"}
