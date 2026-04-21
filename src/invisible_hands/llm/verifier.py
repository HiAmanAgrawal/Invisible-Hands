"""Phase 3 of every step: did it actually work?

After we execute an action we take a fresh screenshot and ask the vision
model to look at it and say "yes, the step is done" or "no, here's what went
wrong". The reply is a small JSON object the agent loop uses to decide
whether to retry.

This is independent of the executor — verification is purely about reading
the *result* of an action, not deciding the *next* action — so it lives in
its own module with its own prompt.
"""

from __future__ import annotations

import json

from invisible_hands import config
from invisible_hands.llm import client
from invisible_hands.parsing.actions import strip_thinking


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
    """Ask the vision model whether a step succeeded.

    Args:
        step:             Original plan step text.
        action_summary:   Human-readable one-liner of what we executed,
                          e.g. 'click (812, 412)' or 'type "youtube.com"'.
        screenshot_bytes: Raw PNG bytes of the screenshot we captured AFTER
                          executing the action.

    Returns:
        {
            "result":       parsed JSON dict from the model
                            ({verified, confidence, observation, ...}),
            "raw_response": full model reply text,
            "model":        LM Studio model id used,
            "duration_s":   seconds for the chat call,
        }

    If the model returns un-parseable JSON we default to "verified: true" so
    the agent can keep moving rather than getting stuck on a flaky verifier.
    """
    prompt = VERIFY_PROMPT.format(step=step, action_summary=action_summary)
    data_url = client.encode_image_data_url(screenshot_bytes)

    response = client.chat(
        model=config.VISION_MODEL,
        system_prompt=prompt,
        user_input=[
            {"type": "text",
             "content": "Look at this screenshot taken after the step. "
                        "Was it completed successfully?"},
            {"type": "image", "data_url": data_url},
        ],
        temperature=0.1,
    )

    raw = response["text"]
    cleaned = strip_thinking(raw).strip()

    result: dict = {
        "verified": True,
        "confidence": "low",
        "observation": "Could not parse verification",
    }
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to recover the JSON object from anywhere in the reply.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

    return {
        "result": result,
        "raw_response": raw,
        "model": config.VISION_MODEL,
        "duration_s": response["duration_s"],
    }
