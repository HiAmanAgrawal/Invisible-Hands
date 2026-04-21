"""Tiny HTTP client for LM Studio's native chat API.

LM Studio exposes two compatible endpoints:
    /v1/chat/completions   OpenAI-compatible (works with the openai SDK)
    /api/v1/chat           LM Studio's native, slightly richer API

We use the native one because it surfaces useful extras like reasoning
tokens, time-to-first-token, and per-call stats that we want to log. The
endpoint accepts either a string or an array of typed message parts (text +
image), which is exactly what we need for the vision model.

Everything in this module is intentionally tiny — there are no retries, no
streaming, no backoff. If LM Studio is unreachable the agent should fail
loudly so the user can fix it; silently retrying for 60 seconds would be
worse.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from typing import Any

from invisible_hands import config


def encode_image_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    """Wrap raw image bytes in a base64 data: URL the chat API accepts."""
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def chat(
    *,
    model: str,
    system_prompt: str,
    user_input: str | list[dict[str, Any]],
    temperature: float = 0.1,
    reasoning: str | None = None,
    max_output_tokens: int | None = None,
    timeout: int = 120,
) -> dict:
    """Send one chat request to LM Studio and return a normalized result.

    Args:
        model:             LM Studio model id (must be loaded in the app).
        system_prompt:     System message that frames the model's behaviour.
        user_input:        Either a plain string, or an array of typed parts
                           like [{"type": "text", "content": "..."},
                                 {"type": "image", "data_url": "..."}].
        temperature:       Sampling temperature, 0.0 - 1.0. Low for
                           deterministic-ish behaviour.
        reasoning:         "off" / "low" / "medium" / "high" / "on", or None
                           to let LM Studio pick the model default. Only
                           supported by reasoning models (Qwen3 etc.).
        max_output_tokens: Hard cap on response length, or None for default.
        timeout:           Seconds before raising urllib.error.URLError.

    Returns a dict with:
        text:           the model's text reply (concatenation of any
                        type=="message" output items).
        raw_response:   the full text reply (same as `text`, kept for clarity
                        in logs).
        duration_s:     wall-clock seconds the call took.
        usage:          {prompt_tokens, completion_tokens, total_tokens,
                         tokens_per_second} or None if stats are missing.
        api:            the full decoded JSON response, in case the caller
                        needs more (rarely used).
    """
    payload: dict[str, Any] = {
        "model": model,
        "system_prompt": system_prompt,
        "input": user_input,
        "temperature": temperature,
    }
    if reasoning is not None:
        payload["reasoning"] = reasoning
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{config.LM_STUDIO_BASE}/api/v1/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        api_data = json.loads(resp.read())
    duration = time.time() - started

    text_parts: list[str] = []
    for item in api_data.get("output", []):
        if item.get("type") == "message":
            text_parts.append(item.get("content", ""))
    text = "".join(text_parts).strip()

    usage = None
    stats = api_data.get("stats")
    if stats:
        prompt_tokens = stats.get("input_tokens", 0)
        completion_tokens = stats.get("total_output_tokens", 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tokens_per_second": stats.get("tokens_per_second"),
        }

    return {
        "text": text,
        "raw_response": text,
        "duration_s": round(duration, 2),
        "usage": usage,
        "api": api_data,
    }
