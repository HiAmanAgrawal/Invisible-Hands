"""Vision-model clicker — wraps the existing executor in the Clicker protocol.

This is the universal fallback at the bottom of the chain. It works on any
UI the model can visually understand (web pages with custom rendering,
games, design tools), at the cost of being the slowest and least accurate
strategy.

Returned alongside the (x, y) coordinates is the full LLM call result
(thinking text, raw response, token usage, ...) so the agent's reporter can
log and persist it the same way as any other LLM call.
"""

from __future__ import annotations

from invisible_hands.clickers.base import Clicker, ClickRequest, ClickResult
from invisible_hands.controllers.screen import screenshot_to_bytes
from invisible_hands.llm import executor


class VisionClicker(Clicker):
    """Use the vision model to decide where to click.

    Unlike the OCR / native-UI clickers, this one returns BOTH a click
    coordinate AND any other action the model might have produced (type,
    press, hotkey, scroll, wait, done, error). The agent peels those out
    of the returned ClickResult.extra dict so they don't get lost.
    """

    name = "vision"

    def find(self, request: ClickRequest) -> ClickResult | None:
        screen_w, screen_h = request.screen_size
        raw_bytes = screenshot_to_bytes(request.screenshot)

        result = executor.decide_action_for_step(
            request.step, raw_bytes, screen_w, screen_h,
        )
        action = result["action"]
        action_type = action.get("action")

        # Only click-style actions become an "x, y" hit. Everything else
        # (type/press/hotkey/scroll/wait/done/error) is forwarded as-is via
        # `extra["action"]` so the agent can execute it without re-asking
        # the model.
        if action_type in ("click", "double_click", "right_click"):
            x = int(action.get("x", 0))
            y = int(action.get("y", 0))
            return ClickResult(
                x=x,
                y=y,
                source="vision",
                confidence=0.5,  # Vision predictions are inherently fuzzy.
                evidence=action.get("reason", "vision model click"),
                extra={"llm_call": result, "action": action},
            )

        # Non-click action: still return a result so the chain stops here,
        # but mark the coordinates as invalid (-1, -1) and let the agent
        # use extra["action"] instead.
        return ClickResult(
            x=-1, y=-1,
            source="vision",
            confidence=0.5,
            evidence=action.get("reason", f"vision returned {action_type}"),
            extra={"llm_call": result, "action": action},
        )
