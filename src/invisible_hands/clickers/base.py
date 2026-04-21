"""Shared types + abstract base class for every click strategy.

Every concrete clicker implements one method:

    find(request) -> ClickResult | None

Returning None means "I can't help with this one, ask the next strategy."
Returning a ClickResult means "click here, I'm reasonably confident."

Keeping the interface this small means new strategies (e.g. template
matching, web-driver injection) can be plugged into the chain with zero
changes to the agent loop.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class ClickRequest:
    """Everything a clicker needs to make a decision about ONE plan step.

    Attributes:
        step:         The original plan step text, e.g. 'Click the Sign In button'.
                      Each clicker can extract its own target from this.
        screenshot:   PIL screenshot at full LOGICAL screen resolution.
        screen_size:  (width, height) in logical pixels, matches the screenshot.
        app_context:  Best guess at the foreground app, if known. Currently
                      reserved for future use (e.g. limiting native-UI search
                      to the active app's window tree).
    """
    step: str
    screenshot: "Image.Image"
    screen_size: tuple[int, int]
    app_context: str | None = None


@dataclass
class ClickResult:
    """The output of a successful clicker.

    Attributes:
        x, y:       Coordinates to click, in logical screen pixels.
        source:     Which strategy produced this result. Used in logs/reports
                    so you can tell at a glance which path is succeeding.
                    One of: 'ocr', 'native_ui', 'vision', 'none'.
        confidence: Self-reported confidence score in [0.0, 1.0]. The chain
                    doesn't filter on this today, but it's logged so you can
                    look at low-confidence wins in the report later.
        evidence:   Human-readable explanation, e.g. 'matched "Sign In" with
                    score 92%' or 'AXButton at (812, 412)'.
        duration_s: Seconds the strategy took to produce the result. Filled
                    in by the chain (clickers don't have to time themselves).
        extra:      Optional bag of strategy-specific debug info (token
                    counts for vision, OCR confidence per word, etc.).
    """
    x: int
    y: int
    source: str
    confidence: float = 0.0
    evidence: str = ""
    duration_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        """True if this result represents an actual hit (not a fallthrough)."""
        return self.source != "none"


class Clicker(ABC):
    """Abstract base class — every click strategy implements this.

    Subclasses must set a `name` class attribute and implement `find`.
    The `is_available` hook lets a clicker disable itself at runtime (e.g.
    Tesseract not installed, atomac not importable on this Python build)
    so the chain can skip it without raising.
    """

    name: str = "abstract"

    def is_available(self) -> bool:
        """Return True if this strategy can actually run on the current system.
        Default: always available. Override when there are runtime dependencies."""
        return True

    @abstractmethod
    def find(self, request: ClickRequest) -> ClickResult | None:
        """Try to locate the click target. Return a ClickResult on success
        or None to defer to the next strategy in the chain."""

    def extract_text_target(self, step: str) -> str | None:
        """Best-effort extract of a text label from a 'Click X' step.

        Examples and what they extract:
            'Click the "Sign In" button'  -> 'Sign In'
            'Click "Submit"'              -> 'Submit'
            'Click the Sign In button'    -> 'Sign In'
            'Click on Cancel'             -> 'Cancel'
            'Click the first video'       -> 'first video'
            'Click the play button'       -> 'play'

        We strip noise words ("the", "on", "button", "icon", "link") because
        they hurt OCR/UI matching more often than they help. Returns None
        if the step doesn't look like a click-by-text request.
        """
        s = step.strip()
        if not re.match(r"(?i)^(click|tap|select|press)\b", s):
            return None

        # Prefer quoted text if the planner included quotes.
        quoted = re.search(r'["\u201c](.+?)["\u201d]', s)
        if quoted:
            return quoted.group(1).strip()

        # Strip leading verb + filler words.
        cleaned = re.sub(
            r"(?i)^(click|tap|select|press)\s+(on\s+|the\s+|a\s+|an\s+)?",
            "", s,
        )
        # Strip trailing UI-noun noise.
        cleaned = re.sub(
            r"(?i)\s+(button|icon|link|tab|menu|item|element|control)$",
            "", cleaned,
        )
        cleaned = cleaned.strip(" .'\"\u201c\u201d\u2018\u2019")
        return cleaned or None
