"""Clicker chain — try each strategy in order, return the first hit.

Why a chain?
    Different strategies have different strengths. OCR is cheap and great
    for visible text. Native UI is pixel-perfect but only works for native
    controls. Vision is universal but slow. Stacking them in order gives
    you the best of all three: fast and accurate when possible, slow but
    universal as a backstop.

Order matters:
    OCR (cheap)      -> first
    Native UI (cheap, very accurate when applicable) -> second
    Vision (expensive but universal) -> last

Each clicker can return None to defer to the next. The first non-None
result wins. If everything misses we return a sentinel ClickResult with
source='none' so the agent knows to retry or give up.

Disabled clickers (via config flags or runtime is_available()) are simply
skipped — they're never even called.
"""

from __future__ import annotations

import time

from invisible_hands.clickers.base import Clicker, ClickRequest, ClickResult


class ClickerChain:
    """Run a list of clickers in priority order until one returns a hit.

    The chain owns the timing: each clicker just returns "found / not found"
    without worrying about timestamps. We fill in `duration_s` on every
    result, including misses, so the reporter can show the cost of each
    strategy attempt.
    """

    def __init__(self, clickers: list[Clicker | None]):
        # Drop None entries (the agent passes None for disabled strategies)
        # AND drop unavailable ones (e.g. Tesseract not installed).
        self.clickers: list[Clicker] = [
            c for c in clickers
            if c is not None and c.is_available()
        ]

    def find(self, request: ClickRequest) -> tuple[ClickResult, list[dict]]:
        """Try each clicker; return (winning_result, attempt_log).

        attempt_log is a list of dicts describing every attempt, in order:
            {"name": "ocr", "found": False, "duration_s": 0.34, ...}
        Useful for the per-step report so you can see "OCR missed in 0.3s,
        Vision succeeded in 4.1s with confidence 0.5".
        """
        attempts: list[dict] = []

        for clicker in self.clickers:
            t0 = time.time()
            try:
                result = clicker.find(request)
            except Exception as e:
                attempts.append({
                    "name": clicker.name,
                    "found": False,
                    "error": str(e),
                    "duration_s": round(time.time() - t0, 3),
                })
                continue

            elapsed = round(time.time() - t0, 3)
            if result is None:
                attempts.append({
                    "name": clicker.name,
                    "found": False,
                    "duration_s": elapsed,
                })
                continue

            result.duration_s = elapsed
            attempts.append({
                "name": clicker.name,
                "found": True,
                "x": result.x,
                "y": result.y,
                "confidence": result.confidence,
                "evidence": result.evidence,
                "duration_s": elapsed,
            })
            return result, attempts

        # Nothing matched — return an explicit miss.
        miss = ClickResult(
            x=-1, y=-1, source="none", confidence=0.0,
            evidence="all clickers missed",
        )
        return miss, attempts

    @property
    def names(self) -> list[str]:
        """Names of clickers actually wired in (after availability filtering)."""
        return [c.name for c in self.clickers]
