"""OCR-based clicker — find text on the screen and click its center.

How it works:
    1. Pull the requested text label out of the step
       ('Click "Sign In"' -> 'Sign In').
    2. Run Tesseract OCR over the screenshot to get every visible word
       together with its bounding box.
    3. Scan the OCR output for a contiguous run of words whose joined text
       fuzzy-matches the target with a high enough score.
    4. Return the center pixel of that run's bounding box.

Why fuzzy matching?
    Tesseract makes small mistakes on UI text all the time: it'll read
    "Sign ln" instead of "Sign In", "Subm1t" instead of "Submit", or
    introduce extra spaces. SequenceMatcher gives a similarity score that
    tolerates those mistakes without blowing past visually-distinct labels.

Why this is great for buttons/links:
    Most clickable UI elements have visible text labels. When they do, this
    strategy is dramatically faster and more accurate than asking a vision
    model to reason about pixel coordinates — and it works on absolutely
    any platform that Tesseract supports (Mac, Windows, Linux).

Why it's not a complete solution:
    Icon-only buttons (the play triangle, hamburger menu, X close icon),
    image thumbnails, and graphical UI elements have no readable text. Those
    fall through to the next clicker in the chain.
"""

from __future__ import annotations

import shutil
from difflib import SequenceMatcher

from invisible_hands import config
from invisible_hands.clickers.base import Clicker, ClickRequest, ClickResult


def _similarity(a: str, b: str) -> float:
    """Return a 0..1 similarity score between two strings (case-insensitive)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _tesseract_available() -> bool:
    """True if both the `tesseract` CLI and `pytesseract` Python wrapper exist."""
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


class TesseractClicker(Clicker):
    """Click on visible text by running OCR over the screenshot.

    Example: 'Click the "Sign In" button' -> OCR finds words near the top
    right whose phrase is "Sign In", returns its center."""

    name = "ocr"

    def __init__(self, fuzzy_threshold: int | None = None):
        # Threshold is on a 0..100 scale so it's easy to set from an env var.
        # Stored as a 0..1 ratio internally for use with SequenceMatcher.
        threshold = fuzzy_threshold if fuzzy_threshold is not None else config.OCR_FUZZY_THRESHOLD
        self._min_similarity = max(0.0, min(1.0, threshold / 100.0))
        self._available = _tesseract_available()

    def is_available(self) -> bool:
        return self._available

    def find(self, request: ClickRequest) -> ClickResult | None:
        if not self._available:
            return None

        target = self.extract_text_target(request.step)
        if not target:
            # Step doesn't reference a text label at all — let the next
            # clicker handle it (e.g. icon clicks).
            return None

        words = self._ocr_words(request.screenshot)
        if not words:
            return None

        best = self._find_best_phrase(words, target)
        if not best:
            return None

        score, x, y, matched_text, line_words = best
        if score < self._min_similarity:
            return None

        return ClickResult(
            x=int(x),
            y=int(y),
            source="ocr",
            confidence=score,
            evidence=f'OCR matched "{matched_text}" (score {score:.2f}) for target "{target}"',
            extra={
                "target": target,
                "matched_text": matched_text,
                "matched_word_count": line_words,
            },
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ocr_words(image) -> list[dict]:
        """Run Tesseract and return one dict per detected word.

        Each word dict has:
            text:  the word string
            x, y:  top-left corner of the word's bounding box
            w, h:  bounding box dimensions
            conf:  Tesseract's own confidence (0..100, -1 if unknown)
            line:  Tesseract's line ID (used to keep multi-word matches on
                   the same UI line — we don't want to merge "Sign" from
                   one button with "In" from a different button).
        """
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:
            return []

        try:
            data = pytesseract.image_to_data(image, output_type=Output.DICT)
        except Exception:
            # Tesseract binary missing or image format issue — give up cleanly.
            return []

        words: list[dict] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            # Skip very low-confidence garbage (Tesseract reports -1 when it
            # didn't try to score the word; we keep those because they're
            # often legitimate tiny labels).
            if 0 <= conf < 30:
                continue
            words.append({
                "text": text,
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
                "conf": conf,
                "line": (
                    int(data["block_num"][i]),
                    int(data["par_num"][i]),
                    int(data["line_num"][i]),
                ),
            })
        return words

    def _find_best_phrase(
        self,
        words: list[dict],
        target: str,
    ) -> tuple[float, int, int, str, int] | None:
        """Slide a window across each line of OCR output and pick the best
        fuzzy match.

        Returns (score, center_x, center_y, matched_text, word_count) or None.

        Why a sliding window per line?
            Multi-word labels like "Sign In" or "Add to cart" appear as
            several adjacent OCR words. We only consider windows of words
            that share the same Tesseract line ID, so we never accidentally
            stitch together words from different buttons.
        """
        target_words = target.split()
        max_window = max(1, min(len(target_words) + 2, 6))

        best: tuple[float, int, int, str, int] | None = None

        # Group words by line, preserving original index order.
        from itertools import groupby

        for _, line_iter in groupby(words, key=lambda w: w["line"]):
            line = list(line_iter)
            for window_size in range(1, max_window + 1):
                for start in range(0, len(line) - window_size + 1):
                    window = line[start:start + window_size]
                    text = " ".join(w["text"] for w in window)
                    score = _similarity(text, target)
                    if best is None or score > best[0]:
                        x0 = min(w["x"] for w in window)
                        y0 = min(w["y"] for w in window)
                        x1 = max(w["x"] + w["w"] for w in window)
                        y1 = max(w["y"] + w["h"] for w in window)
                        cx = (x0 + x1) // 2
                        cy = (y0 + y1) // 2
                        best = (score, cx, cy, text, window_size)

        return best
