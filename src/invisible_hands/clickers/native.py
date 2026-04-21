"""Native UI automation clicker — query the OS accessibility tree.

Modern operating systems publish a tree of every UI element on screen
(window -> toolbar -> button -> ...) along with each element's role,
label, and pixel bounds. Screen readers use this tree to narrate UIs;
we use it to find the exact pixel rectangle of a control with a given
label, with no OCR or vision model needed.

Backends:
    macOS:    atomacos (preferred) or atomac wraps the AXUIElement APIs.
              atomacos is the maintained fork that supports modern macOS;
              atomac is kept as a fallback for older installations.
    Windows:  pywinauto with the UIA backend wraps Microsoft UI Automation.

Both backends are loaded LAZILY. If the relevant library isn't installed
we just disable the clicker on this platform — the chain skips it and
falls through to OCR / Vision. That means the package still installs and
runs cleanly on a system that's missing the optional native deps.

Why this is great for desktop apps (Finder, Word, Outlook, settings
panels, file dialogs, system menus): it's pixel-perfect, deterministic,
and zero-cost compared to anything LLM-based.

Why it doesn't always help for browsers / web pages: a browser's HTML
content gets exposed as a generic 'AXWebArea' / 'PaneControl' blob with
no individual button labels. For those, OCR or Vision wins again.
"""

from __future__ import annotations

import sys
from difflib import SequenceMatcher

from invisible_hands.clickers.base import Clicker, ClickRequest, ClickResult


def _similarity(a: str, b: str) -> float:
    """0..1 similarity score (case-insensitive). Same metric as the OCR clicker."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ─────────────────────────────────────────────────────────────────────────────
# macOS backend (atomac)
# ─────────────────────────────────────────────────────────────────────────────

class _MacBackend:
    """macOS Accessibility tree walker.

    Tries the maintained `atomacos` fork first, then falls back to the
    older `atomac` package. Both expose the same `NativeUIElement` API.

    Public:
        is_available() -> bool      True if either library imported.
        find(target)   -> tuple|None  (cx, cy, role, label) on a hit.
    """

    def __init__(self):
        self._atomac = None
        # Prefer the actively maintained fork.
        try:
            import atomacos as _atomac  # type: ignore

            self._atomac = _atomac
        except Exception:
            try:
                import atomac as _atomac  # type: ignore

                self._atomac = _atomac
            except Exception:
                self._atomac = None

    def is_available(self) -> bool:
        return self._atomac is not None

    def find(self, target: str) -> tuple[int, int, str, str] | None:
        if not self._atomac:
            return None
        atomac = self._atomac

        try:
            # The frontmost app exposes its UI tree to atomac. Walk it
            # looking for a clickable element whose label fuzzy-matches.
            front = atomac.NativeUIElement.getFrontmostApp()
        except Exception:
            return None

        best: tuple[float, int, int, str, str] | None = None

        def walk(element, depth: int = 0):
            nonlocal best
            # Hard cap recursion so a pathological tree can't hang us.
            if depth > 12:
                return
            try:
                role = str(getattr(element, "AXRole", "") or "")
                label = ""
                # Try several label-like attributes — different element kinds
                # expose their visible text in different places.
                for attr in ("AXTitle", "AXValue", "AXDescription", "AXHelp"):
                    val = getattr(element, attr, None)
                    if val:
                        label = str(val)
                        break

                if label and role in (
                    "AXButton", "AXMenuItem", "AXLink",
                    "AXCheckBox", "AXRadioButton", "AXTab",
                    "AXTextField", "AXSearchField", "AXPopUpButton",
                ):
                    score = _similarity(label, target)
                    if best is None or score > best[0]:
                        try:
                            position = element.AXPosition
                            size = element.AXSize
                            cx = int(position.x + size.width / 2)
                            cy = int(position.y + size.height / 2)
                            best = (score, cx, cy, role, label)
                        except Exception:
                            pass

                for child in getattr(element, "AXChildren", []) or []:
                    walk(child, depth + 1)
            except Exception:
                # Some children throw on attribute access — just skip them.
                pass

        try:
            walk(front)
        except Exception:
            return None

        if best is None or best[0] < 0.7:
            return None
        score, cx, cy, role, label = best
        return cx, cy, role, label


# ─────────────────────────────────────────────────────────────────────────────
# Windows backend (pywinauto + UIA)
# ─────────────────────────────────────────────────────────────────────────────

class _WindowsBackend:
    """Windows UI Automation tree walker via `pywinauto`.

    We talk to the foreground window only — pywinauto can attach to a
    process by name or PID, but doing so is slow and we only need the
    visible UI right now.
    """

    def __init__(self):
        self._app = None
        try:
            from pywinauto import Desktop  # type: ignore

            self._desktop = Desktop(backend="uia")
        except Exception:
            self._desktop = None

    def is_available(self) -> bool:
        return self._desktop is not None

    def find(self, target: str) -> tuple[int, int, str, str] | None:
        if not self._desktop:
            return None

        # Foreground window is the most likely place the user wants to click.
        try:
            top_window = self._desktop.top_window()
        except Exception:
            return None

        best: tuple[float, int, int, str, str] | None = None

        try:
            for ctrl in top_window.descendants():
                try:
                    label = ctrl.window_text() or ""
                    role = ctrl.element_info.control_type or ""
                except Exception:
                    continue
                if not label:
                    continue
                # Only consider control types that respond to clicks.
                if role not in (
                    "Button", "Hyperlink", "MenuItem", "TabItem",
                    "CheckBox", "RadioButton", "ListItem", "TreeItem",
                    "Edit", "ComboBox",
                ):
                    continue
                score = _similarity(label, target)
                if best is None or score > best[0]:
                    try:
                        rect = ctrl.rectangle()
                        cx = (rect.left + rect.right) // 2
                        cy = (rect.top + rect.bottom) // 2
                        best = (score, cx, cy, role, label)
                    except Exception:
                        continue
        except Exception:
            return None

        if best is None or best[0] < 0.7:
            return None
        score, cx, cy, role, label = best
        return cx, cy, role, label


# ─────────────────────────────────────────────────────────────────────────────
# Public clicker
# ─────────────────────────────────────────────────────────────────────────────

class NativeUIClicker(Clicker):
    """Resolve a click target by walking the OS accessibility tree.

    On unsupported platforms (or when the relevant library isn't installed)
    `is_available()` returns False and the chain skips this clicker
    silently, falling through to OCR or Vision."""

    name = "native_ui"

    def __init__(self):
        if sys.platform == "darwin":
            self._backend = _MacBackend()
        elif sys.platform == "win32":
            self._backend = _WindowsBackend()
        else:
            self._backend = None

    def is_available(self) -> bool:
        return self._backend is not None and self._backend.is_available()

    def find(self, request: ClickRequest) -> ClickResult | None:
        if not self.is_available():
            return None

        target = self.extract_text_target(request.step)
        if not target:
            return None

        try:
            hit = self._backend.find(target)
        except Exception:
            hit = None

        if not hit:
            return None

        cx, cy, role, label = hit
        return ClickResult(
            x=int(cx),
            y=int(cy),
            source="native_ui",
            confidence=0.95,  # OS-reported bounds are pixel-accurate.
            evidence=f'Native {role} labelled "{label}" at ({cx}, {cy})',
            extra={"target": target, "role": role, "label": label},
        )
