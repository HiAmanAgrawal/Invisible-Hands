"""Hardware-facing layer: anything that touches the real screen, mouse,
keyboard, or operating-system app launcher.

Split into three focused modules so each one is small enough to read at a
glance:

    screen   Capture screenshots, draw the coordinate grid overlay, report
             screen size, and run the start-up permission preflight.
    input    Mouse clicks/scroll/move + keyboard typing/hotkeys, with
             platform branches for Mac (AppleScript) vs Windows (pyautogui).
    apps     Launch and focus desktop applications (open -a / start).

Importing from this package gives you everything you need to drive a
machine without thinking about platform differences.
"""
