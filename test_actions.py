#!/usr/bin/env python3
"""
Interactive action tester — pick an action, it runs after a 3-second countdown.
No AI, no models — just raw device_controller functions.
"""

import time
import sys

from device_controller import (
    open_app, activate_app, click, double_click, right_click,
    hotkey, press_key, type_text, scroll_screen, move_to,
    take_screenshot, get_screen_size,
)

ACTIONS = {
    "1": "open_app",
    "2": "activate_app",
    "3": "click",
    "4": "double_click",
    "5": "right_click",
    "6": "type_text",
    "7": "press_key",
    "8": "hotkey",
    "9": "scroll",
    "10": "move_to",
    "11": "screenshot",
    "12": "screen_size",
}


def countdown(seconds=3):
    print(f"\n  Executing in...")
    for i in range(seconds, 0, -1):
        print(f"  {i}")
        time.sleep(1)
    print()


def run_action(choice: str):
    name = ACTIONS.get(choice)

    if name == "open_app":
        app = input("  App name (e.g. Google Chrome): ").strip()
        countdown()
        open_app(app)
        print(f"  Done: opened {app}")

    elif name == "activate_app":
        app = input("  App name to bring to front: ").strip()
        countdown()
        activate_app(app)
        print(f"  Done: activated {app}")

    elif name == "click":
        x = int(input("  X coordinate: "))
        y = int(input("  Y coordinate: "))
        countdown()
        click(x, y)
        print(f"  Done: clicked ({x}, {y})")

    elif name == "double_click":
        x = int(input("  X coordinate: "))
        y = int(input("  Y coordinate: "))
        countdown()
        double_click(x, y)
        print(f"  Done: double-clicked ({x}, {y})")

    elif name == "right_click":
        x = int(input("  X coordinate: "))
        y = int(input("  Y coordinate: "))
        countdown()
        right_click(x, y)
        print(f"  Done: right-clicked ({x}, {y})")

    elif name == "type_text":
        text = input("  Text to type: ")
        countdown()
        type_text(text)
        print(f"  Done: typed \"{text}\"")

    elif name == "press_key":
        key = input("  Key (e.g. enter, escape, tab, /, k): ").strip()
        countdown()
        press_key(key)
        print(f"  Done: pressed {key}")

    elif name == "hotkey":
        combo = input("  Key combo (e.g. command+l, command+shift+t): ").strip()
        keys = [k.strip() for k in combo.split("+")]
        countdown()
        hotkey(*keys)
        print(f"  Done: hotkey {'+'.join(keys)}")

    elif name == "scroll":
        direction = input("  Direction (up/down): ").strip().lower()
        amount = int(input("  Amount (1-10): ") or "3")
        countdown()
        scroll_screen(direction, amount)
        print(f"  Done: scrolled {direction} x{amount}")

    elif name == "move_to":
        x = int(input("  X coordinate: "))
        y = int(input("  Y coordinate: "))
        countdown()
        move_to(x, y)
        print(f"  Done: moved to ({x}, {y})")

    elif name == "screenshot":
        countdown()
        img = take_screenshot()
        path = "test_screenshot.png"
        img.save(path)
        print(f"  Done: saved screenshot to {path} ({img.size[0]}x{img.size[1]})")

    elif name == "screen_size":
        w, h = get_screen_size()
        print(f"\n  Screen size: {w}x{h}")


def main():
    print()
    print("  Action Tester")
    print("  ─────────────────────────────")
    print("  Pick an action, it runs after 3 seconds.\n")

    while True:
        print("  Actions:")
        for num, name in sorted(ACTIONS.items(), key=lambda x: int(x[0])):
            print(f"    {num:>2}. {name}")
        print(f"     q. quit\n")

        try:
            choice = input("  > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Bye!")
            break

        if choice in ("q", "quit", "exit"):
            print("  Bye!")
            break

        if choice not in ACTIONS:
            print("  Invalid choice.\n")
            continue

        try:
            run_action(choice)
        except KeyboardInterrupt:
            print("\n  Cancelled.")
        except Exception as e:
            print(f"  Error: {e}")

        print()


if __name__ == "__main__":
    main()
