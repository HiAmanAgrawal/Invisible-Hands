# ============================================================
# STEP 1 — FIND THE COORDINATES OF ANY ELEMENT ON SCREEN
# ============================================================
# HOW TO USE THIS SCRIPT:
#   1. Run this script
#   2. You have 5 seconds — switch to Chrome
#   3. Hover your mouse over the Chrome search/address bar
#   4. STAY STILL — the script will print the coordinates
#   5. Copy those coordinates and use them in click_searchbar.py
# ============================================================

import pyautogui
import time

print("=" * 50)
print("   COORDINATE FINDER")
print("=" * 50)
print("You have 5 seconds to hover over Chrome's search bar!")
print("Switch to Chrome NOW and hover over the address bar...")
print()

# Count down so you have time to switch windows
for i in range(5, 0, -1):
    print(f"  Reading position in {i}...")
    time.sleep(1)

# pyautogui.position() captures wherever your mouse is RIGHT NOW
x, y = pyautogui.position()

print()
print(f"  ✅ Found it! Your mouse was at: x={x}, y={y}")
print()
print(f"  👉 Now open click_searchbar.py and set:")
print(f"     SEARCH_BAR_X = {x}")
print(f"     SEARCH_BAR_Y = {y}")
print()
print("=" * 50)