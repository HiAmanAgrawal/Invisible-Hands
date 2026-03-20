# ============================================================
# WHAT THIS SCRIPT DOES:
# 1. Opens Spotify on your computer
# 2. Waits for it to fully load
# 3. Brings Spotify window to the front (focuses it)
# 4. Presses Cmd+K to open the search bar
# 5. Types the song name
# 6. Presses Enter to search
# ============================================================


# --- IMPORTS ---
# These are pre-built Python tools (libraries) we need for this script

import pyautogui   # Controls mouse and keyboard automatically
import subprocess  # Lets Python run terminal/shell commands (like opening apps)
import sys         # Gives info about the operating system (Mac, Windows, Linux)
import time        # Lets us pause the script using time.sleep()


# ============================================================
# FUNCTION 1: send_keystroke
# ------------------------------------------------------------
# A "function" is a reusable block of code you can call by name.
# This function sends a keyboard shortcut (like Cmd+K) to the
# computer using AppleScript — macOS's built-in automation language.
#
# WHY NOT USE pyautogui.hotkey() for Cmd+K?
# pyautogui.hotkey() is unreliable for modifier keys (like Cmd)
# on macOS because of sandboxing restrictions.
# AppleScript sends keystrokes through macOS's own accessibility
# layer, so it works perfectly every time.
#
# PARAMETERS (inputs to the function):
#   key      → the letter to press, e.g. "k"
#   modifier → the modifier key to hold, e.g. "command"
# ============================================================
def send_keystroke(key, modifier="command"):
    # subprocess.run() executes a terminal command from Python
    # "osascript" is the macOS tool to run AppleScript
    # The AppleScript tells "System Events" (macOS automation engine)
    # to press the key while holding the modifier key down
    subprocess.run([
        "osascript", "-e",
        f'tell application "System Events" to keystroke "{key}" using {modifier} down'
        # f"..." is an f-string — it lets us insert variables inside a string
        # {key} gets replaced by "k", {modifier} gets replaced by "command"
    ])


# ============================================================
# FUNCTION 2: open_spotify_and_search
# ------------------------------------------------------------
# This is the main function that does all the work.
# It opens Spotify and searches for a song.
#
# PARAMETER:
#   song → the name of the song to search (default is "bairan")
#          You can change this to any song name you want!
# ============================================================
def open_spotify_and_search(song="bairan"):

    # sys.platform tells us what OS the script is running on:
    #   "darwin"  = macOS (Mac computers)
    #   "win32"   = Windows
    #   "linux"   = Linux

    # -------------------------------------------------------
    # MAC SECTION
    # -------------------------------------------------------
    if sys.platform == "darwin":

        # STEP 1: Launch Spotify
        # subprocess.run() runs a shell command — same as typing in Terminal
        # "open -a Spotify" is a macOS command that opens any installed app by name
        print("Launching Spotify...")  # print() shows a message in the terminal so you know what's happening
        subprocess.run(["open", "-a", "Spotify"], capture_output=True)
        # capture_output=True means we silently capture any output/errors (don't show them)

        # STEP 2: Wait for Spotify to fully load
        # time.sleep(5) pauses the script for 5 seconds
        # If we don't wait, the next steps will run before Spotify is ready
        # and nothing will work
        print("Waiting for Spotify to load...")
        time.sleep(5)  # ← increase this number if Spotify opens slowly on your Mac

        # STEP 3: Bring Spotify window to the front (focus it)
        # Even though Spotify is open, your Terminal might still be the active window.
        # If we send keystrokes without focusing Spotify first,
        # the keys will go to Terminal instead of Spotify — so nothing happens!
        # This AppleScript tells macOS: "make Spotify the active/focused app"
        print("Focusing Spotify window...")
        subprocess.run(["osascript", "-e", 'tell application "Spotify" to activate'])
        time.sleep(2)  # Wait 2 seconds for the focus to fully switch

        # STEP 4: Open Spotify's search bar using Cmd+K
        # We use our send_keystroke() function (defined above) instead of pyautogui
        # because pyautogui.hotkey() is unreliable for Cmd key combos on macOS
        # Cmd+K is Spotify's keyboard shortcut to open the search bar
        print("Sending Cmd+K to open search bar...")
        send_keystroke("k", "command")
        time.sleep(2)  # Wait for the search bar to open and be ready for input

        # STEP 5: Type the song name into the search bar
        # pyautogui.write() simulates typing on the keyboard
        # interval=0.12 means wait 0.12 seconds between each character
        # (typing too fast can sometimes cause characters to be missed)
        print(f"Typing '{song}'...")
        pyautogui.write(song, interval=0.12)
        time.sleep(1)  # Wait a moment before pressing Enter

        # STEP 6: Press Enter to execute the search
        # pyautogui.press() simulates pressing a single key
        print("Pressing Enter to search...")
        pyautogui.press("enter")

        # Let the user know the script finished successfully
        print(f"Done! Searched for '{song}' on Spotify.")

    # -------------------------------------------------------
    # WINDOWS SECTION
    # -------------------------------------------------------
    elif sys.platform == "win32":

        # STEP 1: Open Spotify via the Windows Start menu
        print("Launching Spotify...")
        pyautogui.hotkey("win")         # Press the Windows key to open Start menu
        time.sleep(1)                   # Wait for Start menu to open
        pyautogui.write("Spotify", interval=0.05)  # Type "Spotify" to search for it
        time.sleep(1)                   # Wait for search results to appear
        pyautogui.press("enter")        # Press Enter to open Spotify
        time.sleep(5)                   # Wait for Spotify to fully load

        # STEP 2: Open search bar with Ctrl+K (Windows uses Ctrl instead of Cmd)
        pyautogui.hotkey("ctrl", "k")
        time.sleep(1.5)

        # STEP 3: Type the song name and search
        pyautogui.write(song, interval=0.1)
        pyautogui.press("enter")
        print(f"Done! Searched for '{song}' on Spotify.")


# ============================================================
# MAIN FUNCTION
# ------------------------------------------------------------
# In Python, it's good practice to put your "starting point"
# code inside a function called main().
# This keeps the code clean and organized.
# ============================================================
def main():
    # Call our search function with the song "bairan"
    # You can change "bairan" to any song name you like!
    open_spotify_and_search("bairan")


# ============================================================
# ENTRY POINT
# ------------------------------------------------------------
# This is a special Python pattern.
# When you run this file directly (python test.py),
# Python sets __name__ to "__main__".
# This "if" check makes sure main() only runs when YOU run
# the script — not if someone imports this file into another script.
# ============================================================
if __name__ == "__main__":
    main()