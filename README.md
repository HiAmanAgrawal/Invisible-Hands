# AI Desktop Agent

A proof-of-concept desktop automation agent. Give it a task in plain English, and it will look at your screen and control your mouse/keyboard to get it done.

**Stack:** Ollama (`llama3.2-vision`) + PyAutoGUI

## How It Works

1. You type a task (e.g. *"open Safari and search for Python tutorials"*)
2. The agent captures a screenshot of your screen
3. Sends the screenshot to `llama3.2-vision` running locally via Ollama
4. The model analyzes the screen and decides the next action (click, type, scroll, etc.)
5. The agent executes the action using PyAutoGUI
6. Repeats until the task is done (or max 50 steps)

## Prerequisites

- **Python 3.9+**
- **Ollama** running locally with `llama3.2-vision` pulled
- **macOS permissions** (see below)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running and the model is available
ollama pull llama3.2-vision

# 3. Run the agent
python main.py
```

## macOS Permissions

Your terminal app needs two permissions in **System Settings > Privacy & Security**:

| Permission | Why |
|---|---|
| **Accessibility** | So PyAutoGUI can control mouse & keyboard |
| **Screen Recording** | So PyAutoGUI can take screenshots |

Grant these to your terminal app (Terminal.app, iTerm2, Cursor, etc.).

## Safety

- **Emergency stop:** Move your mouse to the top-left corner of the screen — PyAutoGUI's FAILSAFE will abort immediately.
- **Ctrl+C** also stops the agent.
- There's a 3-second countdown before the agent starts acting.
- The agent stops after 50 steps max.

## Example

```
  AI Desktop Agent
  ────────────────────────────────────
  Powered by Ollama (llama3.2-vision) + PyAutoGUI
  Type a task and press Enter. Type quit to exit.

  > open Notes and create a new note that says hello world
```

## File Structure

| File | Purpose |
|---|---|
| `main.py` | Entry point, agent loop, CLI |
| `brain.py` | Ollama vision integration, action parsing |
| `device_controller.py` | Mouse, keyboard, and screenshot control |
| `requirements.txt` | Python dependencies |
