# Invisible Hands

> An AI agent that controls your computer and does your tasks — fully local, fully private.

---

You open your laptop. Instead of clicking through ten windows, switching tabs, hunting for that one button — you just type what you want done. The agent reads it, makes a plan, and starts working. Mouse moves. Buttons get clicked. Things get done. You watch.

That's Invisible Hands.

It's not a chatbot. It's not a copilot that suggests things. It's an agent that *acts* — on your screen, in real time, while you sit back. And because everything runs on your own machine through local models, nothing you type, nothing on your screen, and nothing the agent does ever touches the internet.

Works on macOS and Windows.

---

## What makes it different

Most "AI productivity" tools either need a cloud subscription, send your data somewhere, or only work inside one specific app. Invisible Hands runs entirely on your hardware. The models are local. The logic is local. The screenshots it takes to verify its work — local. Nothing leaves.

And it's not brittle. It doesn't rely on a single trick to click things. It uses a chain of three strategies, each smarter than the last, so it handles everything from a standard desktop app to a weird custom web UI without breaking a sweat.

---

## How it works, plainly

When you give it a task, three things happen in a loop:

**Plan.** A reasoning model breaks your task into a sequence of steps. "Open this app. Click this. Type that. Confirm."

**Execute.** For each step, it takes a screenshot and figures out where to click. It tries the fastest method first, falls back to a smarter one if needed.

**Verify.** After each action, it takes another screenshot and checks: did that actually work? If not, it retries. If yes, it moves to the next step.

This plan → act → verify loop keeps running until your task is complete — or until it hits a wall and tells you why.

---

## The click chain (the clever part)

Finding the right thing to click on screen is harder than it sounds. Invisible Hands uses three strategies in priority order:

| Strategy | What it's good at | Speed |
|---|---|---|
| OCR (Tesseract) | Buttons and links with visible text | Fast |
| Native UI | Real desktop apps like Finder, Word, Settings | Fast + very accurate |
| Vision model | Anything else — icons, canvases, custom UIs | Slower, but always works |

The cheap, accurate ones go first. Vision is the guaranteed fallback. Each one can be toggled off independently if you're debugging or benchmarking.

---

## Voice input

If you'd rather talk than type, there's an optional voice mode. Press Enter, speak your task, stop talking — Whisper transcribes it offline and hands it to the agent. No cloud. No API key. Just your mic and a local Whisper model.

```bash
invisible-hands --voice
# or switch mid-session
> :voice
> :type
```

---

## Every run is recorded

Each task produces a JSON report and per-step screenshots saved to `reports/`. You can replay exactly what happened, what the agent saw, what it clicked, and where it succeeded or failed.

---

## Setup

You need Python 3.9+, [LM Studio](https://lmstudio.ai) running locally with two models loaded:

| Model | Role |
|---|---|
| `qwen/qwen3.5-9b` | Planning — turns your task into steps |
| `qwen2.5-vl-7b-instruct` | Vision — clicks, verifies, handles anything visual |

Then install Tesseract for OCR:

```bash
# macOS
brew install tesseract

# Windows
choco install tesseract
```

Clone and install:

```bash
git clone <this repo>
cd Invisible-Hands

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -e ".[mac]"         # or [windows]
pip install -e ".[voice]"       # optional, for mic input
```

Run:

```bash
invisible-hands
```

You get a 3-second countdown to switch to the right window, then the agent takes over. Move your mouse to the top-left corner at any point to abort.

---

## macOS permissions

Three things need to be enabled in System Settings → Privacy & Security:

- **Screen Recording** — so it can take screenshots
- **Accessibility** — for mouse control and native UI automation
- **Automation → System Events** — for keyboard input via AppleScript

Grant them, then fully quit and relaunch your terminal. macOS only applies permission changes to newly launched processes.

---

## Configuration

Everything is tunable via environment variables — no code edits needed.

| Variable | Default | What it does |
|---|---|---|
| `INVISIBLE_LM_STUDIO_URL` | `http://localhost:1234` | Where LM Studio is running |
| `INVISIBLE_VISION_MODEL` | `qwen2.5-vl-7b-instruct` | Vision model |
| `INVISIBLE_THINKING_MODEL` | `qwen/qwen3.5-9b` | Planning model |
| `INVISIBLE_VERIFY` | `1` | Toggle post-step verification |
| `INVISIBLE_OCR_FUZZY` | `80` | Fuzzy match threshold for OCR (0–100) |
| `INVISIBLE_DELAY` | `1.5` | Seconds to wait after each action |
| `INVISIBLE_MAX_RETRIES` | `3` | Retry budget per step |
| `INVISIBLE_COUNTDOWN` | `3` | Pre-run countdown |
| `INVISIBLE_REPORTS_DIR` | `./reports` | Where reports and screenshots go |
| `INVISIBLE_VOICE_MODEL` | `base.en` | Whisper model size |
| `INVISIBLE_VOICE_SILENCE_SECONDS` | `1.5` | Silence before recording stops |

Examples:

```bash
INVISIBLE_VERIFY=0 invisible-hands                             # skip verification (faster)
INVISIBLE_VISION_MODEL=qwen2.5-vl-32b-instruct invisible-hands # bigger vision model
INVISIBLE_OCR_FUZZY=70 invisible-hands                         # looser OCR matching
```

---

## CLI flags

| Flag | Effect |
|---|---|
| `--no-verify` | Skip post-step verification |
| `--no-ocr` | Disable OCR clicker |
| `--no-native` | Disable native UI clicker |
| `--no-vision` | Disable vision clicker |
| `--voice` | Start in voice input mode |

---

## Safety

- Move the mouse to the **top-left corner** at any point to trigger pyautogui's failsafe and stop the agent immediately
- **Ctrl+C** in the terminal exits cleanly
- The countdown before each task gives you time to make sure you're in the right window

---

## Project structure

```
Invisible-Hands/
├── src/invisible_hands/
│   ├── cli.py          # CLI and interactive prompt
│   ├── agent.py        # Plan → execute → verify loop
│   ├── config.py       # All settings and env overrides
│   ├── llm/            # Planner, executor, verifier, LM Studio client
│   ├── clickers/       # OCR, native UI, vision, and the chain
│   ├── controllers/    # Screenshot, mouse/keyboard, app launcher
│   ├── parsing/        # Step detection and action parsing
│   ├── reporting/      # Terminal output and JSON report writer
│   └── voice/          # Whisper mic input (optional)
├── scripts/
│   ├── test_actions.py # Test keyboard/mouse/screen without AI
│   └── test_click.py   # Benchmark click chain accuracy
└── reports/            # Per-run output (gitignored)
```

---

## Troubleshooting

**Can't connect to LM Studio** — Open LM Studio, go to the local server tab, start it. Default port is 1234. If you changed it, set `INVISIBLE_LM_STUDIO_URL`.

**Vision model not found** — Open LM Studio's model manager and load both models. Or override with `INVISIBLE_VISION_MODEL`.

**OCR shows "(unavailable)"** — Tesseract isn't on your PATH. Install it and reopen your terminal. Confirm with `tesseract --version`.

**Native UI clicker never works on macOS** — Accessibility permission is missing or was granted before launching the terminal. Revoke, relaunch terminal, re-grant.

**Screen capture fails on macOS** — Screen Recording permission is missing. Grant it to your terminal app and restart.

**Vision model returns bad coordinates** — Try the 32B model: `INVISIBLE_VISION_MODEL=qwen2.5-vl-32b-instruct`. The 3B model isn't reliable for grid-based coordinate tasks.

**Voice mode unavailable** — Run `pip install -e .[voice]`. On macOS also `brew install portaudio ffmpeg`.

**Voice recording doesn't stop** — Room is too noisy. Either raise `INVISIBLE_VOICE_SILENCE_RMS=0.04` or just press Enter to stop manually.

**Whisper transcription is wrong** — Pin the language with `INVISIBLE_VOICE_LANGUAGE=en` and consider upgrading the model: `INVISIBLE_VOICE_MODEL=small.en` or `medium.en`.

---

## Open to feedback

This is still being actively shaped. If something in the click chain misbehaves on your setup, if a specific app gives it trouble, or if you have ideas for new strategies or smarter verification — open an issue or drop a note. The architecture is intentionally modular, so new clickers, planners, or controllers can be added without touching the core loop.

What would make this more useful to you? Happy to hear it.