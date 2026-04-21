"""Central place for every tunable setting in the agent.

Why a single config module?
    Scattering magic numbers across files (1024 here, 0.8 there) makes it
    painful to tweak behaviour. Putting them all here gives one obvious file
    to read when something feels "off", and one obvious place to override
    settings without touching code.

Why env-var overrides?
    Lets you experiment quickly:
        INVISIBLE_VERIFY=0 invisible-hands
        INVISIBLE_VISION_MODEL=qwen2.5-vl-32b-instruct invisible-hands
    No file edits, no rebuild. Good for benchmarking different models, fuzzy
    thresholds, or disabling whole strategies for an A/B comparison.

Naming convention:
    All env vars start with INVISIBLE_ to avoid colliding with anything else
    on the user's system.
"""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var. Accepts 1/0, true/false, yes/no (case-insensitive)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to default if unset or unparseable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to default if unset or unparseable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# LLM (LM Studio) endpoints + models
# ─────────────────────────────────────────────────────────────────────────────

# Where the local LM Studio server is listening. The default matches LM Studio's
# out-of-the-box port. Change INVISIBLE_LM_STUDIO_URL if you're proxying or running
# on a different machine.
LM_STUDIO_BASE = os.getenv("INVISIBLE_LM_STUDIO_URL", "http://localhost:1234")

# Vision model: looks at screenshots and outputs click coordinates.
# We default to the 7B variant — the 3B model struggled with grid reading;
# 7B gives noticeably better grounding without the heavy RAM cost of the 32B.
VISION_MODEL = os.getenv("INVISIBLE_VISION_MODEL", "qwen/qwen3-vl-8b")

# Thinking model: turns the user's task into a numbered step plan.
# qwen3.5-9b is a solid free-tier reasoning model available in LM Studio.
THINKING_MODEL = os.getenv("INVISIBLE_THINKING_MODEL", "qwen/qwen3.5-9b")


# ─────────────────────────────────────────────────────────────────────────────
# Vision pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Before we send a screenshot to the vision model we resize it down so the grid
# numbers we overlay are large enough for a small model to read. Coordinates
# returned by the model are scaled back up to real screen pixels afterwards.
# 1024 hits a nice sweet spot between legibility and detail loss on 1728-wide
# Retina screens.
VISION_MAX_WIDTH = _env_int("INVISIBLE_VISION_MAX_WIDTH", 1024)


# ─────────────────────────────────────────────────────────────────────────────
# Click strategies
# ─────────────────────────────────────────────────────────────────────────────
# The agent tries clickers in order: OCR -> Native UI -> Vision. Each can be
# disabled independently if it's misbehaving or not installed on this machine.

# Tesseract-based OCR clicker. Requires the `tesseract` binary to be installed
# (brew install tesseract / choco install tesseract). The clicker will detect
# this at runtime and skip itself cleanly if unavailable.
ENABLE_OCR_CLICKER = _env_bool("INVISIBLE_OCR", True)

# Native-OS UI automation clicker. Uses atomac on macOS or pywinauto on Windows
# to query the OS accessibility tree for pixel-perfect element bounds.
ENABLE_NATIVE_CLICKER = _env_bool("INVISIBLE_NATIVE_UI", True)

# Vision-model clicker. This is the existing screenshot-and-LLM path. It's the
# slowest and least accurate but works on absolutely any UI, so it's the
# guaranteed fallback at the end of the chain.
ENABLE_VISION_CLICKER = _env_bool("INVISIBLE_VISION", True)

# OCR fuzzy-match threshold (0-100). Higher = stricter text matching.
# 80 means the matched text must be >= 80% similar to the requested target.
# Lower it if Tesseract often misreads the right word; raise it if you're
# getting false positives.
OCR_FUZZY_THRESHOLD = _env_int("INVISIBLE_OCR_FUZZY", 80)


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────
# After each step we can ask the vision model whether the step actually
# succeeded by looking at a fresh screenshot. This roughly doubles per-step
# latency, so we expose a switch.

ENABLE_VERIFICATION = _env_bool("INVISIBLE_VERIFY", False)


# ─────────────────────────────────────────────────────────────────────────────
# Timing knobs
# ─────────────────────────────────────────────────────────────────────────────

# Wait this many seconds after every action before doing anything else, so the
# UI has a chance to update (page navigation, animations, focus shift).
DELAY_AFTER_ACTION = _env_float("INVISIBLE_DELAY", 1.5)

# How many times to retry a failed step (vision miss + verification fail) before
# giving up and moving on.
MAX_RETRIES_PER_STEP = _env_int("INVISIBLE_MAX_RETRIES", 3)

# Countdown shown before the agent starts acting, so you can switch to the
# target window without the agent typing into the terminal.
COUNTDOWN_SECONDS = _env_int("INVISIBLE_COUNTDOWN", 3)


# ─────────────────────────────────────────────────────────────────────────────
# Voice input (OpenAI Whisper)
# ─────────────────────────────────────────────────────────────────────────────
# When the optional [voice] extra is installed, the CLI exposes a `:voice`
# command (and a --voice flag) that records from the microphone, transcribes
# with Whisper, and uses the result as the task. Everything runs locally; the
# first run downloads the chosen model into ~/.cache/whisper.

# Whisper model size. Trade-off: tiny=fastest+lowest-RAM, base=good default,
# small/medium=better accuracy + more RAM. "english-only" variants (.en) are
# noticeably better for English-only users (tiny.en, base.en, small.en).
VOICE_MODEL = os.getenv("INVISIBLE_VOICE_MODEL", "base.en")

# Force a transcription language ("en", "es", ...). None = auto-detect.
# Pinning the language is faster and avoids the rare "wrong language" misfire.
VOICE_LANGUAGE = os.getenv("INVISIBLE_VOICE_LANGUAGE", "en") or None

# Mic capture sample rate. Whisper internally resamples to 16 kHz, so anything
# higher just wastes CPU. 16000 is the right default for speech.
VOICE_SAMPLE_RATE = _env_int("INVISIBLE_VOICE_SAMPLE_RATE", 16000)

# Maximum recording length, in seconds, before we stop listening even if the
# user hasn't paused. A safety net so a stuck mic can't capture forever.
VOICE_MAX_SECONDS = _env_int("INVISIBLE_VOICE_MAX_SECONDS", 30)

# Silence detection: stop recording after this many continuous seconds of
# audio below VOICE_SILENCE_RMS. Smaller = snappier but more cut-offs.
VOICE_SILENCE_SECONDS = _env_float("INVISIBLE_VOICE_SILENCE_SECONDS", 1.5)

# RMS threshold for "silence". 0.0..1.0 on normalised float32 audio. Typical
# room noise lands around 0.005-0.015; speech is usually >= 0.05. Bump this
# up if the recorder won't stop in a noisy room.
VOICE_SILENCE_RMS = _env_float("INVISIBLE_VOICE_SILENCE_RMS", 0.015)

# Don't bother transcribing recordings shorter than this — they're almost
# always accidental key presses or the user clearing their throat.
VOICE_MIN_SECONDS = _env_float("INVISIBLE_VOICE_MIN_SECONDS", 0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem
# ─────────────────────────────────────────────────────────────────────────────

# Each agent run writes a JSON report + screenshots into a timestamped folder
# under this directory. Set INVISIBLE_REPORTS_DIR to send them somewhere else.
_DEFAULT_REPORTS = os.path.join(os.getcwd(), "reports")
REPORTS_DIR = os.getenv("INVISIBLE_REPORTS_DIR", _DEFAULT_REPORTS)
