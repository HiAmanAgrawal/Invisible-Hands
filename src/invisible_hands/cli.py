"""Command-line entry point and interactive REPL.

This module is intentionally short — its job is to:

    1. Parse CLI flags into an AgentOptions object.
    2. Print the welcome banner.
    3. Run the start-up preflight check.
    4. Loop: read a task from the user, run the agent, repeat.

Tasks can be supplied two ways at the prompt:
    > some task here       (typed)
    > :voice               (record from the mic, transcribed by Whisper)

The --voice CLI flag flips the default so the user is dropped straight
into the mic on every turn (typing `:type` switches back).

All the heavy lifting lives in `agent.py` and `voice/recorder.py`.
"""

from __future__ import annotations

import argparse
import sys
import time

from invisible_hands import config
from invisible_hands.agent import Agent, AgentOptions
from invisible_hands.controllers.screen import preflight_check
from invisible_hands.reporting.reporter import c


def _build_parser() -> argparse.ArgumentParser:
    """argparse spec for all CLI flags. Each --no-* turns a default OFF."""
    parser = argparse.ArgumentParser(
        prog="invisible-hands",
        description="Local-first AI desktop agent. Type a task, watch it happen.",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Disable post-step vision verification (faster but less safe).",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Force verification on (overrides INVISIBLE_VERIFY=0).",
    )
    parser.add_argument(
        "--no-ocr", action="store_true",
        help="Disable the Tesseract OCR clicker.",
    )
    parser.add_argument(
        "--no-native", action="store_true",
        help="Disable the native UI automation clicker (atomac/pywinauto).",
    )
    parser.add_argument(
        "--no-vision", action="store_true",
        help="Disable the vision-model fallback clicker. ONLY do this if "
             "you know OCR/Native cover all your steps - otherwise visual "
             "steps will fail.",
    )
    parser.add_argument(
        "--voice", action="store_true",
        help="Start in voice mode: every turn records from the mic and "
             "transcribes with Whisper. Type :type at the prompt to switch "
             "back to typed input. Requires the [voice] extra.",
    )
    return parser


def _options_from_args(args: argparse.Namespace) -> AgentOptions:
    """Apply CLI flags on top of config.py defaults to produce AgentOptions."""
    enable_verification = config.ENABLE_VERIFICATION
    if args.no_verify:
        enable_verification = False
    if args.verify:
        enable_verification = True

    return AgentOptions(
        enable_verification=enable_verification,
        enable_ocr=config.ENABLE_OCR_CLICKER and not args.no_ocr,
        enable_native=config.ENABLE_NATIVE_CLICKER and not args.no_native,
        enable_vision=config.ENABLE_VISION_CLICKER and not args.no_vision,
    )


def _print_banner(options: AgentOptions, voice_default: bool) -> None:
    """Welcome banner with current model + clicker config."""
    print()
    print(f"  {c('Invisible Hands', 'bold')} - local AI desktop agent")
    print(f"  {c('-' * 36, 'dim')}")
    print(f"  Vision model:   {config.VISION_MODEL}")
    print(f"  Thinking model: {config.THINKING_MODEL}")
    print(f"  LM Studio:      {config.LM_STUDIO_BASE}")
    print()

    enabled_clickers = []
    if options.enable_ocr:
        enabled_clickers.append("OCR")
    if options.enable_native:
        enabled_clickers.append("Native UI")
    if options.enable_vision:
        enabled_clickers.append("Vision")
    print(f"  Click chain:    "
          f"{c(' -> '.join(enabled_clickers) or '(none!)', 'cyan')}")

    verify_status = (c("ON", "green") if options.enable_verification
                     else c("OFF", "yellow"))
    print(f"  Verification:   {verify_status}")

    input_mode = c("voice (Whisper)", "magenta") if voice_default else c("text", "cyan")
    print(f"  Input mode:     {input_mode}")
    print()


def _run_preflight() -> None:
    """Execute the startup checks and exit hard if anything is wrong."""
    print(f"  {c('Running pre-flight checks...', 'dim')}")
    errors = preflight_check()
    if errors:
        print(f"  {c('Pre-flight checks FAILED:', 'red')}")
        for err in errors:
            for line in err.split("\n"):
                print(f"    {c('x', 'red')} {line}")
        print()
        print(f"  Fix the issues above and try again.")
        sys.exit(1)

    print(f"  {c('OK', 'green')} Screen capture")
    print(f"  {c('OK', 'green')} Keyboard automation")
    print(f"  {c('OK', 'green')} LM Studio + models")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Voice helpers
# ─────────────────────────────────────────────────────────────────────────────
# Voice support is optional. We import it lazily and degrade gracefully if
# the [voice] extra (whisper + sounddevice) isn't installed.

def _make_voice_recorder():
    """Build a VoiceRecorder, returning None (with a friendly message) on failure.

    We swallow VoiceUnavailableError here so the CLI never crashes just
    because Whisper isn't installed — the user sees an actionable hint and
    can fall back to typing immediately."""
    try:
        from invisible_hands.voice import VoiceRecorder, is_available

        if not is_available():
            print(f"  {c('Voice mode unavailable.', 'yellow')} "
                  f"Install with: {c('pip install -e .[voice]', 'cyan')}")
            extra_hint = "Also requires PortAudio (brew install portaudio) and ffmpeg."
            print(f"  {c(extra_hint, 'dim')}")
            return None

        recorder = VoiceRecorder()
        print(f"  {c('Loading Whisper model', 'dim')} "
              f"({c(config.VOICE_MODEL, 'cyan')})... "
              f"{c('first run downloads it (~75-500 MB)', 'dim')}")
        recorder.warm_up()
        print(f"  {c('OK', 'green')} Whisper ready")
        return recorder
    except Exception as e:
        print(f"  {c('Voice mode unavailable:', 'yellow')} {e}")
        return None


def _read_voice_task(recorder) -> str:
    """Capture one mic utterance + transcribe; return the resulting text."""
    def status(msg: str) -> None:
        print(f"  {c('mic:', 'magenta')} {msg}")

    try:
        text = recorder.listen_and_transcribe(on_status=status)
    except Exception as e:
        print(f"  {c('Voice capture failed:', 'red')} {e}")
        return ""
    if text:
        print(f"  {c('heard:', 'magenta')} \"{text}\"")
    return text


def _read_text_task() -> str:
    """Plain typed prompt. Returns the trimmed string."""
    return input(f"  {c('>', 'cyan')} ").strip()


def _prompt_for_task(voice_mode: bool, recorder) -> str:
    """Get the next task, in either text or voice mode.

    Returns the task string (possibly empty) — the caller decides what to
    do with empties (skip, prompt again, etc.)."""
    if voice_mode and recorder is not None:
        # Hint that pressing Enter starts the recording, in case the user
        # is staring at a quiet prompt wondering what to do.
        print(f"  {c('>', 'magenta')} (press Enter to start recording, "
              f"then speak; pause to stop)")
        try:
            input("    ")
        except EOFError:
            return ""
        return _read_voice_task(recorder)
    return _read_text_task()


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point.

    Wired up two ways:
        - As a console script via pyproject.toml (`invisible-hands`)
        - As `python -m invisible_hands` via __main__.py
    """
    args = _build_parser().parse_args()
    options = _options_from_args(args)
    voice_mode = bool(args.voice)

    _print_banner(options, voice_mode)
    _run_preflight()

    # Build the voice recorder up front (only if needed). If construction
    # fails we drop back to text mode and tell the user why.
    recorder = _make_voice_recorder() if voice_mode else None
    if voice_mode and recorder is None:
        voice_mode = False
        print(f"  {c('Falling back to text input.', 'dim')}\n")

    _print_help_line(voice_mode)

    agent = Agent(options)

    while True:
        try:
            task = _prompt_for_task(voice_mode, recorder)

            if not task:
                continue

            # In-REPL mode-switch commands. Cheap to support, very useful
            # when the mic is misbehaving in a meeting room.
            lowered = task.lower().strip().rstrip(".")
            if lowered in ("quit", "exit", "q", ":quit", ":exit"):
                print(f"  {c('Goodbye!', 'dim')}")
                break
            if lowered in (":voice", "voice mode", "voice"):
                if recorder is None:
                    recorder = _make_voice_recorder()
                if recorder is None:
                    continue
                voice_mode = True
                print(f"  {c('Switched to voice input.', 'magenta')}\n")
                continue
            if lowered in (":type", ":text", "type mode", "text mode"):
                voice_mode = False
                print(f"  {c('Switched to text input.', 'cyan')}\n")
                continue

            # Quick countdown so the user can switch focus to the right window.
            print(f"\n  {c('Starting in...', 'yellow')}")
            for i in range(config.COUNTDOWN_SECONDS, 0, -1):
                print(f"  {c(str(i), 'bold')}")
                time.sleep(1)

            agent.run(task)
        except KeyboardInterrupt:
            print(f"\n\n  {c('Interrupted. Goodbye!', 'dim')}")
            sys.exit(0)


def _print_help_line(voice_mode: bool) -> None:
    """One-liner shown after the banner explaining how to drive the REPL."""
    if voice_mode:
        print(f"  {c('Voice mode.', 'magenta')} "
              f"Press Enter to record. Type {c(':type', 'cyan')} to switch "
              f"to keyboard input, {c('quit', 'yellow')} to exit.")
    else:
        print(f"  Type a task and press Enter. "
              f"Type {c(':voice', 'magenta')} for voice input, "
              f"{c('quit', 'yellow')} to exit.")
    print(f"  {c('Safety:', 'red')} Move mouse to top-left corner to emergency stop.")
    print()


if __name__ == "__main__":
    main()
