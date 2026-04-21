"""Mic capture + offline transcription via OpenAI Whisper (vanilla).

This module turns the user's voice into a task string the agent can run.
The flow is:

    1. Open the default microphone with `sounddevice` at 16 kHz mono.
    2. Stream audio in small chunks while watching the rolling RMS level.
    3. Stop when EITHER:
         - the user has been silent for VOICE_SILENCE_SECONDS, OR
         - we hit VOICE_MAX_SECONDS (safety net), OR
         - the user pressed Enter / Ctrl+C in the calling thread.
    4. Hand the captured float32 buffer straight to the Whisper model
       (no temp files, no ffmpeg needed for the audio path itself —
       Whisper still needs ffmpeg installed for the model load).
    5. Return the trimmed transcription.

Everything is local: once Whisper has cached the model under
~/.cache/whisper, no network calls are made. That matches the agent's
existing "nothing leaves your machine" promise.

Deps come from the `[voice]` extra:
    pip install -e .[voice]

Both libraries are imported lazily so the rest of the package keeps
working when voice isn't installed.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Callable

from invisible_hands import config


class VoiceUnavailableError(RuntimeError):
    """Raised when voice mode is requested but its deps are missing.

    The message is intentionally actionable — it tells the user exactly
    which package to install rather than just dumping the import error.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports
# ─────────────────────────────────────────────────────────────────────────────
# Whisper + sounddevice are heavy and optional. We import them on first use
# (and cache the modules) so `python -m invisible_hands` starts up fast and
# works without the [voice] extra installed.

_whisper = None
_sounddevice = None
_numpy = None


def _import_whisper():
    """Return the cached `whisper` module, importing it on first call."""
    global _whisper
    if _whisper is None:
        try:
            import whisper  # type: ignore
        except ImportError as e:
            raise VoiceUnavailableError(
                "openai-whisper is not installed.\n"
                "  Install the voice extra:  pip install -e .[voice]\n"
                f"  (original import error: {e})"
            )
        _whisper = whisper
    return _whisper


def _import_sounddevice():
    """Return the cached `sounddevice` module, importing it on first call."""
    global _sounddevice
    if _sounddevice is None:
        try:
            import sounddevice  # type: ignore
        except (ImportError, OSError) as e:
            # OSError covers "PortAudio library not found" on systems where
            # sounddevice imports but its native dep is missing.
            raise VoiceUnavailableError(
                "sounddevice is not available.\n"
                "  Install it via the voice extra:  pip install -e .[voice]\n"
                "  Make sure PortAudio is installed:\n"
                "    macOS:    brew install portaudio\n"
                "    Windows:  ships in the wheel — try `pip install --force-reinstall sounddevice`\n"
                "    Linux:    apt install libportaudio2\n"
                f"  (original import error: {e})"
            )
        _sounddevice = sounddevice
    return _sounddevice


def _import_numpy():
    """numpy is already a hard dep of the package; this just caches it."""
    global _numpy
    if _numpy is None:
        import numpy  # noqa: F401

        _numpy = numpy
    return _numpy


def is_available() -> bool:
    """True if voice mode can actually run on this machine.

    We check both deps + the presence of at least one input device.
    Returns False (rather than raising) so the CLI can print a friendly
    "voice unavailable" line instead of crashing.
    """
    try:
        _import_whisper()
        sd = _import_sounddevice()
    except VoiceUnavailableError:
        return False

    try:
        # If there's no default input device this raises sd.PortAudioError.
        sd.check_input_settings(samplerate=config.VOICE_SAMPLE_RATE, channels=1)
    except Exception:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Recorder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Settings:
    """Tunables snapshot used by one VoiceRecorder instance.

    We snapshot these from `config` at construction so a long-lived
    recorder can't be silently re-tuned by a config edit mid-session.
    """
    sample_rate: int
    silence_seconds: float
    silence_rms: float
    max_seconds: int
    min_seconds: float
    model_name: str
    language: str | None


class VoiceRecorder:
    """Records mic audio, transcribes with Whisper, returns plain text.

    Construct once and reuse: the Whisper model is loaded lazily on the
    first transcription and cached on the instance, so subsequent calls
    skip the multi-second model load.

    Typical use (from cli.py):

        recorder = VoiceRecorder()
        task = recorder.listen_and_transcribe()
        if task:
            agent.run(task)
    """

    def __init__(
        self,
        *,
        sample_rate: int | None = None,
        silence_seconds: float | None = None,
        silence_rms: float | None = None,
        max_seconds: int | None = None,
        min_seconds: float | None = None,
        model_name: str | None = None,
        language: str | None = None,
    ):
        self.settings = _Settings(
            sample_rate=sample_rate or config.VOICE_SAMPLE_RATE,
            silence_seconds=silence_seconds if silence_seconds is not None
                            else config.VOICE_SILENCE_SECONDS,
            silence_rms=silence_rms if silence_rms is not None
                        else config.VOICE_SILENCE_RMS,
            max_seconds=max_seconds or config.VOICE_MAX_SECONDS,
            min_seconds=min_seconds if min_seconds is not None
                        else config.VOICE_MIN_SECONDS,
            model_name=model_name or config.VOICE_MODEL,
            language=language if language is not None else config.VOICE_LANGUAGE,
        )
        self._model = None  # loaded lazily

    # ---- public API --------------------------------------------------------

    def warm_up(self) -> None:
        """Pre-load the Whisper model so the first `listen_and_transcribe`
        call doesn't include a 2-10 second model-load pause.

        Optional — called by the CLI when --voice is passed so the user
        gets snappy feedback the very first time they hit the mic."""
        self._ensure_model()

    def listen_and_transcribe(
        self,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Record one utterance from the default mic and return the text.

        Args:
            on_status: optional callback invoked with short status messages
                       ('listening', 'transcribing', 'done'). The CLI uses
                       this to print a live indicator without coupling
                       this module to ANSI codes.

        Returns:
            The transcribed text, stripped of leading/trailing whitespace.
            Returns "" if the recording was too short or empty.

        Raises:
            VoiceUnavailableError: if Whisper or sounddevice can't be loaded.
        """
        notify = on_status or (lambda _msg: None)

        audio = self._record(on_status=notify)
        seconds = len(audio) / self.settings.sample_rate
        if seconds < self.settings.min_seconds:
            notify(f"too short ({seconds:.1f}s) - ignoring")
            return ""

        notify(f"transcribing {seconds:.1f}s of audio...")
        text = self._transcribe(audio).strip()
        notify("done")
        return text

    # ---- internals ---------------------------------------------------------

    def _ensure_model(self):
        """Load the Whisper model into memory the first time it's needed."""
        if self._model is None:
            whisper = _import_whisper()
            # whisper.load_model handles the cache + download under
            # ~/.cache/whisper. First call for a given model size hits the
            # network; subsequent calls are local.
            self._model = whisper.load_model(self.settings.model_name)
        return self._model

    def _record(self, *, on_status: Callable[[str], None]):
        """Capture mic audio into a numpy buffer until silence or timeout.

        Strategy:
            - Open an InputStream that delivers small float32 chunks.
            - Track recent chunk RMS values; stop when the rolling window
              has been below `silence_rms` for `silence_seconds`.
            - Run a parallel watcher thread on stdin so the user can press
              Enter to force-stop early ("done speaking, transcribe now").
        """
        sd = _import_sounddevice()
        np = _import_numpy()

        s = self.settings
        chunk_seconds = 0.1
        chunk_frames = int(s.sample_rate * chunk_seconds)
        silence_chunks_needed = max(1, int(s.silence_seconds / chunk_seconds))
        max_chunks = max(1, int(s.max_seconds / chunk_seconds))

        captured: list = []
        silent_streak = 0
        spoke_at_least_once = False

        # Press-Enter-to-stop watcher. We only attach it when stdin is a TTY,
        # otherwise (piped scripts, tests) we just rely on silence detection.
        stop_event = threading.Event()
        if sys.stdin and sys.stdin.isatty():
            threading.Thread(
                target=_wait_for_enter, args=(stop_event,), daemon=True,
            ).start()

        on_status("listening (speak now; pause or press Enter to stop)")

        with sd.InputStream(
            samplerate=s.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
        ) as stream:
            for _ in range(max_chunks):
                if stop_event.is_set():
                    break
                chunk, _overflow = stream.read(chunk_frames)
                # mono channel - flatten to 1D for the buffer + RMS calc.
                mono = chunk[:, 0]
                captured.append(mono.copy())

                rms = float(np.sqrt(np.mean(mono ** 2)))
                if rms >= s.silence_rms:
                    silent_streak = 0
                    spoke_at_least_once = True
                else:
                    silent_streak += 1
                    # Only count "silent enough to stop" AFTER we've heard
                    # the user say SOMETHING, otherwise we'd stop instantly
                    # in a quiet room before they even start speaking.
                    if spoke_at_least_once and silent_streak >= silence_chunks_needed:
                        break

        if not captured:
            return _import_numpy().zeros(0, dtype="float32")

        return _import_numpy().concatenate(captured)

    def _transcribe(self, audio) -> str:
        """Run the cached Whisper model over a float32 mono buffer."""
        model = self._ensure_model()

        # Whisper's `transcribe` accepts a numpy float32 mono array directly,
        # provided it's already at 16 kHz. We always record at 16 kHz so no
        # resampling is needed.
        kwargs: dict = {"fp16": False}
        if self.settings.language:
            kwargs["language"] = self.settings.language

        result = model.transcribe(audio, **kwargs)
        return str(result.get("text", "")).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for_enter(stop_event: threading.Event) -> None:
    """Block until the user presses Enter on stdin, then set stop_event.

    Used by `_record` to let the user end the recording manually instead
    of waiting for the silence detector. Wrapped in try/except so a
    closed stdin (e.g. backgrounded script) doesn't crash the recorder.
    """
    try:
        sys.stdin.readline()
    except Exception:
        pass
    stop_event.set()


__all__ = ["VoiceRecorder", "VoiceUnavailableError", "is_available"]
