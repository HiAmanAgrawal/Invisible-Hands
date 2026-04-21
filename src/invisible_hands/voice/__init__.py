"""Voice input layer.

Optional subpackage that lets the user dictate tasks instead of typing them.
The implementation lives in `recorder.py` and depends on two third-party
libraries that ship in the `[voice]` extra:

    sounddevice  - cross-platform mic capture via PortAudio.
    whisper      - OpenAI's open-source speech-to-text model (vanilla,
                   runs fully offline once the model is cached).

Neither dependency is imported at package import time, so the rest of the
agent works fine without them. `is_available()` and `VoiceRecorder` both
fail with a clear, actionable error if the user tries to use voice mode
without installing the extra.
"""

from invisible_hands.voice.recorder import (  # noqa: F401
    VoiceRecorder,
    VoiceUnavailableError,
    is_available,
)
