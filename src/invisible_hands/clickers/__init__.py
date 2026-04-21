"""Click strategies — different ways of turning "Click the Sign In button"
into an actual (x, y) pixel coordinate.

The agent uses a chain of strategies that each try in turn:

    OCR (text-on-screen)        -> Best for buttons/links with visible labels.
                                   Works on every UI because it just reads
                                   pixels (cross-platform, no special perms).

    Native UI automation        -> Best when the OS exposes a real
                                   accessibility tree (most desktop apps,
                                   not custom-rendered web canvases). Returns
                                   pixel-perfect bounds straight from the OS.

    Vision-language model       -> Universal fallback. Works on anything the
                                   model can see, but is the slowest and
                                   most error-prone of the three.

The chain orchestrator (chain.py) tries each clicker in order; the first
one that returns a ClickResult wins. This way the cheap, deterministic
strategies handle the easy cases and only the genuinely visual targets
fall through to the LLM.
"""
