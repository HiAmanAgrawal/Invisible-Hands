"""Invisible Hands — a local-first desktop automation agent.

Top-level package. The interesting modules live in submodules:

    invisible_hands.cli         CLI entry point and REPL loop.
    invisible_hands.agent       Plan -> execute -> verify orchestration.
    invisible_hands.config      All tunable settings (env-var overridable).
    invisible_hands.llm.*       Talks to the local LLM (LM Studio).
    invisible_hands.clickers.*  Strategies for translating "click X" into pixels.
    invisible_hands.controllers.*  Mouse, keyboard, screen, and app launcher.
    invisible_hands.parsing.*   Text parsing helpers (steps, JSON actions).
    invisible_hands.reporting.* Run reports + pretty terminal output.

The package is also runnable as `python -m invisible_hands` thanks to __main__.py.
"""

__version__ = "0.2.0"
