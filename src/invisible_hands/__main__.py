"""Allow `python -m invisible_hands` to launch the CLI.

This is the same entry point as the `invisible-hands` console script
declared in pyproject.toml; we just expose it both ways so users who
haven't run `pip install -e .` can still use the package directly.
"""

from invisible_hands.cli import main

if __name__ == "__main__":
    main()
