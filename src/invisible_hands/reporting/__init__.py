"""Run reporting + pretty terminal output.

Two responsibilities:
    1. Print colorful, easy-to-scan progress in the terminal while the
       agent runs (banners, step headers, LLM call summaries, success/fail
       icons).
    2. Persist a structured JSON report + screenshots for every run, so
       you can go back and inspect what happened later.

Both live in reporter.py to keep imports simple.
"""
