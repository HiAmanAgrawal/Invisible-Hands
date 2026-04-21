"""Pure-Python text parsing helpers used by the agent.

Two modules:
    actions  Extract a JSON action object from a model's free-form text reply.
    steps    Detect "simple" plan steps that can be executed without the
             vision model (Open, Type, Press, Wait, ...).

Both are dependency-free (just stdlib) and side-effect-free, so they're
trivial to unit-test in isolation.
"""
