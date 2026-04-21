"""LLM-facing modules.

The agent talks to a single local LLM server (LM Studio) but uses two
different models for different jobs:

    planner.py    Thinking model (qwen3.5-9b by default). Turns a free-form
                  user task into a numbered list of plan steps.
    executor.py   Vision-language model (qwen2.5-vl-7b-instruct by default).
                  Looks at a screenshot + step and returns a click coordinate
                  or other action.
    verifier.py   Same vision model, used to look at a post-action screenshot
                  and decide whether the step actually succeeded.
    client.py     Tiny shared HTTP wrapper around LM Studio's /api/v1/chat.
"""
