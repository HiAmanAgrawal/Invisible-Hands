"""High-level agent loop: plan -> execute -> verify -> repeat.

This file contains the orchestration logic. It calls into:
    - llm.planner            for the step plan
    - parsing.steps          to detect deterministic ("simple") steps
    - clickers.chain         for any step that needs a click coordinate
    - controllers.input/apps to actually move the mouse / launch apps
    - llm.verifier           to confirm each step worked
    - reporting.reporter     for terminal output + JSON reports

Everything platform- or library-specific is hidden behind those imports —
this file should read like a slightly verbose version of the run mechanic
itself.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime

from invisible_hands import config
from invisible_hands.clickers.base import ClickRequest
from invisible_hands.clickers.chain import ClickerChain
from invisible_hands.clickers.native import NativeUIClicker
from invisible_hands.clickers.ocr import TesseractClicker
from invisible_hands.clickers.vision import VisionClicker
from invisible_hands.controllers import apps as apps_ctl
from invisible_hands.controllers import input as input_ctl
from invisible_hands.controllers.screen import (
    annotate_screenshot_with_axes,
    get_screen_size,
    screenshot_to_bytes,
    take_screenshot,
)
from invisible_hands.llm import planner as llm_planner
from invisible_hands.llm import verifier as llm_verifier
from invisible_hands.parsing.steps import is_simple_step
from invisible_hands.reporting import reporter as rep
from invisible_hands.reporting.reporter import c, format_action, log_llm_call


@dataclass
class AgentOptions:
    """Per-run toggles. Defaults come from config.py (which reads env vars).

    The CLI populates this dataclass from --flag arguments. Tests / scripts
    can construct it directly to override any subset of settings.
    """
    enable_verification: bool = field(default_factory=lambda: config.ENABLE_VERIFICATION)
    enable_ocr: bool = field(default_factory=lambda: config.ENABLE_OCR_CLICKER)
    enable_native: bool = field(default_factory=lambda: config.ENABLE_NATIVE_CLICKER)
    enable_vision: bool = field(default_factory=lambda: config.ENABLE_VISION_CLICKER)
    delay_after_action: float = field(default_factory=lambda: config.DELAY_AFTER_ACTION)
    max_retries_per_step: int = field(default_factory=lambda: config.MAX_RETRIES_PER_STEP)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_click_coords(action: dict, screen_w: int, screen_h: int) -> dict:
    """Reject obviously broken click coordinates and clamp borderline ones.

    The vision model occasionally returns (0, 0) or coordinates well outside
    the screen. Acting on those would either do nothing useful or hit some
    chrome the user didn't ask for. We bail with an error action in those
    cases so the agent can retry with a fresh screenshot.
    """
    x = action.get("x", -1)
    y = action.get("y", -1)
    if x <= 0 or y <= 0:
        return {"action": "error",
                "reason": f"Invalid coordinates ({x}, {y}) - zero or negative"}
    if x > screen_w * 1.1 or y > screen_h * 1.1:
        return {"action": "error",
                "reason": f"Coordinates ({x}, {y}) far outside screen {screen_w}x{screen_h}"}

    action["x"] = max(5, min(int(x), screen_w - 5))
    action["y"] = max(5, min(int(y), screen_h - 5))
    return action


# ─────────────────────────────────────────────────────────────────────────────
# Action execution (controller dispatch table)
# ─────────────────────────────────────────────────────────────────────────────

def execute_action(action: dict) -> bool:
    """Execute one action by dispatching to the appropriate controller.

    Returns True on success, False on error. Special-cases:
        "done"  -> the model said the step is already complete; treat as success.
        "error" -> the parser couldn't read the model's reply; fail loudly.
    """
    a = action.get("action")

    if a == "open_app":
        apps_ctl.open_app(action["app"])
        # Apps need a moment to launch before they accept keyboard input.
        time.sleep(3)
        apps_ctl.activate_app(action["app"])
    elif a == "activate_app":
        apps_ctl.activate_app(action["app"])
    elif a == "click":
        input_ctl.click(action["x"], action["y"])
    elif a == "double_click":
        input_ctl.double_click(action["x"], action["y"])
    elif a == "right_click":
        input_ctl.right_click(action["x"], action["y"])
    elif a == "type":
        input_ctl.type_text(action["text"])
    elif a == "press":
        input_ctl.press_key(action["key"])
    elif a == "hotkey":
        input_ctl.hotkey(*action["keys"])
    elif a == "scroll":
        input_ctl.scroll_screen(action.get("direction", "down"),
                                action.get("amount", 3))
    elif a == "move":
        input_ctl.move_to(action["x"], action["y"])
    elif a == "wait":
        time.sleep(min(action.get("seconds", 2), 10))
    elif a == "done":
        return True
    elif a == "error":
        print(f"         {c('Parse error:', 'red')} {action.get('reason', '?')}")
        return False
    else:
        print(f"         {c('Unknown action:', 'red')} {a}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """Runs a single user task end to end.

    Construct once per session (so the clicker chain is built once) and
    call `run(task)` for each user request.
    """

    def __init__(self, options: AgentOptions | None = None):
        self.options = options or AgentOptions()
        self.chain = self._build_clicker_chain()

    def _build_clicker_chain(self) -> ClickerChain:
        """Assemble the clicker chain in priority order.

        Cheap, deterministic strategies come first (OCR, Native UI). Vision
        is the universal fallback and so always sits at the end of the
        list when enabled.
        """
        clickers = [
            TesseractClicker() if self.options.enable_ocr else None,
            NativeUIClicker() if self.options.enable_native else None,
            VisionClicker() if self.options.enable_vision else None,
        ]
        return ClickerChain(clickers)

    # ---- verification helper ------------------------------------------------

    def _verify_and_log(
        self,
        step: str,
        action: dict,
        step_idx: int,
        screenshots_dir: str,
        step_report: dict,
    ) -> dict:
        """Take an after-screenshot, ask the verifier, log + record the result.

        Returns the verifier's parsed `{verified, confidence, observation, ...}`
        dict. Failures inside the verifier itself are caught and treated as
        "verified=true" so a flaky model can never block the agent forever.
        """
        time.sleep(self.options.delay_after_action)

        try:
            after_screenshot = take_screenshot()
            after_path = os.path.join(screenshots_dir, f"step_{step_idx:02d}_after.png")
            after_screenshot.save(after_path)
            step_report["screenshot_after"] = f"screenshots/step_{step_idx:02d}_after.png"

            after_bytes = screenshot_to_bytes(after_screenshot)
            action_summary = format_action(action)

            print(f"         {c('Verifying...', 'cyan')}")
            verify = llm_verifier.verify_step_completion(step, action_summary, after_bytes)
            vr = verify["result"]

            verified = vr.get("verified", True)
            confidence = vr.get("confidence", "?")
            observation = vr.get("observation", "")

            status_icon = c("PASS", "green") if verified else c("FAIL", "red")
            print(f"         {c('Verify:', 'bold')} {status_icon} "
                  f"({confidence} confidence) | {verify['duration_s']}s")
            if observation:
                print(f"         {c('Observed:', 'dim')} {observation[:120]}")

            suggestion = vr.get("suggestion")
            if suggestion and not verified:
                print(f"         {c('Suggestion:', 'yellow')} {suggestion[:120]}")

            step_report["verification"] = {
                "verified": verified,
                "confidence": confidence,
                "observation": observation,
                "suggestion": suggestion,
                "duration_s": verify["duration_s"],
                "raw_response": verify["raw_response"],
            }
            return vr
        except Exception as e:
            print(f"         {c('Verify error:', 'red')} {e}")
            step_report["verification"] = {"verified": True, "error": str(e)}
            return {"verified": True}

    # ---- main loop ----------------------------------------------------------

    def run(self, task: str) -> dict:
        """Plan + execute one user task. Returns the full structured report."""
        run_start = time.time()
        width, height = get_screen_size()

        report_dir, screenshots_dir = rep.make_report_dir(task)
        report = self._fresh_report(task, width, height)

        self._print_run_header(task, width, height)

        # ---- Phase 1: Plan ---------------------------------------------
        print(f"  {c('Phase 1: Planning...', 'magenta')}")
        print(f"         {c('Prompt:', 'dim')} \"Task: {task}\"")
        plan_result = llm_planner.create_plan(task)
        plan = plan_result["steps"]
        log_llm_call("Planner", plan_result)

        report["plan"] = plan
        report["planning"] = {
            "model": plan_result["model"],
            "duration_s": plan_result["duration_s"],
            "usage": plan_result["usage"],
            "thinking": plan_result["thinking"],
            "raw_response": plan_result["raw_response"],
        }

        print(f"\n  {c('Plan:', 'bold')} ({len(plan)} steps, "
              f"{plan_result['duration_s']}s)")
        for i, step in enumerate(plan, 1):
            print(f"    {c(f'{i}.', 'cyan')} {step}")
        print()

        # ---- Phase 2: Execute + Verify ---------------------------------
        print(f"  {c('Phase 2: Executing + Verifying...', 'magenta')}\n")

        for step_idx, step in enumerate(plan, 1):
            step_report = self._execute_one_step(
                step, step_idx, len(plan), width, height, screenshots_dir,
            )
            report["steps"].append(step_report)

        # ---- Save report ----------------------------------------------
        report["status"] = "completed"
        report["total_duration_s"] = round(time.time() - run_start, 2)
        report_file = rep.save_report(report_dir, report)

        input_ctl.play_done_sound()
        print(f"  {c('All steps completed!', 'green')}")
        print(f"  {c('Total time:', 'bold')} {report['total_duration_s']}s")
        print(f"  {c('Report saved:', 'dim')} {report_file}\n")

        return report

    # ---- per-step ----------------------------------------------------------

    def _execute_one_step(
        self,
        step: str,
        step_idx: int,
        total_steps: int,
        width: int,
        height: int,
        screenshots_dir: str,
    ) -> dict:
        """Execute (and optionally verify) one step. Returns its report dict."""
        step_start = time.time()
        print(f"  {c(f'[Step {step_idx}/{total_steps}]', 'cyan')} {step}")

        step_report: dict = {
            "step_number": step_idx,
            "description": step,
            "method": None,
            "action": None,
            "status": "pending",
            "retries": 0,
            "duration_s": 0,
            "llm_calls": [],
            "click_attempts": [],
            "verification": None,
        }

        # ---- 1) Try the simple-step shortcut -----------------------------
        simple = is_simple_step(step)
        if simple:
            self._run_simple_step(simple, step, step_idx, screenshots_dir, step_report)
        else:
            self._run_visual_step(step, step_idx, width, height,
                                  screenshots_dir, step_report)

        step_report["duration_s"] = round(time.time() - step_start, 2)
        print(f"         {c('Step time:', 'dim')} {step_report['duration_s']}s")
        input_ctl.play_step_sound()
        print()
        return step_report

    def _run_simple_step(
        self,
        action: dict,
        step: str,
        step_idx: int,
        screenshots_dir: str,
        step_report: dict,
    ) -> None:
        """Execute a deterministic action (Open/Type/Press/Wait/...)."""
        step_report["method"] = "direct"
        step_report["action"] = action

        print(f"         {c('Action:', 'bold')} "
              f"{c(format_action(action), 'yellow')} (direct)")
        success = execute_action(action)
        step_report["status"] = "success" if success else "error"

        # We skip verification for `wait` because there's nothing meaningful
        # to verify after a sleep — it always "succeeds".
        if (self.options.enable_verification
                and action.get("action") not in ("wait",)):
            vr = self._verify_and_log(step, action, step_idx,
                                      screenshots_dir, step_report)
            if not vr.get("verified", True):
                step_report["status"] = "verify_failed"
        else:
            time.sleep(self.options.delay_after_action)

    def _run_visual_step(
        self,
        step: str,
        step_idx: int,
        width: int,
        height: int,
        screenshots_dir: str,
        step_report: dict,
    ) -> None:
        """Run a step that needs visual reasoning — go through the clicker chain."""
        step_report["method"] = "click_chain"
        retries = 0

        while retries < self.options.max_retries_per_step:
            print(f"         Capturing screen...")
            screenshot = take_screenshot()

            # Always save the raw + grid screenshots — invaluable when
            # debugging "why did the model click there?".
            raw_path = os.path.join(screenshots_dir, f"step_{step_idx:02d}_before.png")
            screenshot.save(raw_path)
            step_report["screenshot_before"] = f"screenshots/step_{step_idx:02d}_before.png"

            annotated = annotate_screenshot_with_axes(screenshot)
            grid_path = os.path.join(screenshots_dir, f"step_{step_idx:02d}_axes.png")
            annotated.save(grid_path)
            step_report["screenshot_axes"] = f"screenshots/step_{step_idx:02d}_axes.png"

            request = ClickRequest(
                step=step,
                screenshot=screenshot,
                screen_size=(width, height),
            )

            print(f"         Trying click strategies: "
                  f"{', '.join(self.chain.names) or '(none enabled)'}")
            click_result, attempts = self.chain.find(request)
            step_report["click_attempts"].append(attempts)

            # Pretty-print every attempt (hit or miss) for visibility.
            for att in attempts:
                if att.get("found"):
                    print(f"         {c(att['name'].upper(), 'green')} -> "
                          f"({att['x']}, {att['y']}) "
                          f"conf={att['confidence']:.2f} "
                          f"in {att['duration_s']}s")
                elif "error" in att:
                    print(f"         {c(att['name'].upper(), 'red')} ERROR: "
                          f"{att['error']} ({att['duration_s']}s)")
                else:
                    print(f"         {c(att['name'].upper(), 'dim')} miss "
                          f"({att['duration_s']}s)")

            if not click_result.found:
                print(f"         {c('All click strategies missed - retrying...', 'red')}")
                retries += 1
                step_report["retries"] = retries
                continue

            # The vision clicker may return a non-click action (type/press/wait/done).
            # Pull it out of `extra` and execute that directly.
            action = click_result.extra.get("action")
            if action is None:
                action = {
                    "action": "click",
                    "x": click_result.x,
                    "y": click_result.y,
                    "reason": click_result.evidence,
                }

            llm_call = click_result.extra.get("llm_call")
            if llm_call:
                log_llm_call("Vision", llm_call)
                step_report["llm_calls"].append({
                    "type": "vision",
                    "model": llm_call["model"],
                    "duration_s": llm_call["duration_s"],
                    "usage": llm_call["usage"],
                    "thinking": llm_call["thinking"],
                    "raw_response": llm_call["raw_response"],
                    "prompt_sent": llm_call["prompt_sent"],
                })

            action_type = action.get("action", "?")
            if action_type in ("click", "double_click", "right_click"):
                action = _validate_click_coords(action, width, height)
                action_type = action.get("action", "?")

            step_report["action"] = action
            step_report["click_source"] = click_result.source

            print(f"         {c('Action:', 'bold')} "
                  f"{c(format_action(action), 'yellow')} "
                  f"(via {c(click_result.source, 'magenta')})")
            reason = action.get("reason") or click_result.evidence
            if reason:
                print(f"         {c('Reason:', 'dim')} {reason}")

            if action_type == "done":
                print(f"         {c('(step already done, skipping)', 'green')}")
                step_report["status"] = "skipped"
                return

            if action_type == "error":
                print(f"         {c('(retrying...)', 'red')}")
                retries += 1
                step_report["retries"] = retries
                continue

            success = execute_action(action)
            step_report["status"] = "success" if success else "error"

            if self.options.enable_verification:
                vr = self._verify_and_log(step, action, step_idx,
                                          screenshots_dir, step_report)
                if (not vr.get("verified", True)
                        and retries < self.options.max_retries_per_step - 1):
                    print(f"         {c('Verification failed - retrying step...', 'red')}")
                    retries += 1
                    step_report["retries"] = retries
                    continue
            else:
                time.sleep(self.options.delay_after_action)

            return

        if retries >= self.options.max_retries_per_step:
            step_report["status"] = "failed_max_retries"

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _fresh_report(task: str, width: int, height: int) -> dict:
        return {
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "screen": {"width": width, "height": height},
            "plan": [],
            "planning": {},
            "steps": [],
            "status": "running",
            "total_duration_s": 0,
        }

    def _print_run_header(self, task: str, width: int, height: int) -> None:
        print(f"\n  {c('Task:', 'bold')}   {task}")
        print(f"  {c('Screen:', 'dim')} {width}x{height}")
        print(f"  {c('Click chain:', 'dim')} "
              f"{' -> '.join(self.chain.names) or '(no clickers enabled!)'}")
        print(f"  {c('Safety:', 'dim')} Move mouse to top-left corner to abort\n")
