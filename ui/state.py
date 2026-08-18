"""Streamlit session-state ownership for the DataChef product shell.

This module owns every ``st.session_state`` key name used by the product, so no
screen scatters string literals. It holds no business logic: the controller
decides everything, and this module only remembers which controller instance
belongs to the browser session and which command IDs have already been minted.

Command IDs are minted exactly once, at the moment the user commits to an
action, and are then replayed verbatim on every subsequent rerun so a refresh or
a double click can never cause a second planning, approval, or execution effect.

Functions here take a plain mutable mapping so they can be exercised without a
running Streamlit script; ``st.session_state`` satisfies that interface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import Any
from uuid import uuid4

from datachef.application import DataChefController
from datachef.contracts import HumanDecision


NAMESPACE = "datachef_"
WIDGET_NAMESPACE = "datachef_w_"

CONTROLLER = "datachef_controller"
AGENT_REGISTRY = "datachef_agent_registry"
RESET_REQUESTED = "datachef_reset_requested"
LAST_RESULT = "datachef_last_result"

PLAN_COMMAND = "datachef_cmd_plan"
HUMAN_COMMAND = "datachef_cmd_human"
EXECUTION_COMMAND = "datachef_cmd_execution"

UPLOADER_WIDGET = "datachef_w_uploader"
JSON_MODE_WIDGET = "datachef_w_json_mode"
GOAL_WIDGET = "datachef_w_goal"
DOWNSTREAM_WIDGET = "datachef_w_downstream"
KEY_COLUMNS_WIDGET = "datachef_w_key_columns"
REQUIRED_COLUMNS_WIDGET = "datachef_w_required_columns"
ROW_LOSS_WIDGET = "datachef_w_row_loss"
PII_WIDGET = "datachef_w_pii"
QUESTIONS_WIDGET = "datachef_w_questions"
SUGGESTED_QUESTIONS_WIDGET = "datachef_w_suggested_questions"
CAST_REQUEST_WIDGET = "datachef_w_cast_requests"
DEDUP_REQUEST_WIDGET = "datachef_w_dedup_requests"
REVISE_ROW_LOSS_WIDGET = "datachef_w_revise_row_loss"
REVISE_KEY_COLUMNS_WIDGET = "datachef_w_revise_key_columns"
REVISE_CAST_REQUEST_WIDGET = "datachef_w_revise_cast_requests"
REVISE_DEDUP_REQUEST_WIDGET = "datachef_w_revise_dedup_requests"
REVISE_SUBMIT_WIDGET = "datachef_w_revise_submit"
PREVIEW_WIDGET = "datachef_w_preview"
PREVIEW_SYNC = "datachef_preview_sync"
UPLOAD_PREVIEW_WIDGET = "datachef_w_upload_preview"
STAGE_NAV_WIDGET = "datachef_w_stage_nav"
RESET_WIDGET = "datachef_w_reset"
DIAGNOSE_WIDGET = "datachef_w_diagnose"
SUBMIT_INTENT_WIDGET = "datachef_w_submit_intent"
PREPARE_PLAN_WIDGET = "datachef_w_prepare_plan"
APPROVE_WIDGET = "datachef_w_approve"
REJECT_WIDGET = "datachef_w_reject"
EXECUTE_WIDGET = "datachef_w_execute"

ACTION_COMMAND_SLOTS = (PLAN_COMMAND, HUMAN_COMMAND, EXECUTION_COMMAND)


def live_mode_permitted(environment: Mapping[str, str] | None = None) -> bool:
    """Configuration decides, never the UI."""

    from datachef.agents.llm import decide_live_model

    return decide_live_model(environment).permitted


class AgentRegistry:
    """Keeps the most recent agent instances so the UI can read their trace.

    The controller builds a fresh planner per planning attempt and discards it,
    and ``WorkflowRuntime`` carries no trace field. Rather than change either —
    both are committed, independently reviewed code — the shell remembers the
    instance its own factory produced. Presentation bookkeeping only: nothing
    here influences a decision.
    """

    def __init__(self, live: bool) -> None:
        self.live = live
        self.planner = None
        self.reviewer = None

    def planner_factory(self):
        from datachef.agents import AgentPlanner

        self.planner = AgentPlanner()
        return self.planner

    def reviewer_factory(self):
        from datachef.agents import AgentReviewer

        self.reviewer = AgentReviewer()
        return self.reviewer

    @property
    def trace(self):
        return getattr(self.planner, "trace", None)

    @property
    def unsupported_requests(self) -> tuple[str, ...]:
        """Scope the planning crew reported it had no tool for. Display only."""

        return tuple(getattr(self.planner, "unsupported_requests", ()) or ())


def build_controller(
    environment: Mapping[str, str] | None = None,
    registry: "AgentRegistry | None" = None,
) -> DataChefController:
    """Wire the agent planner when configuration permits it, else the offline pair."""

    if not live_mode_permitted(environment):
        return DataChefController()
    registry = registry or AgentRegistry(live=True)
    return DataChefController(
        planner_factory=registry.planner_factory,
        reviewer_factory=registry.reviewer_factory,
    )


def get_controller(
    state: MutableMapping[str, Any],
    *,
    factory: Callable[[], DataChefController] = build_controller,
) -> DataChefController:
    """Return the one controller owned by this browser session.

    Deliberately not cached: ``st.cache_resource`` is shared across browser
    sessions and would hand one user's dataset to another.
    """

    controller = state.get(CONTROLLER)
    if controller is None:
        registry = AgentRegistry(live=live_mode_permitted())
        state[AGENT_REGISTRY] = registry
        controller = factory() if factory is not build_controller else factory(
            None, registry
        )
        state[CONTROLLER] = controller
    return controller


def agent_registry(state: MutableMapping[str, Any]) -> "AgentRegistry | None":
    return state.get(AGENT_REGISTRY)


def uploader_key(uploader_generation: int) -> str:
    """Bind the uploader widget to the generation the controller reports."""

    return f"{UPLOADER_WIDGET}_{uploader_generation}"


def human_command_slot(decision: HumanDecision) -> str:
    """Give each decision its own slot so approve and reject never collide."""

    return f"{HUMAN_COMMAND}_{decision.value}"


def command_id(state: MutableMapping[str, Any], slot: str) -> str:
    """Mint one command ID per slot and replay it on every later rerun."""

    existing = state.get(slot)
    if isinstance(existing, str) and existing.strip():
        return existing
    minted = str(uuid4())
    state[slot] = minted
    return minted


def clear_action_commands(state: MutableMapping[str, Any]) -> None:
    """Drop minted command IDs once the controller has invalidated their work."""

    for key in list(state.keys()):
        text = str(key)
        if text in ACTION_COMMAND_SLOTS or text.startswith(f"{HUMAN_COMMAND}_"):
            del state[key]


def remember_result(state: MutableMapping[str, Any], result: Any) -> Any:
    state[LAST_RESULT] = result
    return result


def last_result(state: MutableMapping[str, Any]) -> Any:
    return state.get(LAST_RESULT)


def request_reset(state: MutableMapping[str, Any]) -> None:
    """Flag a reset so it runs before any widget is instantiated next rerun."""

    state[RESET_REQUESTED] = True


def apply_pending_reset(state: MutableMapping[str, Any]) -> bool:
    """Perform a flagged reset. Must run before any widget renders."""

    if not state.get(RESET_REQUESTED):
        return False
    del state[RESET_REQUESTED]
    reset_all(state)
    return True


def reset_all(state: MutableMapping[str, Any]) -> None:
    """Clear application state and widget state, and rotate the uploader key."""

    controller = state.get(CONTROLLER)
    if controller is not None:
        controller.reset()
    for key in list(state.keys()):
        text = str(key)
        if text == CONTROLLER:
            continue
        if text.startswith(NAMESPACE):
            del state[key]


__all__ = [
    "AGENT_REGISTRY",
    "ACTION_COMMAND_SLOTS",
    "AgentRegistry",
    "APPROVE_WIDGET",
    "CAST_REQUEST_WIDGET",
    "CONTROLLER",
    "DEDUP_REQUEST_WIDGET",
    "DIAGNOSE_WIDGET",
    "DOWNSTREAM_WIDGET",
    "EXECUTE_WIDGET",
    "EXECUTION_COMMAND",
    "GOAL_WIDGET",
    "HUMAN_COMMAND",
    "JSON_MODE_WIDGET",
    "KEY_COLUMNS_WIDGET",
    "LAST_RESULT",
    "NAMESPACE",
    "PII_WIDGET",
    "PLAN_COMMAND",
    "PREPARE_PLAN_WIDGET",
    "PREVIEW_SYNC",
    "PREVIEW_WIDGET",
    "QUESTIONS_WIDGET",
    "REJECT_WIDGET",
    "REQUIRED_COLUMNS_WIDGET",
    "RESET_REQUESTED",
    "RESET_WIDGET",
    "REVISE_CAST_REQUEST_WIDGET",
    "REVISE_DEDUP_REQUEST_WIDGET",
    "REVISE_KEY_COLUMNS_WIDGET",
    "REVISE_ROW_LOSS_WIDGET",
    "REVISE_SUBMIT_WIDGET",
    "ROW_LOSS_WIDGET",
    "STAGE_NAV_WIDGET",
    "SUBMIT_INTENT_WIDGET",
    "SUGGESTED_QUESTIONS_WIDGET",
    "UPLOADER_WIDGET",
    "UPLOAD_PREVIEW_WIDGET",
    "WIDGET_NAMESPACE",
    "agent_registry",
    "apply_pending_reset",
    "build_controller",
    "clear_action_commands",
    "command_id",
    "get_controller",
    "human_command_slot",
    "last_result",
    "live_mode_permitted",
    "remember_result",
    "request_reset",
    "reset_all",
    "uploader_key",
]
