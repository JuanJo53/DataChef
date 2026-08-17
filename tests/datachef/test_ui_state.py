from __future__ import annotations

from datetime import datetime, timezone

import pytest

from datachef.application import (
    CsvParserOptions,
    DataChefController,
    UploadFormat,
    UploadRequest,
)
from datachef.contracts import HumanDecision
from ui import state as ui_state


NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
CSV = (
    b"order_id,region,amount,ordered_on\n"
    b"1,North,10,2026-01-01\n"
    b"2,South,20,2026-01-02\n"
    b"2,South,20,2026-01-02\n"
    b"3,North,30,2026-01-03\n"
)


def _loaded_controller() -> DataChefController:
    controller = DataChefController(clock=lambda: NOW)
    controller.load_upload(
        UploadRequest(
            content=CSV,
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(encoding="utf-8-sig"),
        )
    )
    controller.diagnose()
    return controller


def test_every_owned_key_is_namespaced() -> None:
    owned = (
        ui_state.CONTROLLER,
        ui_state.RESET_REQUESTED,
        ui_state.LAST_RESULT,
        ui_state.PLAN_COMMAND,
        ui_state.HUMAN_COMMAND,
        ui_state.EXECUTION_COMMAND,
        ui_state.UPLOADER_WIDGET,
        ui_state.PREVIEW_WIDGET,
        ui_state.RESET_WIDGET,
        ui_state.GOAL_WIDGET,
        ui_state.CAST_REQUEST_WIDGET,
        ui_state.DEDUP_REQUEST_WIDGET,
    )

    assert all(key.startswith(ui_state.NAMESPACE) for key in owned)
    assert len(set(owned)) == len(owned)
    assert ui_state.PREVIEW_WIDGET.startswith(ui_state.WIDGET_NAMESPACE)


def test_revise_widget_keys_are_namespaced_and_distinct_from_the_intent_screen() -> None:
    revise = (
        ui_state.REVISE_ROW_LOSS_WIDGET,
        ui_state.REVISE_KEY_COLUMNS_WIDGET,
        ui_state.REVISE_CAST_REQUEST_WIDGET,
        ui_state.REVISE_DEDUP_REQUEST_WIDGET,
        ui_state.REVISE_SUBMIT_WIDGET,
    )
    intent_screen = (
        ui_state.ROW_LOSS_WIDGET,
        ui_state.KEY_COLUMNS_WIDGET,
        ui_state.CAST_REQUEST_WIDGET,
        ui_state.DEDUP_REQUEST_WIDGET,
        ui_state.SUBMIT_INTENT_WIDGET,
    )

    assert all(key.startswith(ui_state.WIDGET_NAMESPACE) for key in revise)
    assert not set(revise) & set(intent_screen)
    assert len(set(revise)) == len(revise)


def test_reset_clears_revise_widget_state() -> None:
    state: dict = {ui_state.CONTROLLER: _loaded_controller()}
    state[ui_state.REVISE_ROW_LOSS_WIDGET] = 42.0
    state[ui_state.REVISE_CAST_REQUEST_WIDGET] = ["price"]

    ui_state.reset_all(state)

    assert ui_state.REVISE_ROW_LOSS_WIDGET not in state
    assert ui_state.REVISE_CAST_REQUEST_WIDGET not in state


def test_controller_is_created_once_per_session_and_never_cached() -> None:
    state: dict = {}

    first = ui_state.get_controller(state)
    second = ui_state.get_controller(state)

    assert first is second
    assert state[ui_state.CONTROLLER] is first
    assert ui_state.get_controller({}) is not first


def test_command_id_is_minted_once_and_replayed() -> None:
    state: dict = {}

    first = ui_state.command_id(state, ui_state.PLAN_COMMAND)
    second = ui_state.command_id(state, ui_state.PLAN_COMMAND)
    third = ui_state.command_id(state, ui_state.PLAN_COMMAND)

    assert first == second == third
    assert state[ui_state.PLAN_COMMAND] == first
    assert first.strip()


def test_each_human_decision_owns_a_separate_command_slot() -> None:
    state: dict = {}

    approve = ui_state.command_id(
        state,
        ui_state.human_command_slot(HumanDecision.APPROVE),
    )
    reject = ui_state.command_id(
        state,
        ui_state.human_command_slot(HumanDecision.REJECT),
    )

    assert approve != reject
    assert ui_state.human_command_slot(HumanDecision.APPROVE) != (
        ui_state.human_command_slot(HumanDecision.REJECT)
    )


def test_clear_action_commands_drops_every_action_slot() -> None:
    state: dict = {"unrelated": "keep"}
    ui_state.command_id(state, ui_state.PLAN_COMMAND)
    ui_state.command_id(state, ui_state.EXECUTION_COMMAND)
    ui_state.command_id(state, ui_state.human_command_slot(HumanDecision.APPROVE))
    ui_state.command_id(state, ui_state.human_command_slot(HumanDecision.REJECT))

    ui_state.clear_action_commands(state)

    assert state == {"unrelated": "keep"}


def test_uploader_key_rotates_with_the_controller_generation() -> None:
    controller = _loaded_controller()
    before = controller.session.uploader_generation

    first = ui_state.uploader_key(before)
    controller.reset()
    after = controller.session.uploader_generation
    second = ui_state.uploader_key(after)

    assert after == before + 1
    assert first != second
    assert second.startswith(ui_state.UPLOADER_WIDGET)


def test_reset_clears_application_and_widget_state_but_keeps_the_controller() -> None:
    state: dict = {}
    controller = _loaded_controller()
    state[ui_state.CONTROLLER] = controller
    generation = controller.session.uploader_generation
    state[ui_state.uploader_key(generation)] = "stale-upload"
    state[ui_state.GOAL_WIDGET] = "an old goal"
    state[ui_state.PREVIEW_WIDGET] = True
    ui_state.command_id(state, ui_state.PLAN_COMMAND)
    ui_state.remember_result(state, controller.diagnose())
    state["not_ours"] = "survives"

    ui_state.reset_all(state)

    assert state[ui_state.CONTROLLER] is controller
    assert state["not_ours"] == "survives"
    assert set(state) == {ui_state.CONTROLLER, "not_ours"}
    session = controller.session
    assert session.source is None
    assert session.display_diagnostic_report is None
    assert session.intent is None
    assert session.workflow_runtime is None
    assert session.findings == ()
    assert session.pending_approval is None
    assert session.command_history == ()
    assert session.screen.value == "UPLOAD"
    assert session.uploader_generation == generation + 1
    assert ui_state.uploader_key(session.uploader_generation) not in state


def test_pending_reset_runs_once_and_only_when_requested() -> None:
    state: dict = {ui_state.CONTROLLER: _loaded_controller()}
    state[ui_state.GOAL_WIDGET] = "an old goal"

    assert ui_state.apply_pending_reset(state) is False
    assert state[ui_state.GOAL_WIDGET] == "an old goal"

    ui_state.request_reset(state)
    assert ui_state.apply_pending_reset(state) is True
    assert ui_state.GOAL_WIDGET not in state
    assert ui_state.RESET_REQUESTED not in state
    assert ui_state.apply_pending_reset(state) is False


def test_remember_result_round_trips_the_controller_outcome() -> None:
    state: dict = {}
    controller = _loaded_controller()

    result = ui_state.remember_result(state, controller.diagnose())

    assert ui_state.last_result(state) is result
    assert result.code
    assert ui_state.last_result({}) is None


@pytest.mark.parametrize("generation", (0, 1, 7))
def test_uploader_key_is_deterministic_for_a_generation(generation: int) -> None:
    assert ui_state.uploader_key(generation) == ui_state.uploader_key(generation)
    assert ui_state.uploader_key(generation).endswith(str(generation))
