from __future__ import annotations

import ast
from datetime import datetime, timezone
import inspect
from pathlib import Path
import sys

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import datachef.application.controller as controller_module

from datachef.application import (
    CsvParserOptions,
    DataChefController,
    JsonRecordsParserOptions,
    ParsedDataset,
    RequestedTransformation,
    ScreenId,
    UploadFormat,
    UploadPolicy,
    UploadRequest,
    parse_upload,
    source_metadata_for_upload,
)
from datachef.contracts import (
    CastColumnParameters,
    CastTarget,
    DeduplicateByKeysParameters,
    DownstreamUse,
    HumanDecision,
    KeepPolicy,
    OperationType,
    PIIHandling,
    QAStatus,
    ReviewerDecision,
    UserIntent,
    WorkflowStage,
)
from datachef.diagnostics import identify_dataset
from datachef.planning import RuleBasedPlanner, SequenceReviewer
from datachef.workflow import WorkflowRuntime, execute_workflow, prepare_workflow


FIXTURE = Path(__file__).parents[1] / "fixtures" / "phase1b_orders.csv"
NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


def _csv_request(
    content: bytes = b"order_id,region,amount\n1,North,10\n2,South,20\n2,South,20\n",
    *,
    encoding: str = "utf-8-sig",
) -> UploadRequest:
    return UploadRequest(
        content=content,
        declared_suffix=".csv",
        format=UploadFormat.CSV,
        parser_options=CsvParserOptions(encoding=encoding),
    )


def _json_request(content: bytes) -> UploadRequest:
    return UploadRequest(
        content=content,
        declared_suffix=".json",
        format=UploadFormat.JSON_RECORDS,
        parser_options=JsonRecordsParserOptions(),
    )


def _intent(
    *,
    intent_id: str = "intent-controller",
    keys: tuple[str, ...] = (),
    row_loss: float = 40,
    pii: PIIHandling = PIIHandling.NONE,
    prose_requests: tuple[str, ...] = (),
) -> UserIntent:
    return UserIntent(
        intent_id=intent_id,
        user_goal="Prepare the table for analysis.",
        downstream_use=DownstreamUse.ANALYSIS,
        selected_key_columns=keys,
        required_columns=("order_id",),
        acceptable_row_loss_pct=row_loss,
        pii_handling=pii,
        explicit_requested_transformations=prose_requests,
    )


def _cast_request(column: str, diagnostic_issue_id: str | None = None):
    return RequestedTransformation(
        request_id=f"request-cast-{column}",
        operation_type=OperationType.CAST_COLUMN,
        target_columns=(column,),
        parameters=CastColumnParameters(target_type=CastTarget.NUMERIC),
        diagnostic_issue_id=diagnostic_issue_id,
    )


def _dedup_request(keys: tuple[str, ...] = ("order_id",)):
    return RequestedTransformation(
        request_id="request-dedup-keys",
        operation_type=OperationType.DEDUPLICATE_BY_KEYS,
        target_columns=keys,
        parameters=DeduplicateByKeysParameters(keys=keys, keep=KeepPolicy.FIRST),
    )


def _loaded_controller(
    request: UploadRequest | None = None,
    **kwargs,
) -> DataChefController:
    controller = DataChefController(clock=lambda: NOW, **kwargs)
    assert controller.load_upload(request or _csv_request()).changed
    assert controller.diagnose().changed
    return controller


def _approve_and_execute(controller: DataChefController) -> None:
    approved = controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="approve-1",
    )
    assert approved.changed
    executed = controller.execute_current_plan(command_id="execute-1")
    assert executed.changed


def _assert_session_unchanged(controller: DataChefController, before) -> None:
    after = controller.session
    assert after.revision == before.revision
    assert after.command_history == before.command_history
    assert after.pending_approval == before.pending_approval
    assert after.findings == before.findings
    assert after.source is not None and before.source is not None
    assert after.source.identity == before.source.identity
    assert_frame_equal(after.source.raw_copy(), before.source.raw_copy())
    if before.workflow_runtime is None:
        assert after.workflow_runtime is None
        return
    assert after.workflow_runtime is not None
    assert after.workflow_runtime.state == before.workflow_runtime.state
    assert_frame_equal(
        after.workflow_runtime.raw_dataframe,
        before.workflow_runtime.raw_dataframe,
    )


def test_complete_csv_to_gold_happy_path_with_key_deduplication() -> None:
    controller = _loaded_controller()
    original = controller.session.source.raw_copy()
    controller.submit_intent(
        _intent(keys=("order_id",)),
        (_dedup_request(),),
    )

    planned = controller.prepare_plan(command_id="plan-1")
    _approve_and_execute(controller)

    runtime = controller.session.workflow_runtime
    assert planned.screen is ScreenId.APPROVAL
    assert runtime is not None
    assert runtime.state.stage is WorkflowStage.QA_PASSED
    assert runtime.state.qa_report is not None
    assert runtime.state.qa_report.status is QAStatus.PASS
    assert runtime.gold_dataframe is not None
    assert len(runtime.gold_dataframe) == 2
    assert_frame_equal(controller.session.source.raw_copy(), original)


def test_numeric_cast_request_reconciles_by_target_type_and_display_issue() -> None:
    request = _json_request(
        b'[{"order_id":1,"amount_text":"10"},{"order_id":2,"amount_text":"20"}]'
    )
    controller = _loaded_controller(request)
    report = controller.session.display_diagnostic_report
    assert report is not None
    issue = next(
        item
        for item in report.issues
        if item.kind.value == "CANDIDATE_TYPE_CONVERSION"
    )
    controller.submit_intent(_intent(), (_cast_request("amount_text", issue.issue_id),))

    result = controller.prepare_plan(command_id="plan-cast")

    assert result.findings == ()
    runtime = controller.session.workflow_runtime
    assert runtime is not None
    assert runtime.state.transformation_plan is not None
    assert runtime.state.transformation_plan.operations[0].operation_type is OperationType.CAST_COLUMN


def test_key_dedup_request_reconciles_exact_ordered_keys() -> None:
    controller = _loaded_controller()
    controller.submit_intent(
        _intent(keys=("order_id",)),
        (_dedup_request(("order_id",)),),
    )

    result = controller.prepare_plan(command_id="plan-dedup")

    assert result.findings == ()


def test_unplanned_request_is_blocking_and_prevents_approval() -> None:
    controller = _loaded_controller()
    controller.submit_intent(_intent(), (_cast_request("amount"),))
    planned = controller.prepare_plan(command_id="plan-unplanned")

    refused = controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="approve-blocked",
    )

    assert any(
        finding.code == "REQUEST_NOT_PLANNED" and finding.blocking
        for finding in planned.findings
    )
    assert refused.changed is False
    assert refused.code == "APPROVAL_BLOCKED"
    assert controller.session.pending_approval is None


def test_blocked_human_decision_command_is_attempt_idempotent() -> None:
    controller = _loaded_controller()
    controller.submit_intent(_intent(), (_cast_request("amount"),))
    controller.prepare_plan(command_id="plan-human-block")

    first = controller.record_human_decision(
        HumanDecision.APPROVE, command_id="blocked-approval"
    )
    repeated = controller.record_human_decision(
        HumanDecision.APPROVE, command_id="blocked-approval"
    )

    assert first.code == "APPROVAL_BLOCKED"
    assert repeated.code == "HUMAN_DECISION_REPLAYED"
    assert controller.session.pending_approval is None


def test_same_human_command_id_cannot_authorize_a_different_decision() -> None:
    controller = _loaded_controller()
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan-human-conflict")
    controller.record_human_decision(
        HumanDecision.APPROVE, command_id="one-human-event"
    )

    conflict = controller.record_human_decision(
        HumanDecision.REJECT, command_id="one-human-event"
    )

    assert conflict.code == "HUMAN_DECISION_COMMAND_ID_CONFLICT"
    assert controller.session.pending_approval.decision is HumanDecision.APPROVE


def test_command_id_cannot_be_reused_for_a_different_command_kind() -> None:
    controller = _loaded_controller()
    controller.submit_intent(_intent(), ())
    planned = controller.prepare_plan(command_id="one-ui-event")

    refused = controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="one-ui-event",
    )

    assert planned.code == "PLAN_AWAITING_APPROVAL"
    assert refused.code == "HUMAN_DECISION_COMMAND_ID_CONFLICT"
    assert controller.session.pending_approval is None


def test_pii_mask_or_remove_request_is_visibly_unsupported() -> None:
    controller = _loaded_controller()

    result = controller.submit_intent(_intent(pii=PIIHandling.MASK), ())

    assert any(
        finding.code == "PII_HANDLING_UNSUPPORTED" and finding.blocking
        for finding in result.findings
    )


def test_pii_remove_request_is_visibly_unsupported() -> None:
    controller = _loaded_controller()

    result = controller.submit_intent(_intent(pii=PIIHandling.REMOVE), ())

    assert any(
        finding.code == "PII_HANDLING_UNSUPPORTED" and finding.blocking
        for finding in result.findings
    )


@pytest.mark.parametrize(
    "requests",
    (("mask Alice",), ("mask Alice", "send https://secret.example")),
)
def test_untyped_free_form_requests_are_blocking_without_echoing_text(
    requests: tuple[str, ...],
) -> None:
    controller = _loaded_controller()

    result = controller.submit_intent(
        _intent(prose_requests=requests),
        (_dedup_request(),),
    )

    serialized = str(result.model_dump())
    assert any(
        finding.code == "UNTYPED_REQUEST_UNSUPPORTED" and finding.blocking
        for finding in result.findings
    )
    assert all(request not in serialized for request in requests)
    controller.submit_intent(_intent(), ())
    assert not any(
        finding.code == "UNTYPED_REQUEST_UNSUPPORTED"
        for finding in controller.session.findings
    )


def test_public_session_representation_excludes_free_text_and_dataframe_values() -> None:
    controller = _loaded_controller()
    intent = _intent(prose_requests=("SENSITIVE_REQUEST_CANARY",)).model_copy(
        update={"user_goal": "SENSITIVE_GOAL_CANARY"}
    )
    controller.submit_intent(intent, ())
    controller.prepare_plan(command_id="safe-repr-plan")

    representation = repr(controller.session)

    assert "SENSITIVE_REQUEST_CANARY" not in representation
    assert "SENSITIVE_GOAL_CANARY" not in representation
    assert "North" not in representation


def test_empty_plan_executes_copy_and_qa_proves_zero_change() -> None:
    controller = _loaded_controller(
        _csv_request(b"order_id,region,amount\n1,North,10\n2,South,20\n")
    )
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="empty-plan")
    runtime_before = controller.session.workflow_runtime
    assert runtime_before is not None
    assert runtime_before.state.transformation_plan is not None
    assert runtime_before.state.transformation_plan.operations == ()
    raw = controller.session.source.raw_copy()

    _approve_and_execute(controller)

    runtime = controller.session.workflow_runtime
    assert runtime is not None
    assert runtime.state.stage is WorkflowStage.QA_PASSED
    assert runtime.state.execution_result is not None
    assert runtime.state.execution_result.operation_records == ()
    assert runtime.state.qa_report is not None
    assert runtime.state.qa_report.dtype_changes == ()
    assert runtime.state.qa_report.null_count_changes == ()
    assert runtime.state.qa_report.row_loss_pct == 0
    assert runtime.gold_dataframe is not None
    assert runtime.gold_dataframe is not runtime.raw_dataframe
    assert_frame_equal(runtime.gold_dataframe, raw)


@pytest.mark.parametrize(
    ("decisions", "expected_error"),
    (
        ((ReviewerDecision.REJECT,), "PLAN_REJECTED_BY_REVIEWER"),
        ((ReviewerDecision.REVISE,) * 3, "PLANNING_ATTEMPTS_EXHAUSTED"),
    ),
)
def test_reviewer_rejection_and_exhaustion_stop_cleanly(
    decisions: tuple[ReviewerDecision, ...],
    expected_error: str,
) -> None:
    controller = _loaded_controller(
        reviewer_factory=lambda: SequenceReviewer(decisions)
    )
    controller.submit_intent(_intent(), ())

    result = controller.prepare_plan(command_id="review-route")

    runtime = controller.session.workflow_runtime
    assert runtime is not None
    assert runtime.state.stage is WorkflowStage.PLAN_REJECTED
    assert runtime.state.last_error_code == expected_error
    assert result.screen is ScreenId.PLAN


def test_reviewer_revision_then_acceptance_uses_two_attempts() -> None:
    controller = _loaded_controller(
        reviewer_factory=lambda: SequenceReviewer(
            (ReviewerDecision.REVISE, ReviewerDecision.ACCEPT)
        )
    )
    controller.submit_intent(_intent(), ())

    controller.prepare_plan(command_id="review-revision")

    runtime = controller.session.workflow_runtime
    assert runtime is not None
    assert runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert runtime.state.planning_attempts == 2
    assert tuple(item.decision for item in runtime.state.review_history) == (
        ReviewerDecision.REVISE,
        ReviewerDecision.ACCEPT,
    )


def test_approval_is_exact_and_stale_approval_cannot_execute() -> None:
    controller = _loaded_controller()
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan-exact")
    controller.record_human_decision(HumanDecision.APPROVE, command_id="approval-exact")
    approval = controller.session.pending_approval
    runtime = controller.session.workflow_runtime
    assert approval is not None and runtime is not None
    plan = runtime.state.transformation_plan
    assert plan is not None
    assert approval.plan_id == plan.plan_id
    assert approval.plan_version == plan.version
    assert approval.approved_operation_ids == tuple(op.operation_id for op in plan.operations)

    controller.revise_intent(_intent(intent_id="intent-revised"), ())
    refused = controller.execute_current_plan(command_id="stale-execution")

    assert refused.changed is False
    assert refused.code == "APPROVAL_REQUIRED"
    assert controller.session.workflow_runtime is None


def test_repeated_prepare_and_execute_commands_are_idempotent() -> None:
    controller = _loaded_controller()
    controller.submit_intent(_intent(), ())
    first_plan = controller.prepare_plan(command_id="plan-idempotent")
    runtime = controller.session.workflow_runtime
    second_plan = controller.prepare_plan(command_id="plan-idempotent")
    assert first_plan.changed
    assert not second_plan.changed
    assert controller.session.workflow_runtime.state == runtime.state
    controller.record_human_decision(HumanDecision.APPROVE, command_id="approve")
    first_execution = controller.execute_current_plan(command_id="execute-idempotent")
    completed = controller.session.workflow_runtime
    second_execution = controller.execute_current_plan(command_id="execute-idempotent")

    assert first_execution.changed
    assert not second_execution.changed
    assert controller.session.workflow_runtime.state == completed.state


def test_historical_command_id_remains_reserved_after_newer_attempts() -> None:
    calls = {"prepare": 0}

    def counted_prepare(source, intent, planner, reviewer):
        calls["prepare"] += 1
        return prepare_workflow(source, intent, planner, reviewer)

    controller = _loaded_controller(prepare_service=counted_prepare)
    controller.submit_intent(
        _intent(prose_requests=("password=hunter2",)),
        (),
    )
    controller.prepare_plan(command_id="initial-plan")
    first_blocked = controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="historical-id",
    )
    second_blocked = controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="newer-blocked-id",
    )

    before = controller.session
    reused = controller.prepare_plan(command_id="historical-id")

    assert first_blocked.code == second_blocked.code == "APPROVAL_BLOCKED"
    assert reused.code == "PLAN_COMMAND_ID_CONFLICT"
    assert calls == {"prepare": 1}
    _assert_session_unchanged(controller, before)
    assert tuple(item.command_id for item in controller.session.command_history) == (
        "initial-plan",
        "historical-id",
        "newer-blocked-id",
    )
    assert "hunter2" not in repr(controller.session)


@pytest.mark.parametrize(
    ("first_kind", "second_kind", "expected_code"),
    (
        ("prepare", "human", "HUMAN_DECISION_COMMAND_ID_CONFLICT"),
        ("prepare", "execute", "EXECUTION_COMMAND_ID_CONFLICT"),
        ("human", "prepare", "PLAN_COMMAND_ID_CONFLICT"),
        ("human", "execute", "EXECUTION_COMMAND_ID_CONFLICT"),
        ("execute", "prepare", "PLAN_COMMAND_ID_CONFLICT"),
        ("execute", "human", "HUMAN_DECISION_COMMAND_ID_CONFLICT"),
    ),
)
def test_command_ids_are_globally_unique_across_command_kinds(
    first_kind: str,
    second_kind: str,
    expected_code: str,
) -> None:
    calls = {"prepare": 0, "execute": 0}

    def counted_prepare(source, intent, planner, reviewer):
        calls["prepare"] += 1
        return prepare_workflow(source, intent, planner, reviewer)

    def counted_execute(runtime, approval):
        calls["execute"] += 1
        return execute_workflow(runtime, approval)

    controller = _loaded_controller(
        prepare_service=counted_prepare,
        execute_service=counted_execute,
    )
    controller.submit_intent(_intent(), ())
    shared = "cross-kind-command"
    plan_id = shared if first_kind == "prepare" else "setup-plan"
    controller.prepare_plan(command_id=plan_id)
    if first_kind == "human":
        controller.record_human_decision(HumanDecision.APPROVE, command_id=shared)
    elif first_kind == "execute":
        controller.record_human_decision(
            HumanDecision.APPROVE,
            command_id="setup-approval",
        )
        controller.execute_current_plan(command_id=shared)

    if second_kind == "prepare":
        action = lambda: controller.prepare_plan(command_id=shared)
    elif second_kind == "human":
        action = lambda: controller.record_human_decision(
            HumanDecision.APPROVE,
            command_id=shared,
        )
    else:
        if first_kind == "prepare":
            controller.record_human_decision(
                HumanDecision.APPROVE,
                command_id="setup-approval",
            )
        action = lambda: controller.execute_current_plan(command_id=shared)

    before_calls = calls.copy()
    before = controller.session
    result = action()

    assert result.code == expected_code
    assert calls == before_calls
    _assert_session_unchanged(controller, before)


def test_preview_does_not_clear_command_history() -> None:
    controller = _loaded_controller()
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="retained-plan-command")
    before = controller.session.command_history

    controller.set_preview_enabled(True)

    assert controller.session.command_history == before


def test_material_intent_change_and_full_reset_clear_command_history() -> None:
    controller = _loaded_controller()
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="old-plan-command")

    controller.revise_intent(_intent(intent_id="changed-material-intent"), ())
    assert controller.session.command_history == ()
    controller.prepare_plan(command_id="old-plan-command")
    assert controller.session.command_history

    controller.reset()
    assert controller.session.command_history == ()


def test_same_kind_command_id_with_different_binding_is_rejected() -> None:
    controller = _loaded_controller()
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="binding-plan")
    controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="bound-human-command",
    )
    before = controller.session

    result = controller.record_human_decision(
        HumanDecision.REJECT,
        command_id="bound-human-command",
    )

    assert result.code == "HUMAN_DECISION_COMMAND_ID_CONFLICT"
    _assert_session_unchanged(controller, before)


def test_human_decision_replay_remains_idempotent_after_terminal_execution() -> None:
    controller = _loaded_controller()
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="terminal-replay-plan")
    controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="terminal-replay-human",
    )
    controller.execute_current_plan(command_id="terminal-replay-execute")
    before = controller.session

    replayed = controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="terminal-replay-human",
    )

    assert replayed.code == "HUMAN_DECISION_REPLAYED"
    _assert_session_unchanged(controller, before)


def test_human_rejection_uses_only_fixed_phase1a_rejection_path(monkeypatch) -> None:
    calls = {"injected": 0, "fixed": 0, "verifier": 0}
    fixed_execute = execute_workflow

    def injected(runtime, approval):
        del runtime, approval
        calls["injected"] += 1
        raise AssertionError("rejection must bypass the injected execution service")

    def fixed(runtime, approval):
        calls["fixed"] += 1
        return fixed_execute(runtime, approval)

    def verifier(*args, **kwargs):
        del args, kwargs
        calls["verifier"] += 1
        raise AssertionError("rejection is not a completed-runtime candidate")

    monkeypatch.setattr(controller_module, "execute_workflow", fixed)
    monkeypatch.setattr(
        controller_module,
        "verify_completed_workflow_runtime",
        verifier,
    )

    controller = _loaded_controller(execute_service=injected)
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan-reject")
    controller.record_human_decision(HumanDecision.REJECT, command_id="reject")

    first = controller.execute_current_plan(command_id="execute-reject")
    repeated = controller.execute_current_plan(command_id="execute-reject")

    assert calls == {"injected": 0, "fixed": 1, "verifier": 0}
    assert first.code == "EXECUTION_COMPLETED"
    assert repeated.code == "EXECUTION_COMMAND_REPLAYED"
    assert controller.session.workflow_runtime.state.stage is WorkflowStage.PLAN_REJECTED
    assert controller.session.workflow_runtime.transformed_dataframe is None
    assert controller.session.workflow_runtime.gold_dataframe is None
    assert controller.session.screen is ScreenId.PLAN


def test_human_rejection_fixed_path_failure_preserves_pre_rejection_runtime(
    monkeypatch,
) -> None:
    controller = _loaded_controller(execute_service=lambda *_: pytest.fail("injected"))
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan-reject-failure")
    controller.record_human_decision(
        HumanDecision.REJECT,
        command_id="reject-failure",
    )
    before = controller.session

    def failing_fixed(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("private rejection detail")

    monkeypatch.setattr(controller_module, "execute_workflow", failing_fixed)
    result = controller.execute_current_plan(command_id="execute-reject-failure")

    assert result.code == "HUMAN_REJECTION_FAILURE"
    assert controller.session.workflow_runtime.state == before.workflow_runtime.state
    assert controller.session.workflow_runtime.transformed_dataframe is None
    assert controller.session.workflow_runtime.gold_dataframe is None
    assert "private rejection detail" not in repr(result)


@pytest.mark.parametrize("invalid_result", (object(), None))
def test_human_rejection_refuses_invalid_fixed_path_result(
    monkeypatch,
    invalid_result,
) -> None:
    calls = {"injected": 0, "verifier": 0}

    def injected(*args, **kwargs):
        del args, kwargs
        calls["injected"] += 1
        return object()

    def verifier(*args, **kwargs):
        del args, kwargs
        calls["verifier"] += 1
        return None

    monkeypatch.setattr(
        controller_module,
        "execute_workflow",
        lambda runtime, approval: invalid_result,
    )
    monkeypatch.setattr(
        controller_module,
        "verify_completed_workflow_runtime",
        verifier,
    )
    controller = _loaded_controller(execute_service=injected)
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="invalid-reject-plan")
    controller.record_human_decision(
        HumanDecision.REJECT,
        command_id="invalid-reject-decision",
    )
    before = controller.session

    result = controller.execute_current_plan(command_id="invalid-reject-execute")

    assert result.code == "HUMAN_REJECTION_EVIDENCE_INVALID"
    assert calls == {"injected": 0, "verifier": 0}
    assert controller.session.workflow_runtime.state == before.workflow_runtime.state
    assert controller.session.workflow_runtime.gold_dataframe is None


@pytest.mark.parametrize(
    "decisions",
    (
        (ReviewerDecision.REJECT,),
        (ReviewerDecision.REVISE,) * 3,
    ),
)
def test_reviewer_stopped_routes_never_reach_any_execution_boundary(
    monkeypatch,
    decisions: tuple[ReviewerDecision, ...],
) -> None:
    calls = {"injected": 0, "fixed": 0, "verifier": 0}

    def forbidden(name):
        def call(*args, **kwargs):
            del args, kwargs
            calls[name] += 1
            raise AssertionError(f"{name} must not run")

        return call

    monkeypatch.setattr(controller_module, "execute_workflow", forbidden("fixed"))
    monkeypatch.setattr(
        controller_module,
        "verify_completed_workflow_runtime",
        forbidden("verifier"),
    )
    controller = _loaded_controller(
        reviewer_factory=lambda: SequenceReviewer(decisions),
        execute_service=forbidden("injected"),
    )
    controller.submit_intent(_intent(), ())

    planned = controller.prepare_plan(command_id="stopped-plan")
    attempted = controller.execute_current_plan(command_id="stopped-execute")

    assert planned.screen is ScreenId.PLAN
    assert attempted.code == "APPROVAL_REQUIRED"
    assert calls == {"injected": 0, "fixed": 0, "verifier": 0}
    assert controller.session.workflow_runtime.state.stage is WorkflowStage.PLAN_REJECTED


def test_executor_exception_is_sanitized_and_does_not_replace_runtime() -> None:
    def failing_execute(runtime, approval):
        del runtime, approval
        raise RuntimeError("secret provider-like detail")

    controller = _loaded_controller(execute_service=failing_execute)
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan-failure")
    controller.record_human_decision(HumanDecision.APPROVE, command_id="approve-failure")
    before = controller.session.workflow_runtime

    result = controller.execute_current_plan(command_id="execute-failure")

    assert result.changed is False
    assert result.code == "EXECUTION_SERVICE_FAILURE"
    assert "secret" not in str(result.model_dump())
    assert controller.session.workflow_runtime.state == before.state


def test_destructive_cast_qa_failure_withholds_gold() -> None:
    request = _json_request(
        b'[{"order_id":1,"amount_text":"1"},{"order_id":2,"amount_text":"2"},'
        b'{"order_id":3,"amount_text":"3"},{"order_id":4,"amount_text":"4"},'
        b'{"order_id":5,"amount_text":"bad"}]'
    )
    controller = _loaded_controller(request)
    controller.submit_intent(_intent(), (_cast_request("amount_text"),))
    controller.prepare_plan(command_id="plan-cast-fail")
    _approve_and_execute(controller)

    runtime = controller.session.workflow_runtime
    assert runtime is not None
    assert runtime.state.stage is WorkflowStage.QA_FAILED
    assert runtime.gold_dataframe is None


def test_fabricated_warn_result_is_rejected_and_never_exposes_gold() -> None:
    def warning_execute(runtime, approval):
        completed = execute_workflow(runtime, approval)
        assert completed.state.qa_report is not None
        warning_report = completed.state.qa_report.model_copy(update={"status": QAStatus.WARN})
        warning_state = completed.state.model_copy(
            update={"stage": WorkflowStage.QA_WARNING, "qa_report": warning_report}
        )
        return WorkflowRuntime(
            state=warning_state,
            raw_dataframe=completed.raw_dataframe,
            transformed_dataframe=completed.transformed_dataframe,
            gold_dataframe=None,
            user_intent=completed.user_intent,
            column_alias_map=completed.column_alias_map,
        )

    controller = _loaded_controller(execute_service=warning_execute)
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan-warning")
    controller.record_human_decision(HumanDecision.APPROVE, command_id="approve-warning")
    result = controller.execute_current_plan(command_id="execute-warning")

    runtime = controller.session.workflow_runtime
    assert runtime is not None
    assert result.code == "EXECUTION_EVIDENCE_INVALID"
    assert runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert runtime.gold_dataframe is None


def test_same_upload_skips_parser_but_option_change_resets_downstream_state() -> None:
    calls = {"parse": 0}
    from datachef.application import parse_upload

    def counted(request, policy):
        calls["parse"] += 1
        return parse_upload(request, policy)

    controller = _loaded_controller(upload_parser=counted)
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan-before-rerun")
    same = controller.load_upload(_csv_request())
    assert same.changed is False
    assert calls == {"parse": 1}

    changed = controller.load_upload(_csv_request(encoding="utf-8"))

    assert changed.changed
    assert calls == {"parse": 2}
    assert controller.session.intent is None
    assert controller.session.workflow_runtime is None


def test_parser_result_must_match_the_complete_current_request_metadata() -> None:
    expected_request = _csv_request()
    expected_metadata = source_metadata_for_upload(expected_request, UploadPolicy())
    assert not hasattr(expected_metadata, "safe_message")
    frame = pd.DataFrame({"order_id": [1], "region": ["North"], "amount": [10]})

    mismatches = (
        {"request_id": "upload-" + "0" * 64},
        {"format": UploadFormat.JSON_RECORDS, "parser_options": JsonRecordsParserOptions()},
        {"parser_options": CsvParserOptions(encoding="utf-8")},
        {"byte_size": expected_metadata.byte_size + 1},
    )
    for update in mismatches:
        foreign = ParsedDataset(
            metadata=expected_metadata.model_copy(update=update),
            dataframe=frame,
        )
        controller = DataChefController(upload_parser=lambda request, policy, item=foreign: item)

        result = controller.load_upload(expected_request)

        assert result.code == "UPLOAD_PARSER_RESULT_MISMATCH"
        assert controller.session.source is None


def test_parser_exception_and_non_dataset_result_are_sanitized() -> None:
    calls = {"raising": 0, "invalid": 0}

    def raising_parser(request, policy):
        del request, policy
        calls["raising"] += 1
        raise RuntimeError("password=hunter2 source row secret")

    def invalid_parser(request, policy):
        del request, policy
        calls["invalid"] += 1
        return {"dataframe": "not trusted"}

    first = DataChefController(upload_parser=raising_parser)
    result_one = first.load_upload(_csv_request())
    result_two = first.load_upload(_csv_request())
    second = DataChefController(upload_parser=invalid_parser)
    invalid = second.load_upload(_csv_request())

    assert calls == {"raising": 2, "invalid": 1}
    assert result_one.code == result_two.code == "UPLOAD_PARSER_FAILURE"
    assert invalid.code == "UPLOAD_PARSER_FAILURE"
    assert "hunter2" not in str((result_one.model_dump(), invalid.model_dump()))
    assert first.session.source is None and second.session.source is None


def test_ragged_csv_cannot_install_a_source_or_reach_workflow() -> None:
    controller = DataChefController()

    uploaded = controller.load_upload(_csv_request(b"a,b\n1,2,3\n"))
    diagnosed = controller.diagnose()
    planned = controller.prepare_plan(command_id="ragged-plan")

    assert uploaded.code == "UPLOAD_RAGGED_CSV_RECORD"
    assert diagnosed.code == "SOURCE_REQUIRED"
    assert planned.code == "INTENT_REQUIRED"
    assert controller.session.source is None
    assert controller.session.workflow_runtime is None


def test_foreign_prepared_runtime_is_rejected_and_failed_command_is_idempotent() -> None:
    calls = {"prepare": 0}
    foreign_source = pd.DataFrame(
        {"order_id": [91, 92], "region": ["West", "East"], "amount": [9, 10]}
    )

    def foreign_prepare(source, intent, planner, reviewer):
        del source
        calls["prepare"] += 1
        return prepare_workflow(foreign_source, intent, planner, reviewer)

    controller = _loaded_controller(prepare_service=foreign_prepare)
    controller.submit_intent(_intent(), ())
    before_source = controller.session.source.raw_copy()

    first = controller.prepare_plan(command_id="foreign-plan")
    repeated = controller.prepare_plan(command_id="foreign-plan")
    retried = controller.prepare_plan(command_id="foreign-plan-retry")

    assert first.code == "PLANNING_RUNTIME_INVALID"
    assert repeated.code == "PLAN_COMMAND_REPLAYED"
    assert retried.code == "PLANNING_RUNTIME_INVALID"
    assert calls == {"prepare": 2}
    assert controller.session.workflow_runtime is None
    assert_frame_equal(controller.session.source.raw_copy(), before_source)


def test_non_runtime_prepare_result_is_rejected() -> None:
    controller = _loaded_controller(
        prepare_service=lambda source, intent, planner, reviewer: object()
    )
    controller.submit_intent(_intent(), ())

    result = controller.prepare_plan(command_id="non-runtime-plan")

    assert result.code == "PLANNING_RUNTIME_INVALID"
    assert controller.session.workflow_runtime is None


def test_prepare_exception_is_attempt_idempotent_and_retry_needs_new_id() -> None:
    calls = {"prepare": 0}

    def failing_prepare(source, intent, planner, reviewer):
        del source, intent, planner, reviewer
        calls["prepare"] += 1
        raise RuntimeError("sensitive planning detail")

    controller = _loaded_controller(prepare_service=failing_prepare)
    controller.submit_intent(_intent(), ())

    first = controller.prepare_plan(command_id="plan-fails")
    repeated = controller.prepare_plan(command_id="plan-fails")
    retried = controller.prepare_plan(command_id="plan-retry")

    assert first.code == retried.code == "PLANNING_SERVICE_FAILURE"
    assert repeated.code == "PLAN_COMMAND_REPLAYED"
    assert calls == {"prepare": 2}


def test_material_reset_invalidates_failed_command_binding() -> None:
    calls = {"prepare": 0}

    def failing_prepare(source, intent, planner, reviewer):
        del source, intent, planner, reviewer
        calls["prepare"] += 1
        raise RuntimeError("safe test failure")

    controller = _loaded_controller(prepare_service=failing_prepare)
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="reusable-after-reset")
    controller.reset()
    controller.load_upload(_csv_request())
    controller.diagnose()
    controller.submit_intent(_intent(intent_id="new-session-intent"), ())

    result = controller.prepare_plan(command_id="reusable-after-reset")

    assert result.code == "PLANNING_SERVICE_FAILURE"
    assert calls == {"prepare": 2}


def _completed_controller(content: bytes) -> DataChefController:
    controller = _loaded_controller(_csv_request(content))
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="completed-plan")
    _approve_and_execute(controller)
    return controller


def test_foreign_qa_passed_runtime_is_rejected_without_exposing_gold() -> None:
    foreign = _completed_controller(
        b"order_id,region,amount\n91,West,900\n92,East,800\n"
    ).session.workflow_runtime
    assert foreign is not None and foreign.gold_dataframe is not None
    calls = {"execute": 0}

    def foreign_execute(runtime, approval):
        del runtime, approval
        calls["execute"] += 1
        return foreign

    controller = _loaded_controller(execute_service=foreign_execute)
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="victim-plan")
    controller.record_human_decision(HumanDecision.APPROVE, command_id="victim-approve")
    before = controller.session

    first = controller.execute_current_plan(command_id="foreign-execution")
    repeated = controller.execute_current_plan(command_id="foreign-execution")
    after = controller.session

    assert first.code == "EXECUTION_EVIDENCE_INVALID"
    assert repeated.code == "EXECUTION_COMMAND_REPLAYED"
    assert calls == {"execute": 1}
    assert after.workflow_runtime.state == before.workflow_runtime.state
    assert after.workflow_runtime.gold_dataframe is None
    assert_frame_equal(after.source.raw_copy(), before.source.raw_copy())


def test_same_dataset_modified_gold_with_stale_qa_is_rejected() -> None:
    calls = {"execute": 0}

    def stale_qa_execute(runtime, approval):
        calls["execute"] += 1
        completed = execute_workflow(runtime, approval)
        assert completed.transformed_dataframe is not None
        assert completed.gold_dataframe is not None
        assert completed.state.execution_result is not None
        transformed = completed.transformed_dataframe.copy(deep=True)
        transformed.loc[0, "amount"] = 999
        forged_result = completed.state.execution_result.model_copy(
            update={"result_fingerprint": identify_dataset(transformed).fingerprint}
        )
        return WorkflowRuntime(
            state=completed.state.model_copy(
                update={"execution_result": forged_result}
            ),
            raw_dataframe=completed.raw_dataframe.copy(deep=True),
            transformed_dataframe=transformed,
            gold_dataframe=transformed.copy(deep=True),
            user_intent=completed.user_intent,
            column_alias_map=completed.column_alias_map,
        )

    controller = _loaded_controller(execute_service=stale_qa_execute)
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="stale-qa-plan")
    controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="stale-qa-approval",
    )
    before = controller.session

    result = controller.execute_current_plan(command_id="stale-qa-execution")

    assert result.code == "EXECUTION_EVIDENCE_INVALID"
    assert calls == {"execute": 1}
    assert controller.session.workflow_runtime.state == before.workflow_runtime.state
    assert controller.session.workflow_runtime.gold_dataframe is None
    assert_frame_equal(controller.session.source.raw_copy(), before.source.raw_copy())


def test_controller_installs_the_verifiers_copy_not_the_service_candidate() -> None:
    returned: dict[str, WorkflowRuntime] = {}

    def captured_execute(runtime, approval):
        candidate = execute_workflow(runtime, approval)
        returned["candidate"] = candidate
        return candidate

    controller = _loaded_controller(execute_service=captured_execute)
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="verified-copy-plan")
    controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="verified-copy-approval",
    )
    completed = controller.execute_current_plan(command_id="verified-copy-execution")
    candidate = returned["candidate"]
    assert candidate.gold_dataframe is not None
    candidate.gold_dataframe.loc[0, "amount"] = 999

    installed = controller.session.workflow_runtime

    assert completed.code == "EXECUTION_COMPLETED"
    assert installed is not None and installed.gold_dataframe is not None
    assert installed.gold_dataframe.loc[0, "amount"] == 10


def test_execute_exception_is_attempt_idempotent_and_retry_needs_new_id() -> None:
    calls = {"execute": 0}

    def failing_execute(runtime, approval):
        del runtime, approval
        calls["execute"] += 1
        raise RuntimeError("private execution detail")

    controller = _loaded_controller(execute_service=failing_execute)
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan-for-failing-execute")
    controller.record_human_decision(HumanDecision.APPROVE, command_id="approve-failing")

    first = controller.execute_current_plan(command_id="execute-fails")
    repeated = controller.execute_current_plan(command_id="execute-fails")
    retried = controller.execute_current_plan(command_id="execute-retry")

    assert first.code == retried.code == "EXECUTION_SERVICE_FAILURE"
    assert repeated.code == "EXECUTION_COMMAND_REPLAYED"
    assert calls == {"execute": 2}


def test_non_runtime_execution_result_is_rejected_without_losing_prepared_state() -> None:
    controller = _loaded_controller(
        execute_service=lambda runtime, approval: object()
    )
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan-before-invalid-execution")
    controller.record_human_decision(HumanDecision.APPROVE, command_id="approve-invalid")
    before = controller.session.workflow_runtime

    result = controller.execute_current_plan(command_id="invalid-execution-result")

    assert result.code == "EXECUTION_EVIDENCE_INVALID"
    assert controller.session.workflow_runtime.state == before.state
    assert controller.session.workflow_runtime.gold_dataframe is None


def test_public_session_views_defensively_copy_all_runtime_dataframes() -> None:
    controller = _completed_controller(
        b"order_id,region,amount\n1,North,10\n2,South,20\n"
    )
    first = controller.session
    assert first.workflow_runtime is not None
    first.workflow_runtime.raw_dataframe.loc[0, "amount"] = 999
    first.workflow_runtime.transformed_dataframe.loc[0, "amount"] = 998
    first.workflow_runtime.gold_dataframe.loc[0, "amount"] = 997

    second = controller.session
    assert second.workflow_runtime.raw_dataframe.loc[0, "amount"] == 10
    assert second.workflow_runtime.transformed_dataframe.loc[0, "amount"] == 10
    assert second.workflow_runtime.gold_dataframe.loc[0, "amount"] == 10


def test_public_prepared_runtime_raw_frame_is_a_defensive_copy() -> None:
    controller = _loaded_controller()
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="prepared-copy")
    first = controller.session
    first.workflow_runtime.raw_dataframe.loc[0, "amount"] = 999

    second = controller.session

    assert second.workflow_runtime.raw_dataframe.loc[0, "amount"] == 10
    assert second.source.raw_copy().loc[0, "amount"] == 10


def test_public_qa_failure_transformed_frame_is_a_defensive_copy() -> None:
    request = _json_request(
        b'[{"order_id":1,"amount_text":"1"},{"order_id":2,"amount_text":"2"},'
        b'{"order_id":3,"amount_text":"3"},{"order_id":4,"amount_text":"4"},'
        b'{"order_id":5,"amount_text":"bad"}]'
    )
    controller = _loaded_controller(request)
    controller.submit_intent(_intent(), (_cast_request("amount_text"),))
    controller.prepare_plan(command_id="failure-plan-copy")
    _approve_and_execute(controller)
    first = controller.session
    assert first.workflow_runtime.transformed_dataframe is not None
    first.workflow_runtime.transformed_dataframe.loc[0, "amount_text"] = 999

    second = controller.session
    assert second.workflow_runtime.transformed_dataframe.loc[0, "amount_text"] == 1


def test_qa_passed_reset_clears_gold_and_rotates_uploader() -> None:
    controller = _completed_controller(
        b"order_id,region,amount\n1,North,10\n2,South,20\n"
    )
    generation = controller.session.uploader_generation

    controller.reset()

    assert controller.session.source is None
    assert controller.session.workflow_runtime is None
    assert controller.session.uploader_generation == generation + 1


def test_new_dataset_changed_intent_preview_and_reset_follow_invalidation_rules() -> None:
    controller = _loaded_controller()
    assert controller.session.preview_enabled is False
    assert controller.set_preview_enabled(True).changed
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan-before-change")

    controller.revise_intent(_intent(intent_id="changed-intent"), ())
    assert controller.session.source is not None
    assert controller.session.display_diagnostic_report is not None
    assert controller.session.workflow_runtime is None

    controller.load_upload(_csv_request(b"order_id,region,amount\n9,West,90\n"))
    assert controller.session.preview_enabled is False
    assert controller.session.intent is None
    generation = controller.session.uploader_generation
    controller.reset()
    assert controller.session.screen is ScreenId.UPLOAD
    assert controller.session.source is None
    assert controller.session.uploader_generation == generation + 1


def test_controller_import_graph_is_deterministic_and_provider_free() -> None:
    root = Path(__file__).parents[2]
    files = tuple((root / "datachef" / "application").glob("*.py"))
    forbidden_roots = {
        "crewai",
        "google",
        "langchain_google_genai",
        "streamlit",
        "ui",
    }
    forbidden_modules = {
        "datachef.transform.executor",
        "datachef.transform.runner",
        "datachef.qa.service",
        "crew.transformation_agent",
    }
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not imported.intersection(forbidden_modules)
        assert not {name.split(".")[0] for name in imported}.intersection(forbidden_roots)

    assert "crewai" not in sys.modules
    assert "google.genai" not in sys.modules
    assert "completed_runtime_verifier" not in inspect.signature(
        DataChefController
    ).parameters
