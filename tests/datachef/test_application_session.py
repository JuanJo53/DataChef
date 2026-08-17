from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from datachef.application import (
    ApplicationFinding,
    ApplicationSession,
    CommandAttempt,
    CommandKind,
    CommandOutcome,
    CsvParserOptions,
    ParsedDataset,
    RequestedTransformation,
    ScreenId,
    SourceMetadata,
    UploadFormat,
)
from datachef.application.session import (
    accept_source,
    navigate,
    new_session,
    record_approval,
    record_command_attempt,
    record_diagnosis,
    record_intent,
    record_runtime,
    reset_session,
    screen_for_workflow_stage,
    set_preview,
)
from datachef.contracts import (
    CastColumnParameters,
    CastTarget,
    DownstreamUse,
    HumanApproval,
    HumanDecision,
    OperationType,
    UserIntent,
    WorkflowStage,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.planning import RuleBasedPlanner, RuleBasedReviewer
from datachef.workflow import prepare_workflow


def _parsed(values: tuple[int, ...], request_id: str) -> ParsedDataset:
    dataframe = pd.DataFrame({"order_id": values, "amount": range(len(values))})
    return ParsedDataset(
        metadata=SourceMetadata(
            request_id=request_id,
            format=UploadFormat.CSV,
            byte_size=20,
            parser_options=CsvParserOptions(),
        ),
        dataframe=dataframe,
    )


def _attempt(command_id: str, kind: CommandKind) -> CommandAttempt:
    return CommandAttempt(
        command_id=command_id,
        kind=kind,
        binding_id="binding-test",
        outcome=CommandOutcome.SUCCEEDED,
        result_code="TEST_SUCCESS",
    )


def _intent(intent_id: str = "intent-1") -> UserIntent:
    return UserIntent(
        intent_id=intent_id,
        user_goal="Prepare a trustworthy table.",
        downstream_use=DownstreamUse.ANALYSIS,
        selected_key_columns=("order_id",),
        required_columns=("order_id",),
        acceptable_row_loss_pct=50,
    )


def _request(request_id: str = "request-1") -> RequestedTransformation:
    return RequestedTransformation(
        request_id=request_id,
        operation_type=OperationType.CAST_COLUMN,
        target_columns=("amount",),
        parameters=CastColumnParameters(target_type=CastTarget.NUMERIC),
    )


def _populated_session() -> ApplicationSession:
    source = _parsed((1, 2), "upload-1")
    report = diagnose_raw_dataframe(source.raw_copy(), selected_key_columns=("order_id",))
    intent = _intent()
    runtime = prepare_workflow(
        source.raw_copy(),
        intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )
    session = accept_source(new_session(), source)[0]
    session = record_diagnosis(session, report, ())
    session = record_intent(session, intent, (_request(),))
    session = record_runtime(
        session,
        runtime,
        findings=(
            ApplicationFinding(
                code="EXAMPLE",
                blocking=False,
                safe_message="Example planning finding.",
            ),
        ),
        command_attempt=_attempt("plan-command-1", CommandKind.PLAN_PREPARATION),
    )
    plan = runtime.state.transformation_plan
    assert plan is not None
    approval = HumanApproval(
        dataset_id=plan.dataset_id,
        dataset_fingerprint=plan.dataset_fingerprint,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        decision=HumanDecision.APPROVE,
        approved_operation_ids=tuple(op.operation_id for op in plan.operations),
        decided_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )
    return record_approval(
        session,
        approval,
        command_attempt=_attempt(
            "approval-command-1", CommandKind.HUMAN_DECISION
        ),
    )


def test_same_upload_is_an_idempotent_no_change_transition() -> None:
    session = _populated_session()
    assert session.source is not None

    repeated, changed = accept_source(session, session.source)

    assert changed is False
    assert repeated is session


def test_new_source_invalidates_all_dataset_bound_evidence() -> None:
    session = set_preview(_populated_session(), True)
    replacement = _parsed((3, 4), "upload-2")

    changed, did_change = accept_source(session, replacement)

    assert did_change is True
    assert changed.source is replacement
    assert changed.screen is ScreenId.DIAGNOSE
    assert changed.display_diagnostic_report is None
    assert changed.intent is None
    assert changed.requested_transformations == ()
    assert changed.suggested_questions == ()
    assert changed.selected_question_ids == ()
    assert changed.workflow_runtime is None
    assert changed.findings == ()
    assert changed.pending_approval is None
    assert changed.preview_enabled is False
    assert changed.last_plan_command_id is None
    assert changed.last_human_decision_command_id is None
    assert changed.last_execution_command_id is None
    assert changed.command_history == ()


def test_material_intent_change_preserves_source_report_and_questions_only() -> None:
    session = _populated_session()
    source = session.source
    report = session.display_diagnostic_report
    questions = session.suggested_questions

    changed = record_intent(session, _intent("intent-2"), ())

    assert changed.source is source
    assert changed.display_diagnostic_report is report
    assert changed.suggested_questions is questions
    assert changed.intent is not session.intent
    assert changed.requested_transformations == ()
    assert changed.workflow_runtime is None
    assert changed.findings == ()
    assert changed.pending_approval is None
    assert changed.last_plan_command_id is None
    assert changed.last_human_decision_command_id is None
    assert changed.last_execution_command_id is None
    assert changed.command_history == ()


def test_duplicate_requested_transformation_ids_are_rejected() -> None:
    session = _populated_session()

    with pytest.raises(ValueError, match="request IDs must be unique"):
        record_intent(session, _intent("intent-2"), (_request(), _request()))


def test_semantically_duplicate_requests_are_rejected_even_with_distinct_ids() -> None:
    session = _populated_session()

    with pytest.raises(ValueError, match="semantically unique"):
        record_intent(
            session,
            _intent("intent-2"),
            (_request("first"), _request("second")),
        )


@pytest.mark.parametrize(
    ("stage", "screen"),
    (
        (WorkflowStage.INITIAL, ScreenId.UPLOAD),
        (WorkflowStage.DIAGNOSED, ScreenId.INTENT),
        (WorkflowStage.INTENT_CAPTURED, ScreenId.INTENT),
        (WorkflowStage.CONTEXT_READY, ScreenId.PLAN),
        (WorkflowStage.PLANNING, ScreenId.PLAN),
        (WorkflowStage.PLAN_REJECTED, ScreenId.PLAN),
        (WorkflowStage.AWAITING_APPROVAL, ScreenId.APPROVAL),
        (WorkflowStage.EXECUTING, ScreenId.QA),
        (WorkflowStage.EXECUTION_FAILED, ScreenId.QA),
        (WorkflowStage.QA_PASSED, ScreenId.RESULTS),
        (WorkflowStage.QA_WARNING, ScreenId.QA),
        (WorkflowStage.QA_FAILED, ScreenId.QA),
    ),
)
def test_every_phase1a_stage_has_an_explicit_presentation_mapping(
    stage: WorkflowStage,
    screen: ScreenId,
) -> None:
    assert screen_for_workflow_stage(stage) is screen


def test_navigation_and_preview_do_not_invalidate_business_evidence() -> None:
    session = _populated_session()
    navigated = navigate(session, ScreenId.QA)
    previewed = set_preview(navigated, True)

    assert navigated.workflow_runtime is session.workflow_runtime
    assert navigated.source is session.source
    assert previewed.workflow_runtime is session.workflow_runtime
    assert previewed.preview_enabled is True
    assert previewed.command_history == session.command_history


def test_session_source_cannot_be_mutated_through_dataframe_aliases() -> None:
    session = _populated_session()
    assert session.source is not None
    before = session.source.raw_copy()
    alias = session.source.raw_copy()
    alias.loc[0, "amount"] = 999

    assert_frame_equal(session.source.raw_copy(), before)


def test_terminal_runtime_is_preserved_across_navigation_reruns() -> None:
    session = _populated_session()
    runtime = session.workflow_runtime

    rerun = navigate(navigate(session, ScreenId.APPROVAL), ScreenId.APPROVAL)

    assert rerun.workflow_runtime is runtime
    assert rerun.pending_approval is session.pending_approval


def test_reset_returns_empty_session_and_rotates_uploader_generation() -> None:
    session = _populated_session()

    reset = reset_session(session)

    assert reset == new_session(uploader_generation=session.uploader_generation + 1)
    assert reset.source is None
    assert reset.screen is ScreenId.UPLOAD
    assert reset.command_history == ()


def test_command_history_is_immutable_complete_and_rejects_rebinding() -> None:
    first = _attempt("historical-command", CommandKind.HUMAN_DECISION)
    second = _attempt("newer-command", CommandKind.HUMAN_DECISION)
    session = record_command_attempt(new_session(), first)
    session = record_command_attempt(session, second)

    assert session.command_history == (first, second)
    assert isinstance(session.command_history, tuple)
    assert record_command_attempt(session, first) is session
    with pytest.raises(ValueError, match="already bound"):
        record_command_attempt(
            session,
            _attempt("historical-command", CommandKind.EXECUTION),
        )


def test_command_history_rejects_malformed_or_duplicate_evidence() -> None:
    attempt = _attempt("duplicate", CommandKind.PLAN_PREPARATION)

    with pytest.raises(TypeError, match="typed command attempts"):
        ApplicationSession(command_history=("not-an-attempt",))
    with pytest.raises(ValueError, match="history IDs must be unique"):
        ApplicationSession(command_history=(attempt, attempt))
    with pytest.raises(ValueError, match="must exist in command history"):
        ApplicationSession(plan_command_attempt=attempt)


def test_new_plan_runtime_clears_only_downstream_command_history() -> None:
    session = _populated_session()
    assert session.workflow_runtime is not None
    replacement_attempt = _attempt("plan-command-2", CommandKind.PLAN_PREPARATION)

    changed = record_runtime(
        session,
        session.workflow_runtime,
        findings=(),
        command_attempt=replacement_attempt,
    )

    assert tuple(item.kind for item in changed.command_history) == (
        CommandKind.PLAN_PREPARATION,
        CommandKind.PLAN_PREPARATION,
    )
    assert tuple(item.command_id for item in changed.command_history) == (
        "plan-command-1",
        "plan-command-2",
    )
    assert changed.human_decision_command_attempt is None
    assert changed.execution_command_attempt is None


def test_session_repr_summarizes_history_without_command_identifiers() -> None:
    session = record_command_attempt(
        new_session(),
        _attempt("sensitive-command-id", CommandKind.PLAN_PREPARATION),
    )

    rendered = repr(session)

    assert "command_attempt_count=1" in rendered
    assert "sensitive-command-id" not in rendered
    assert "binding-test" not in rendered
