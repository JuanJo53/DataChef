from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

import datachef.transform as transform_api
import datachef.workflow.service as workflow_service
from datachef.contracts import (
    HumanApproval,
    HumanDecision,
    ReviewerDecision,
    ReviewerVerdict,
    RiskLevel,
    TransformationOperation,
    TrimWhitespaceParameters,
    OperationType,
    UserIntent,
    WorkflowState,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.planning import (
    SequencePlanner,
    SequenceReviewer,
    create_transformation_plan,
)
from datachef.privacy import build_planning_context
from datachef.workflow import WorkflowRuntime, execute_workflow, prepare_workflow


def _plan(source: pd.DataFrame, *, version: int = 1):
    intent = UserIntent(intent_id="r3-review-history")
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    operation = TransformationOperation(
        operation_id="trim-label",
        operation_type=OperationType.TRIM_WHITESPACE,
        target_columns=("label",),
        parameters=TrimWhitespaceParameters(),
        user_requirement_ids=("trim",),
        rationale="Remove surrounding whitespace.",
        expected_effect="The label is trimmed.",
        risk=RiskLevel.LOW,
        requires_human_approval=False,
    )
    plan = create_transformation_plan(
        dataset_id=report.dataset_identity.dataset_id,
        dataset_fingerprint=report.dataset_identity.fingerprint,
        version=version,
        operations=(operation,),
        summary="Review-history binding test plan.",
    )
    return intent, plan


def _prepared(
    decisions: tuple[ReviewerDecision, ...] = (ReviewerDecision.ACCEPT,),
    *,
    version: int = 1,
) -> WorkflowRuntime:
    source = pd.DataFrame({"label": [" A "]})
    intent, plan = _plan(source, version=version)
    return prepare_workflow(
        source,
        intent,
        SequencePlanner(tuple(plan for _ in decisions)),
        SequenceReviewer(decisions),
    )


def _approval(runtime: WorkflowRuntime) -> HumanApproval:
    plan = runtime.state.transformation_plan
    assert plan is not None
    return HumanApproval(
        dataset_id=plan.dataset_id,
        dataset_fingerprint=plan.dataset_fingerprint,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        decision=HumanDecision.APPROVE,
        approved_operation_ids=tuple(
            operation.operation_id for operation in plan.operations
        ),
        decided_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )


def _state_payload(runtime: WorkflowRuntime) -> dict[str, object]:
    return runtime.state.model_dump()


@pytest.mark.parametrize(
    "history_kind",
    (
        "empty",
        "revise_only",
        "reject_only",
        "foreign_plan",
        "stale_attempt",
        "future_attempt",
        "accept_then_reject",
    ),
)
def test_awaiting_state_rejects_nonmatching_final_history(history_kind: str) -> None:
    runtime = _prepared()
    plan = runtime.state.transformation_plan
    assert plan is not None
    accept = ReviewerVerdict(
        plan_id=plan.plan_id,
        attempt=1,
        decision=ReviewerDecision.ACCEPT,
    )
    variants = {
        "empty": (),
        "revise_only": (
            accept.model_copy(update={"decision": ReviewerDecision.REVISE}),
        ),
        "reject_only": (
            accept.model_copy(update={"decision": ReviewerDecision.REJECT}),
        ),
        "foreign_plan": (accept.model_copy(update={"plan_id": "foreign-plan"}),),
        "stale_attempt": (accept.model_copy(update={"attempt": 1}),),
        "future_attempt": (accept.model_copy(update={"attempt": 2}),),
        "accept_then_reject": (
            accept,
            accept.model_copy(update={"decision": ReviewerDecision.REJECT}),
        ),
    }
    payload = _state_payload(runtime)
    if history_kind == "stale_attempt":
        payload["planning_attempts"] = 2
        receipt = payload["accepted_review"]
        assert isinstance(receipt, dict)
        receipt["attempt"] = 2
    payload["review_history"] = variants[history_kind]

    with pytest.raises(ValidationError):
        WorkflowState.model_validate(payload)


@pytest.mark.parametrize("receipt_kind", ("foreign_plan", "stale_attempt"))
def test_receipt_must_match_actual_accept_verdict(receipt_kind: str) -> None:
    runtime = _prepared()
    payload = _state_payload(runtime)
    receipt = payload["accepted_review"]
    assert isinstance(receipt, dict)
    if receipt_kind == "foreign_plan":
        receipt["plan_id"] = "foreign-plan"
        receipt["validation_plan_id"] = "foreign-plan"
    else:
        receipt["attempt"] = 2

    with pytest.raises(ValidationError):
        WorkflowState.model_validate(payload)


@pytest.mark.parametrize(
    ("decisions", "expected_attempt"),
    (
        ((ReviewerDecision.ACCEPT,), 1),
        ((ReviewerDecision.REVISE, ReviewerDecision.ACCEPT), 2),
        (
            (
                ReviewerDecision.REVISE,
                ReviewerDecision.REVISE,
                ReviewerDecision.ACCEPT,
            ),
            3,
        ),
    ),
)
def test_valid_acceptance_histories_round_trip_and_execute(
    decisions: tuple[ReviewerDecision, ...],
    expected_attempt: int,
) -> None:
    runtime = _prepared(decisions, version=2)

    reconstructed = WorkflowState.model_validate_json(runtime.state.model_dump_json())
    completed = execute_workflow(runtime, _approval(runtime))

    assert reconstructed.planning_attempts == expected_attempt
    assert reconstructed.transformation_plan is not None
    assert reconstructed.transformation_plan.version == 2
    assert completed.state.stage.value == "QA_PASSED"
    assert completed.gold_dataframe is not None


def test_rejected_and_exhausted_histories_remain_valid() -> None:
    rejected = _prepared((ReviewerDecision.REJECT,))
    exhausted = _prepared((ReviewerDecision.REVISE,) * 3)

    assert WorkflowState.model_validate_json(
        rejected.state.model_dump_json()
    ).stage.value == "PLAN_REJECTED"
    assert WorkflowState.model_validate_json(
        exhausted.state.model_dump_json()
    ).stage.value == "PLAN_REJECTED"


def test_runtime_refuses_unvalidated_empty_history_before_executor_or_qa(
    monkeypatch,
) -> None:
    runtime = _prepared()
    source_before = runtime.raw_dataframe.copy(deep=True)
    bypass_state = runtime.state.model_copy(update={"review_history": ()})
    bypass_runtime = WorkflowRuntime(
        state=bypass_state,
        raw_dataframe=runtime.raw_dataframe,
        transformed_dataframe=runtime.transformed_dataframe,
        gold_dataframe=runtime.gold_dataframe,
        user_intent=runtime.user_intent,
        column_alias_map=runtime.column_alias_map,
    )
    calls = {"executor": 0, "qa": 0}

    def forbidden_executor(*args, **kwargs):
        calls["executor"] += 1
        raise AssertionError("executor must not run")

    def forbidden_qa(*args, **kwargs):
        calls["qa"] += 1
        raise AssertionError("QA must not run")

    monkeypatch.setattr(workflow_service, "execute_approved_plan", forbidden_executor)
    monkeypatch.setattr(workflow_service, "run_quality_assurance", forbidden_qa)

    refused = execute_workflow(bypass_runtime, _approval(runtime))

    assert refused is bypass_runtime
    assert calls == {"executor": 0, "qa": 0}
    assert refused.gold_dataframe is None
    assert_frame_equal(runtime.raw_dataframe, source_before)


def test_empty_review_history_cannot_reconstruct_from_json() -> None:
    runtime = _prepared()
    bypass_state = runtime.state.model_copy(update={"review_history": ()})

    with pytest.raises(ValidationError):
        WorkflowState.model_validate_json(bypass_state.model_dump_json())


def test_lower_level_executor_is_not_application_facing() -> None:
    assert "execute_approved_plan" not in transform_api.__all__
    assert not hasattr(transform_api, "execute_approved_plan")
