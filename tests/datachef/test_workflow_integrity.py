from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from datachef.contracts import (
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    HumanApproval,
    HumanDecision,
    InvariantKind,
    OperationType,
    QualityInvariant,
    ReviewerDecision,
    RiskLevel,
    TransformationOperation,
    UserIntent,
    WorkflowStage,
    WorkflowState,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.planning import (
    RuleBasedPlanner,
    RuleBasedReviewer,
    SequencePlanner,
    SequenceReviewer,
    create_transformation_plan,
)
from datachef.privacy import build_planning_context
from datachef.workflow import execute_workflow, prepare_workflow
from support import human_approval


def _accepted_runtime(source, intent):
    return prepare_workflow(source, intent, RuleBasedPlanner(), RuleBasedReviewer())


def test_awaiting_resume_without_or_with_stale_approval_is_idempotent(
    raw_dataframe,
    user_intent,
) -> None:
    runtime = _accepted_runtime(raw_dataframe, user_intent)
    plan = runtime.state.transformation_plan
    assert plan is not None
    stale = human_approval(plan).model_copy(update={"plan_id": "stale-plan"})

    assert execute_workflow(runtime, None) is runtime
    assert execute_workflow(runtime, stale) is runtime


def test_all_exercised_terminal_routes_are_idempotent(raw_dataframe, user_intent) -> None:
    accepted = _accepted_runtime(raw_dataframe, user_intent)
    plan = accepted.state.transformation_plan
    assert plan is not None
    passed = execute_workflow(accepted, human_approval(plan))

    mandatory_failure = QualityInvariant(
        invariant_id="missing-mandatory",
        kind=InvariantKind.REQUIRED_COLUMN,
        column="not-present",
        mandatory=True,
    )
    failed = execute_workflow(
        accepted,
        human_approval(plan),
        user_invariants=(mandatory_failure,),
    )
    warning = QualityInvariant(
        invariant_id="missing-warning",
        kind=InvariantKind.REQUIRED_COLUMN,
        column="not-present",
        mandatory=False,
    )
    warned = execute_workflow(
        accepted,
        human_approval(plan),
        user_invariants=(warning,),
    )
    rejected = prepare_workflow(
        raw_dataframe,
        user_intent,
        RuleBasedPlanner(),
        SequenceReviewer((ReviewerDecision.REJECT,)),
    )
    exhausted = prepare_workflow(
        raw_dataframe,
        user_intent,
        RuleBasedPlanner(),
        SequenceReviewer((ReviewerDecision.REVISE,) * 3),
    )

    for terminal in (passed, failed, warned, rejected, exhausted):
        resumed = execute_workflow(terminal, None)
        assert resumed is terminal
        assert resumed.raw_dataframe is terminal.raw_dataframe
        assert resumed.transformed_dataframe is terminal.transformed_dataframe
        assert resumed.gold_dataframe is terminal.gold_dataframe


def test_execution_failure_route_is_idempotent() -> None:
    source = pd.DataFrame({"value": ["1", "bad"]})
    intent = UserIntent(intent_id="execution-failure")
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    operation = TransformationOperation(
        operation_id="strict-cast",
        operation_type=OperationType.CAST_COLUMN,
        target_columns=("value",),
        parameters=CastColumnParameters(
            target_type=CastTarget.NUMERIC,
            errors=CastErrorPolicy.RAISE,
        ),
        user_requirement_ids=("strict",),
        rationale="Reject invalid numeric input.",
        expected_effect="Abort rather than coerce invalid input.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(operation,),
        summary="Strict failure plan.",
    )
    runtime = prepare_workflow(
        source,
        intent,
        SequencePlanner((plan,)),
        RuleBasedReviewer(),
    )
    failed = execute_workflow(runtime, human_approval(plan))

    assert failed.state.stage is WorkflowStage.EXECUTION_FAILED
    assert execute_workflow(failed, None) is failed


def test_valid_state_round_trip_and_impossible_states_are_rejected(
    raw_dataframe,
    user_intent,
) -> None:
    awaiting = _accepted_runtime(raw_dataframe, user_intent)
    restored = WorkflowState.model_validate_json(awaiting.state.model_dump_json())
    assert restored == awaiting.state

    awaiting_payload = awaiting.state.model_dump(mode="json")
    awaiting_payload["accepted_review"] = None
    with pytest.raises(ValidationError):
        WorkflowState.model_validate(awaiting_payload)

    rejected_payload = awaiting.state.model_dump(mode="json")
    rejected_payload["stage"] = WorkflowStage.PLAN_REJECTED.value
    with pytest.raises(ValidationError):
        WorkflowState.model_validate(rejected_payload)

    with pytest.raises(ValidationError):
        WorkflowState(stage=WorkflowStage.QA_PASSED)


def test_qa_stage_must_match_serialized_qa_evidence(raw_dataframe, user_intent) -> None:
    runtime = _accepted_runtime(raw_dataframe, user_intent)
    plan = runtime.state.transformation_plan
    assert plan is not None
    completed = execute_workflow(runtime, human_approval(plan))
    payload = completed.state.model_dump(mode="json")
    payload["stage"] = WorkflowStage.QA_FAILED.value

    with pytest.raises(ValidationError):
        WorkflowState.model_validate(payload)
