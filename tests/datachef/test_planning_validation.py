from __future__ import annotations

import pytest
from pydantic import ValidationError

from datachef.contracts import (
    OperationType,
    PIIHandling,
    RenameColumnParameters,
    ReviewerDecision,
    RiskLevel,
    TransformationOperation,
    TrimWhitespaceParameters,
    WorkflowStage,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.intent import discover_questions
from datachef.planning import (
    SequencePlanner,
    SequenceReviewer,
    create_transformation_plan,
    validate_plan,
)
from datachef.privacy import build_planning_context
from datachef.workflow import prepare_workflow


def _operation(
    operation_id: str = "op-trim-category",
    *,
    target: str = "category",
) -> TransformationOperation:
    return TransformationOperation(
        operation_id=operation_id,
        operation_type=OperationType.TRIM_WHITESPACE,
        target_columns=(target,),
        parameters=TrimWhitespaceParameters(),
        user_requirement_ids=("intent.explicit.0",),
        rationale="User requested normalized surrounding whitespace.",
        expected_effect="Remove surrounding whitespace from configured text columns.",
        risk=RiskLevel.LOW,
        requires_human_approval=False,
    )


def _context(raw_dataframe, user_intent):
    report = diagnose_raw_dataframe(
        raw_dataframe,
        selected_key_columns=user_intent.selected_key_columns,
    )
    return build_planning_context(report, user_intent, discover_questions(report))


def _plan(context, operations, *, version: int = 1, summary: str = "Test plan"):
    return create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=version,
        operations=tuple(operations),
        summary=summary,
    )


def test_valid_plan_passes_validation(raw_dataframe, user_intent) -> None:
    context = _context(raw_dataframe, user_intent)
    plan = _plan(context, (_operation(),))

    result = validate_plan(context, plan)

    assert result.valid is True
    assert result.findings == ()


def test_unknown_operation_fails_structural_validation() -> None:
    payload = _operation().model_dump(mode="json")
    payload["operation_type"] = "EXECUTE_PYTHON"

    with pytest.raises(ValidationError):
        TransformationOperation.model_validate(payload)


def test_missing_column_returns_plan_finding(raw_dataframe, user_intent) -> None:
    context = _context(raw_dataframe, user_intent)
    plan = _plan(context, (_operation(target="not_a_column"),))

    result = validate_plan(context, plan)

    assert result.valid is False
    assert "MISSING_COLUMN" in {finding.code for finding in result.findings}


def test_rename_collision_returns_plan_finding(raw_dataframe, user_intent) -> None:
    context = _context(raw_dataframe, user_intent)
    operation = TransformationOperation(
        operation_id="op-rename",
        operation_type=OperationType.RENAME_COLUMN,
        target_columns=("category",),
        parameters=RenameColumnParameters(new_name="amount_text"),
        user_requirement_ids=("intent.explicit.0",),
        rationale="User requested a clearer column name.",
        expected_effect="Rename one column.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )

    result = validate_plan(context, _plan(context, (operation,)))

    assert result.valid is False
    assert "RENAME_COLLISION" in {finding.code for finding in result.findings}


def test_unsupported_pii_request_is_reported(raw_dataframe, user_intent) -> None:
    pii_intent = user_intent.model_copy(update={"pii_handling": PIIHandling.MASK})
    context = _context(raw_dataframe, pii_intent)
    plan = _plan(context, (_operation(),))

    result = validate_plan(context, plan)

    assert result.valid is False
    assert "UNSUPPORTED_PII_OPERATION" in {
        finding.code for finding in result.findings
    }


def test_reviewer_stops_on_first_acceptance(raw_dataframe, user_intent) -> None:
    context = _context(raw_dataframe, user_intent)
    plan = _plan(context, (_operation(),))
    planner = SequencePlanner((plan,))
    reviewer = SequenceReviewer((ReviewerDecision.ACCEPT, ReviewerDecision.REVISE))

    runtime = prepare_workflow(raw_dataframe, user_intent, planner, reviewer)

    assert runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert planner.calls == 1
    assert reviewer.calls == 1


def test_reviewer_stops_after_maximum_attempts(raw_dataframe, user_intent) -> None:
    context = _context(raw_dataframe, user_intent)
    plans = tuple(
        _plan(context, (_operation(),), version=version, summary=f"Plan {version}")
        for version in (1, 2, 3)
    )
    planner = SequencePlanner(plans)
    reviewer = SequenceReviewer((ReviewerDecision.REVISE,) * 3)

    runtime = prepare_workflow(raw_dataframe, user_intent, planner, reviewer)

    assert runtime.state.stage is WorkflowStage.PLAN_REJECTED
    assert runtime.state.planning_attempts == 3
    assert planner.calls == 3
    assert reviewer.calls == 3


def test_invalid_plans_bypass_reviewer(raw_dataframe, user_intent) -> None:
    context = _context(raw_dataframe, user_intent)
    invalid = _plan(context, (_operation(target="missing"),))
    planner = SequencePlanner((invalid, invalid, invalid))
    reviewer = SequenceReviewer((ReviewerDecision.ACCEPT,))

    runtime = prepare_workflow(raw_dataframe, user_intent, planner, reviewer)

    assert runtime.state.stage is WorkflowStage.PLAN_REJECTED
    assert planner.calls == 3
    assert reviewer.calls == 0
