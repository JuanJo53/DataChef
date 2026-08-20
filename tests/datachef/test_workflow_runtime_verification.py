from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import datachef.workflow.service as workflow_service
from datachef.contracts import (
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    HumanApproval,
    HumanDecision,
    InvariantKind,
    OperationType,
    QAStatus,
    QualityInvariant,
    RiskLevel,
    TransformationOperation,
    UserIntent,
    WorkflowStage,
)
from datachef.diagnostics import diagnose_raw_dataframe, identify_dataset
from datachef.planning import (
    RuleBasedPlanner,
    RuleBasedReviewer,
    SequencePlanner,
    create_transformation_plan,
)
from datachef.privacy import build_planning_context
from datachef.workflow import (
    WorkflowRuntime,
    execute_workflow,
    prepare_workflow,
    verify_completed_workflow_runtime,
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
        decided_at=datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc),
    )


def test_completed_runtime_verifier_returns_an_independent_verified_copy(
    raw_dataframe,
    user_intent,
) -> None:
    prepared = prepare_workflow(
        raw_dataframe,
        user_intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )
    completed = execute_workflow(prepared, _approval(prepared))

    verified = verify_completed_workflow_runtime(prepared, completed)

    assert verified is not None
    assert verified is not completed
    assert verified.state == completed.state
    assert verified.state.stage is WorkflowStage.QA_PASSED
    assert verified.transformed_dataframe is not completed.transformed_dataframe
    assert verified.gold_dataframe is not completed.gold_dataframe
    assert_frame_equal(verified.transformed_dataframe, completed.transformed_dataframe)
    assert_frame_equal(verified.gold_dataframe, completed.gold_dataframe)
    completed.transformed_dataframe.iloc[0, 0] = 999_999
    completed.gold_dataframe.iloc[0, 0] = 999_999
    assert verified.transformed_dataframe.iloc[0, 0] != 999_999
    assert verified.gold_dataframe.iloc[0, 0] != 999_999


def test_completed_runtime_verifier_rejects_stale_qa_for_modified_values() -> None:
    source = pd.DataFrame({"amount": [10, 20]})
    user_intent = UserIntent(intent_id="runtime-verifier-stale-qa")
    prepared = prepare_workflow(
        source,
        user_intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )
    completed = execute_workflow(prepared, _approval(prepared))
    assert completed.transformed_dataframe is not None
    assert completed.gold_dataframe is not None
    assert completed.state.execution_result is not None
    assert completed.state.qa_report is not None

    transformed = completed.transformed_dataframe.copy(deep=True)
    transformed.loc[0, "amount"] = 999
    forged_result = completed.state.execution_result.model_copy(
        update={"result_fingerprint": identify_dataset(transformed).fingerprint}
    )
    candidate = WorkflowRuntime(
        state=completed.state.model_copy(
            update={"execution_result": forged_result}
        ),
        raw_dataframe=completed.raw_dataframe.copy(deep=True),
        transformed_dataframe=transformed,
        gold_dataframe=transformed.copy(deep=True),
        user_intent=completed.user_intent,
        column_alias_map=completed.column_alias_map,
    )

    assert verify_completed_workflow_runtime(prepared, candidate) is None


def _empty_plan_pair() -> tuple[WorkflowRuntime, WorkflowRuntime]:
    source = pd.DataFrame({"category": ["a", "b"], "amount": [1, 2]})
    intent = UserIntent(
        intent_id="empty-plan-verifier",
        user_goal="Use the already clean table.",
    )
    prepared = prepare_workflow(
        source,
        intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )
    completed = execute_workflow(prepared, _approval(prepared))
    assert completed.state.stage is WorkflowStage.QA_PASSED
    return prepared, completed


@pytest.mark.parametrize(
    "candidate_factory",
    (
        pytest.param(lambda completed: object(), id="non-runtime"),
        pytest.param(
            lambda completed: WorkflowRuntime(
                state=completed.state.model_copy(
                    update={"stage": WorkflowStage.AWAITING_APPROVAL}
                ),
                raw_dataframe=completed.raw_dataframe.copy(deep=True),
                transformed_dataframe=completed.transformed_dataframe.copy(deep=True),
                gold_dataframe=completed.gold_dataframe.copy(deep=True),
                user_intent=completed.user_intent,
                column_alias_map=completed.column_alias_map,
            ),
            id="wrong-stage",
        ),
        pytest.param(
            lambda completed: WorkflowRuntime(
                state=completed.state.model_copy(
                    update={
                        "qa_report": completed.state.qa_report.model_copy(
                            update={"qa_report_id": "foreign-qa-report"}
                        )
                    }
                ),
                raw_dataframe=completed.raw_dataframe.copy(deep=True),
                transformed_dataframe=completed.transformed_dataframe.copy(deep=True),
                gold_dataframe=completed.gold_dataframe.copy(deep=True),
                user_intent=completed.user_intent,
                column_alias_map=completed.column_alias_map,
            ),
            id="foreign-qa-report",
        ),
        pytest.param(
            lambda completed: WorkflowRuntime(
                state=completed.state.model_copy(
                    update={
                        "stage": WorkflowStage.QA_WARNING,
                        "qa_report": completed.state.qa_report.model_copy(
                            update={"status": QAStatus.WARN}
                        ),
                    }
                ),
                raw_dataframe=completed.raw_dataframe.copy(deep=True),
                transformed_dataframe=completed.transformed_dataframe.copy(deep=True),
                gold_dataframe=None,
                user_intent=completed.user_intent,
                column_alias_map=completed.column_alias_map,
            ),
            id="fabricated-warn",
        ),
        pytest.param(
            lambda completed: WorkflowRuntime(
                state=completed.state.model_copy(
                    update={
                        "stage": WorkflowStage.QA_FAILED,
                        "qa_report": completed.state.qa_report.model_copy(
                            update={"status": QAStatus.FAIL}
                        ),
                    }
                ),
                raw_dataframe=completed.raw_dataframe.copy(deep=True),
                transformed_dataframe=completed.transformed_dataframe.copy(deep=True),
                gold_dataframe=None,
                user_intent=completed.user_intent,
                column_alias_map=completed.column_alias_map,
            ),
            id="fabricated-fail",
        ),
    ),
)
def test_completed_runtime_verifier_rejects_fabricated_candidate_matrix(
    candidate_factory,
) -> None:
    prepared, completed = _empty_plan_pair()

    candidate = candidate_factory(completed)

    assert verify_completed_workflow_runtime(prepared, candidate) is None


def test_completed_runtime_verifier_rejects_wrong_authoritative_stage() -> None:
    _, completed = _empty_plan_pair()

    assert verify_completed_workflow_runtime(completed, completed) is None


def test_completed_runtime_verifier_rejects_stale_operation_evidence(
    raw_dataframe,
    user_intent,
) -> None:
    prepared = prepare_workflow(
        raw_dataframe,
        user_intent,
        RuleBasedPlanner(),
        RuleBasedReviewer(),
    )
    completed = execute_workflow(prepared, _approval(prepared))
    result = completed.state.execution_result
    assert result is not None and result.operation_records
    first = result.operation_records[0].model_copy(
        update={"affected_cell_count": result.operation_records[0].affected_cell_count + 1}
    )
    forged_result = result.model_copy(
        update={"operation_records": (first, *result.operation_records[1:])}
    )
    candidate = WorkflowRuntime(
        state=completed.state.model_copy(update={"execution_result": forged_result}),
        raw_dataframe=completed.raw_dataframe.copy(deep=True),
        transformed_dataframe=completed.transformed_dataframe.copy(deep=True),
        gold_dataframe=(
            completed.gold_dataframe.copy(deep=True)
            if completed.gold_dataframe is not None
            else None
        ),
        user_intent=completed.user_intent,
        column_alias_map=completed.column_alias_map,
    )

    assert verify_completed_workflow_runtime(prepared, candidate) is None


@pytest.mark.parametrize("failure_boundary", ("replay", "qa"))
def test_completed_runtime_verifier_fails_closed_on_recomputation_exception(
    monkeypatch,
    failure_boundary: str,
) -> None:
    prepared, completed = _empty_plan_pair()

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("private recomputation detail")

    target = "execute_workflow" if failure_boundary == "replay" else "run_quality_assurance"
    monkeypatch.setattr(workflow_service, target, fail)

    assert verify_completed_workflow_runtime(prepared, completed) is None


@pytest.mark.parametrize(
    ("mandatory", "expected_stage"),
    (
        (False, WorkflowStage.QA_WARNING),
        (True, WorkflowStage.QA_FAILED),
    ),
)
def test_completed_runtime_verifier_accepts_genuine_qa_outcomes(
    mandatory: bool,
    expected_stage: WorkflowStage,
) -> None:
    prepared, _ = _empty_plan_pair()
    invariant = QualityInvariant(
        invariant_id=f"missing-column-{mandatory}",
        kind=InvariantKind.REQUIRED_COLUMN,
        column="not_present",
        mandatory=mandatory,
    )
    completed = execute_workflow(
        prepared,
        _approval(prepared),
        user_invariants=(invariant,),
    )

    verified = verify_completed_workflow_runtime(
        prepared,
        completed,
        user_invariants=(invariant,),
    )

    assert verified is not None
    assert verified.state.stage is expected_stage
    assert verified.gold_dataframe is None


def test_completed_runtime_verifier_accepts_genuine_execution_failure() -> None:
    source = pd.DataFrame({"value": ["1", "bad"]})
    intent = UserIntent(intent_id="verifier-execution-failure")
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
        rationale="Abort on invalid input.",
        expected_effect="Do not coerce invalid input.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(operation,),
        summary="Strict cast.",
    )
    prepared = prepare_workflow(
        source,
        intent,
        SequencePlanner((plan,)),
        RuleBasedReviewer(),
    )
    completed = execute_workflow(prepared, _approval(prepared))

    verified = verify_completed_workflow_runtime(prepared, completed)

    assert completed.state.stage is WorkflowStage.EXECUTION_FAILED
    assert verified is not None
    assert verified.state.stage is WorkflowStage.EXECUTION_FAILED
    assert verified.transformed_dataframe is None
    assert verified.gold_dataframe is None


def test_empty_plan_verification_is_repeatable_and_preserves_both_inputs() -> None:
    prepared, completed = _empty_plan_pair()
    prepared_before = prepared.raw_dataframe.copy(deep=True)
    completed_before = completed.gold_dataframe.copy(deep=True)

    first = verify_completed_workflow_runtime(prepared, completed)
    second = verify_completed_workflow_runtime(prepared, completed)

    assert first is not None and second is not None
    assert first.state == second.state
    assert first.gold_dataframe is not completed.gold_dataframe
    assert_frame_equal(first.gold_dataframe, prepared.raw_dataframe)
    assert_frame_equal(second.gold_dataframe, first.gold_dataframe)
    assert_frame_equal(prepared.raw_dataframe, prepared_before)
    assert_frame_equal(completed.gold_dataframe, completed_before)
    completed.gold_dataframe.iloc[0, 0] = "mutated-after-verification"
    assert first.gold_dataframe.iloc[0, 0] == "a"
