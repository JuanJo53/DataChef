from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from datachef.contracts import (
    DeduplicateByKeysParameters,
    DiagnosticIssueKind,
    ExecutionResult,
    KeepPolicy,
    OperationType,
    OperationExecutionRecord,
    OperationExecutionStatus,
    QAStatus,
    ReviewerDecision,
    ReviewerVerdict,
    RiskLevel,
    TransformationOperation,
    TrimWhitespaceParameters,
    UserIntent,
)
from datachef.diagnostics import diagnose_raw_dataframe, identify_dataset
from datachef.planning import (
    ReviewEvidenceError,
    accept_review,
    create_transformation_plan,
    validate_plan,
    SequencePlanner,
)
from datachef.privacy import build_planning_context
from datachef.qa import run_quality_assurance
from datachef.transform.executor import ApprovalGateError, execute_approved_plan
from datachef.workflow import prepare_workflow
from support import accepted_review, human_approval


def _dedup(operation_id: str, keys: tuple[str, ...]) -> TransformationOperation:
    return TransformationOperation(
        operation_id=operation_id,
        operation_type=OperationType.DEDUPLICATE_BY_KEYS,
        target_columns=keys,
        parameters=DeduplicateByKeysParameters(keys=keys, keep=KeepPolicy.FIRST),
        user_requirement_ids=("key-uniqueness",),
        rationale="Require unique non-null keys.",
        expected_effect="Keep the first row for each non-null key.",
        risk=RiskLevel.HIGH,
        requires_human_approval=True,
    )


def _trim() -> TransformationOperation:
    return TransformationOperation(
        operation_id="trim-label",
        operation_type=OperationType.TRIM_WHITESPACE,
        target_columns=("label",),
        parameters=TrimWhitespaceParameters(),
        user_requirement_ids=("trim",),
        rationale="Remove surrounding whitespace.",
        expected_effect="Normalize the configured label.",
        risk=RiskLevel.LOW,
        requires_human_approval=False,
    )


def _plan_context(source, intent, operations):
    report = diagnose_raw_dataframe(
        source,
        selected_key_columns=intent.selected_key_columns,
    )
    context = build_planning_context(report, intent, ())
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=tuple(operations),
        summary="Trust-boundary regression plan.",
    )
    return report, context, plan, validate_plan(context, plan)


@pytest.mark.parametrize(
    ("source", "keys", "expected_null_rows"),
    [
        (pd.DataFrame({"key": [None, None, 1, 1]}), ("key",), 2),
        (
            pd.DataFrame(
                {
                    "left_key": [1, 1, None, 2],
                    "right_key": ["A", None, "B", "C"],
                }
            ),
            ("left_key", "right_key"),
            2,
        ),
    ],
)
def test_null_keys_are_reported_separately_and_block_dedup(
    source,
    keys,
    expected_null_rows,
) -> None:
    intent = UserIntent(
        intent_id="null-keys",
        selected_key_columns=keys,
        acceptable_row_loss_pct=100,
    )
    report, context, plan, validation = _plan_context(
        source,
        intent,
        (_dedup("dedup", keys),),
    )

    metric = next(item for item in report.key_duplicate_metrics if item.key_columns == keys)
    assert metric.null_key_row_count == expected_null_rows
    assert any(issue.kind is DiagnosticIssueKind.NULL_KEYS for issue in report.issues)
    assert validation.valid is False
    assert "NULL_KEYS_UNSAFE" in {finding.code for finding in validation.findings}


def test_runtime_dataset_change_refuses_dedup_before_null_rows_can_merge() -> None:
    source = pd.DataFrame({"key": [1, 1, 2], "value": ["a", "b", "c"]})
    intent = UserIntent(
        intent_id="runtime-null-key",
        selected_key_columns=("key",),
        acceptable_row_loss_pct=40,
    )
    report, context, plan, validation = _plan_context(
        source,
        intent,
        (_dedup("dedup", ("key",)),),
    )
    review = accepted_review(plan, validation)
    changed = source.copy(deep=True)
    changed.loc[0, "key"] = None

    with pytest.raises(ApprovalGateError) as captured:
        execute_approved_plan(
            changed,
            report,
            context,
            intent,
            plan,
            validation,
            review,
            human_approval(plan),
            expected_review_attempt=review.attempt,
        )

    assert "DATASET_CHANGED" in {failure.value for failure in captured.value.failures}


def test_cumulative_row_loss_rejects_individually_acceptable_operations() -> None:
    source = pd.DataFrame(
        {
            "left_id": [1, 1, 2, 3],
            "right_id": [1, 2, 3, 3],
        }
    )
    intent = UserIntent(intent_id="cumulative", acceptable_row_loss_pct=25)
    _, _, _, validation = _plan_context(
        source,
        intent,
        (
            _dedup("dedup-left", ("left_id",)),
            _dedup("dedup-right", ("right_id",)),
        ),
    )

    assert [estimate.estimated_pct for estimate in validation.row_loss_estimates] == [25, 25]
    assert validation.cumulative_estimated_row_loss_pct == 50
    assert "CUMULATIVE_ROW_LOSS_THRESHOLD" in {
        finding.code for finding in validation.findings
    }


@pytest.mark.parametrize(
    ("decision", "plan_id", "attempt"),
    [
        (ReviewerDecision.REVISE, "current", 1),
        (ReviewerDecision.REJECT, "current", 1),
        (ReviewerDecision.ACCEPT, "foreign", 1),
        (ReviewerDecision.ACCEPT, "current", 2),
    ],
)
def test_review_evidence_rejects_nonaccepting_foreign_and_stale_verdicts(
    decision,
    plan_id,
    attempt,
) -> None:
    source = pd.DataFrame({"label": [" A "]})
    intent = UserIntent(intent_id="review")
    _, _, plan, validation = _plan_context(source, intent, (_trim(),))
    verdict = ReviewerVerdict(
        plan_id=plan.plan_id if plan_id == "current" else plan_id,
        attempt=attempt,
        decision=decision,
    )

    with pytest.raises(ReviewEvidenceError):
        accept_review(plan, validation, verdict, attempt=1)


def test_executor_requires_accepted_review_evidence() -> None:
    source = pd.DataFrame({"label": [" A "]})
    intent = UserIntent(intent_id="review")
    report, context, plan, validation = _plan_context(source, intent, (_trim(),))

    with pytest.raises(ApprovalGateError) as captured:
        execute_approved_plan(
            source,
            report,
            context,
            intent,
            plan,
            validation,
            None,
            human_approval(plan),
            expected_review_attempt=1,
        )

    assert "MISSING_ACCEPTED_REVIEW" in {
        failure.value for failure in captured.value.failures
    }

    valid_review = accepted_review(plan, validation)
    stale_review = valid_review.model_copy(update={"attempt": 2})
    with pytest.raises(ApprovalGateError) as stale:
        execute_approved_plan(
            source,
            report,
            context,
            intent,
            plan,
            validation,
            stale_review,
            human_approval(plan),
            expected_review_attempt=valid_review.attempt,
        )
    assert "REVIEW_EVIDENCE_CHANGED" in {
        failure.value for failure in stale.value.failures
    }


def test_workflow_never_accepts_foreign_reviewer_output() -> None:
    source = pd.DataFrame({"label": [" A "]})
    intent = UserIntent(intent_id="foreign-review")
    _, _, plan, _ = _plan_context(source, intent, (_trim(),))

    class ForeignReviewer:
        def review(
            self,
            context,
            proposed_plan,
            validation,
            *,
            previous_feedback,
            attempt,
        ):
            del context, proposed_plan, validation, previous_feedback
            return ReviewerVerdict(
                plan_id="foreign-plan",
                attempt=attempt,
                decision=ReviewerDecision.ACCEPT,
            )

    runtime = prepare_workflow(
        source,
        intent,
        SequencePlanner((plan, plan, plan)),
        ForeignReviewer(),
    )

    assert runtime.state.stage.value == "PLAN_REJECTED"
    assert runtime.state.last_error_code == "REVIEW_EVIDENCE_ATTEMPTS_EXHAUSTED"


def test_qa_provenance_mismatch_forces_failure() -> None:
    source = pd.DataFrame({"label": [" A "]})
    intent = UserIntent(intent_id="provenance")
    report, context, plan, validation = _plan_context(source, intent, (_trim(),))
    review = accepted_review(plan, validation)
    bundle = execute_approved_plan(
        source,
        report,
        context,
        intent,
        plan,
        validation,
        review,
        human_approval(plan),
        expected_review_attempt=review.attempt,
    )
    assert bundle.dataframe is not None
    foreign_execution = bundle.result.model_copy(update={"plan_id": "foreign-plan"})

    qa = run_quality_assurance(
        source,
        bundle.dataframe,
        foreign_execution,
        report,
        context,
        intent,
        plan,
        validation,
        review,
        human_approval(plan),
    )

    assert qa.status is QAStatus.FAIL
    assert any(
        result.invariant_id == "provenance-execution-plan"
        and result.status.value == "FAIL"
        for result in qa.invariant_results
    )


def test_execution_result_rejects_structural_contradictions() -> None:
    identity = identify_dataset(pd.DataFrame({"value": [1]}))
    common = dict(
        execution_id="execution-contradiction",
        dataset_id=identity.dataset_id,
        plan_id="plan-contradiction",
        plan_version=1,
        accepted_review_attempt=1,
        source_fingerprint=identity.fingerprint,
        before_row_count=1,
        after_row_count=1,
        before_column_count=1,
        after_column_count=1,
    )

    with pytest.raises(ValidationError):
        ExecutionResult(success=True, **common)
    with pytest.raises(ValidationError):
        ExecutionResult(
            success=False,
            result_fingerprint=identity.fingerprint,
            error_code="FAILED",
            **common,
        )
    with pytest.raises(ValidationError):
        ExecutionResult(
            success=False,
            error_code="FAILED",
            operation_records=(
                OperationExecutionRecord(
                    operation_id="failed-operation",
                    status=OperationExecutionStatus.FAILED,
                    rows_before=2,
                    rows_after=1,
                    affected_cell_count=0,
                    error_code="OPERATION_RUNTIME_ERROR",
                ),
            ),
            **common,
        )
