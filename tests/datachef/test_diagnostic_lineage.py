from __future__ import annotations

import pandas as pd

from datachef.contracts import (
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    DiagnosticIssueKind,
    DiagnosticResolution,
    ExecutionResult,
    OperationType,
    QAStatus,
    RenameColumnParameters,
    RiskLevel,
    TransformationOperation,
    UserIntent,
)
from datachef.diagnostics import dataframe_fingerprint, diagnose_raw_dataframe
from datachef.planning import create_transformation_plan, validate_plan
from datachef.privacy import build_planning_context
from datachef.qa import run_quality_assurance
from datachef.transform.executor import execute_approved_plan
from support import accepted_review, human_approval


def _rename(operation_id: str, source: str, target: str) -> TransformationOperation:
    return TransformationOperation(
        operation_id=operation_id,
        operation_type=OperationType.RENAME_COLUMN,
        target_columns=(source,),
        parameters=RenameColumnParameters(new_name=target),
        user_requirement_ids=(operation_id,),
        rationale="Trace diagnostic evidence through a rename.",
        expected_effect="Change only the column label.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )


def _numeric_cast(operation_id: str, column: str) -> TransformationOperation:
    return TransformationOperation(
        operation_id=operation_id,
        operation_type=OperationType.CAST_COLUMN,
        target_columns=(column,),
        parameters=CastColumnParameters(
            target_type=CastTarget.NUMERIC,
            errors=CastErrorPolicy.COERCE,
        ),
        user_requirement_ids=(operation_id,),
        rationale="Convert numeric text after rename.",
        expected_effect="Produce numeric dtype without losing values.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )


def _run(source, operations):
    intent = UserIntent(intent_id="lineage", acceptable_row_loss_pct=0)
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=tuple(operations),
        summary="Diagnostic lineage plan.",
    )
    validation = validate_plan(context, plan)
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
    qa = run_quality_assurance(
        source,
        bundle.dataframe,
        bundle.result,
        report,
        context,
        intent,
        plan,
        validation,
        review,
        human_approval(plan),
    )
    return report, qa


def _comparison(report, qa, kind):
    issue_id = next(issue.issue_id for issue in report.issues if issue.kind is kind)
    return next(item for item in qa.diagnostic_comparisons if item.issue_id == issue_id)


def test_renamed_nullable_column_remains_unresolved() -> None:
    source = pd.DataFrame({"old": [1, None]})
    report, qa = _run(source, (_rename("rename", "old", "new"),))

    comparison = _comparison(report, qa, DiagnosticIssueKind.NULL_VALUES)
    assert comparison.status is DiagnosticResolution.UNCHANGED
    assert comparison.before_value == comparison.after_value == 1


def test_chained_renames_follow_original_diagnostic_column() -> None:
    source = pd.DataFrame({"old": [1, None]})
    report, qa = _run(
        source,
        (
            _rename("rename-one", "old", "middle"),
            _rename("rename-two", "middle", "final"),
        ),
    )

    comparison = _comparison(report, qa, DiagnosticIssueKind.NULL_VALUES)
    assert comparison.status is DiagnosticResolution.UNCHANGED
    assert comparison.after_value == 1


def test_rename_followed_by_lossless_cast_resolves_conversion_not_null_issue() -> None:
    source = pd.DataFrame({"amount_text": ["1", "2", None]})
    report, qa = _run(
        source,
        (
            _rename("rename", "amount_text", "amount_clean"),
            _numeric_cast("cast", "amount_clean"),
        ),
    )

    null_comparison = _comparison(report, qa, DiagnosticIssueKind.NULL_VALUES)
    cast_comparison = _comparison(
        report,
        qa,
        DiagnosticIssueKind.CANDIDATE_TYPE_CONVERSION,
    )
    assert null_comparison.status is DiagnosticResolution.UNCHANGED
    assert cast_comparison.status is DiagnosticResolution.RESOLVED


def test_unavailable_lineage_is_not_reported_as_resolved() -> None:
    source = pd.DataFrame({"nullable": [1, None]})
    transformed = pd.DataFrame(index=source.index)
    intent = UserIntent(intent_id="unavailable")
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(),
        summary="No declared operation.",
    )
    validation = validate_plan(context, plan)
    review = accepted_review(plan, validation)
    execution = ExecutionResult(
        execution_id="execution-unavailable",
        dataset_id=plan.dataset_id,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        accepted_review_attempt=review.attempt,
        success=True,
        source_fingerprint=dataframe_fingerprint(source),
        result_fingerprint=dataframe_fingerprint(transformed),
        before_row_count=len(source),
        after_row_count=len(transformed),
        before_column_count=source.shape[1],
        after_column_count=transformed.shape[1],
    )

    qa = run_quality_assurance(
        source,
        transformed,
        execution,
        report,
        context,
        intent,
        plan,
        validation,
        review,
        human_approval(plan),
    )

    comparison = _comparison(report, qa, DiagnosticIssueKind.NULL_VALUES)
    assert comparison.status is DiagnosticResolution.NOT_APPLICABLE
    assert qa.status is QAStatus.FAIL
