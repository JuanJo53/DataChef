from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from datachef.contracts import (
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    DeduplicateByKeysParameters,
    DownstreamUse,
    ExecutionResult,
    KeepPolicy,
    OperationType,
    QAStatus,
    RenameColumnParameters,
    RiskLevel,
    TransformationOperation,
    UserIntent,
)
from datachef.diagnostics import diagnose_raw_dataframe, identify_dataset
from datachef.planning import create_transformation_plan, validate_plan
from datachef.privacy import build_planning_context
from datachef.qa import run_quality_assurance
from datachef.transform.executor import ApprovalGateError, execute_approved_plan
from support import accepted_review, human_approval


def _setup(source: pd.DataFrame, intent: UserIntent, operation: TransformationOperation):
    report = diagnose_raw_dataframe(
        source,
        selected_key_columns=intent.selected_key_columns,
    )
    context = build_planning_context(report, intent, ())
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(operation,),
        summary="Execution test plan.",
    )
    validation = validate_plan(context, plan)
    review = accepted_review(plan, validation)
    approval = human_approval(plan)
    return report, context, plan, validation, review, approval


def _dedup_operation(keep: KeepPolicy = KeepPolicy.FIRST) -> TransformationOperation:
    return TransformationOperation(
        operation_id="op-deduplicate",
        operation_type=OperationType.DEDUPLICATE_BY_KEYS,
        target_columns=("customer_id",),
        parameters=DeduplicateByKeysParameters(
            keys=("customer_id",),
            keep=keep,
        ),
        user_requirement_ids=("intent.key_columns",),
        rationale="User requires one row per approved key.",
        expected_effect="Remove later or earlier duplicate-key rows.",
        risk=RiskLevel.HIGH,
        requires_human_approval=True,
    )


def test_unapproved_plan_cannot_execute(raw_dataframe, user_intent) -> None:
    report, context, plan, validation, review, _ = _setup(
        raw_dataframe, user_intent, _dedup_operation()
    )

    with pytest.raises(ApprovalGateError) as captured:
        execute_approved_plan(
            raw_dataframe,
            report,
            context,
            user_intent,
            plan,
            validation,
            review,
            None,
            expected_review_attempt=review.attempt,
        )

    assert captured.value.failures[0].value == "MISSING_APPROVAL"


def test_changed_plan_invalidates_approval(raw_dataframe, user_intent) -> None:
    report, context, original, _, review, approval = _setup(
        raw_dataframe, user_intent, _dedup_operation()
    )
    changed = create_transformation_plan(
        dataset_id=original.dataset_id,
        dataset_fingerprint=original.dataset_fingerprint,
        version=2,
        operations=original.operations,
        summary="Changed after approval.",
    )
    changed_validation = validate_plan(context, changed)

    with pytest.raises(ApprovalGateError) as captured:
        execute_approved_plan(
            raw_dataframe,
            report,
            context,
            user_intent,
            changed,
            changed_validation,
            review,
            approval,
            expected_review_attempt=review.attempt,
        )

    assert "PLAN_CHANGED" in {failure.value for failure in captured.value.failures}


def test_changed_plan_cannot_reuse_canonical_plan_id(raw_dataframe, user_intent) -> None:
    report, context, original, _, review, approval = _setup(
        raw_dataframe, user_intent, _dedup_operation()
    )
    stale_identity = original.model_copy(update={"summary": "Changed content."})
    changed_validation = validate_plan(context, stale_identity)

    assert changed_validation.valid is False
    assert "PLAN_ID_MISMATCH" in {
        finding.code for finding in changed_validation.findings
    }
    with pytest.raises(ApprovalGateError) as captured:
        execute_approved_plan(
            raw_dataframe,
            report,
            context,
            user_intent,
            stale_identity,
            changed_validation,
            review,
            approval,
            expected_review_attempt=review.attempt,
        )
    assert "INVALID_PLAN" in {failure.value for failure in captured.value.failures}


def test_changed_dataset_invalidates_approval(raw_dataframe, user_intent) -> None:
    report, context, plan, validation, review, approval = _setup(
        raw_dataframe, user_intent, _dedup_operation()
    )
    changed = raw_dataframe.copy(deep=True)
    changed.loc[0, "category"] = "Changed"

    with pytest.raises(ApprovalGateError) as captured:
        execute_approved_plan(
            changed,
            report,
            context,
            user_intent,
            plan,
            validation,
            review,
            approval,
            expected_review_attempt=review.attempt,
        )

    assert "DATASET_CHANGED" in {failure.value for failure in captured.value.failures}


def test_execution_and_retry_use_fresh_copies(raw_dataframe, user_intent) -> None:
    original = raw_dataframe.copy(deep=True)
    report, context, plan, validation, review, approval = _setup(
        raw_dataframe, user_intent, _dedup_operation()
    )

    first = execute_approved_plan(
        raw_dataframe,
        report,
        context,
        user_intent,
        plan,
        validation,
        review,
        approval,
        expected_review_attempt=review.attempt,
    )
    second = execute_approved_plan(
        raw_dataframe,
        report,
        context,
        user_intent,
        plan,
        validation,
        review,
        approval,
        expected_review_attempt=review.attempt,
    )

    assert first.dataframe is not None and second.dataframe is not None
    assert first.dataframe is not second.dataframe
    assert_frame_equal(first.dataframe, second.dataframe)
    assert_frame_equal(raw_dataframe, original)


def test_deduplication_respects_keys_and_keep_policy(raw_dataframe, user_intent) -> None:
    source = raw_dataframe.copy(deep=True)
    source.loc[0, "category"] = "first"
    source.loc[1, "category"] = "last"
    report, context, plan, validation, review, approval = _setup(
        source, user_intent, _dedup_operation(KeepPolicy.LAST)
    )

    bundle = execute_approved_plan(
        source,
        report,
        context,
        user_intent,
        plan,
        validation,
        review,
        approval,
        expected_review_attempt=review.attempt,
    )

    assert bundle.dataframe is not None
    kept = bundle.dataframe.loc[bundle.dataframe["customer_id"] == 101, "category"]
    assert kept.tolist() == ["last"]


def test_type_conversion_produces_measurable_qa() -> None:
    source = pd.DataFrame({"amount_text": ["10", "20", "30"]})
    intent = UserIntent(
        intent_id="intent-cast",
        downstream_use=DownstreamUse.ANALYSIS,
        required_columns=("amount_text",),
        acceptable_row_loss_pct=0.0,
    )
    operation = TransformationOperation(
        operation_id="op-cast",
        operation_type=OperationType.CAST_COLUMN,
        target_columns=("amount_text",),
        parameters=CastColumnParameters(
            target_type=CastTarget.NUMERIC,
            errors=CastErrorPolicy.RAISE,
        ),
        user_requirement_ids=("intent.explicit.0",),
        rationale="User requested numeric analysis.",
        expected_effect="Convert numeric text to numeric dtype.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )
    report, context, plan, validation, review, approval = _setup(source, intent, operation)
    bundle = execute_approved_plan(
        source,
        report,
        context,
        intent,
        plan,
        validation,
        review,
        approval,
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
        approval,
    )

    assert qa.status is QAStatus.PASS
    assert qa.dtype_changes[0].column == "amount_text"


def _successful_execution(
    source: pd.DataFrame,
    transformed: pd.DataFrame,
    plan,
    review,
) -> ExecutionResult:
    identity = identify_dataset(source)
    return ExecutionResult(
        execution_id="execution-direct-qa",
        dataset_id=identity.dataset_id,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        accepted_review_attempt=review.attempt,
        success=True,
        source_fingerprint=identity.fingerprint,
        result_fingerprint=identify_dataset(transformed).fingerprint,
        before_row_count=len(source),
        after_row_count=len(transformed),
        before_column_count=source.shape[1],
        after_column_count=transformed.shape[1],
    )


def _empty_plan(source: pd.DataFrame):
    identity = identify_dataset(source)
    return create_transformation_plan(
        dataset_id=identity.dataset_id,
        dataset_fingerprint=identity.fingerprint,
        version=1,
        operations=(),
        summary="Direct QA test plan.",
    )


def test_excessive_row_loss_fails_qa(raw_dataframe) -> None:
    transformed = raw_dataframe.iloc[:1].copy()
    intent = UserIntent(
        intent_id="intent-row-loss",
        acceptable_row_loss_pct=10.0,
    )
    report = diagnose_raw_dataframe(raw_dataframe)
    context = build_planning_context(report, intent, ())
    plan = _empty_plan(raw_dataframe)
    validation = validate_plan(context, plan)
    review = accepted_review(plan, validation)
    qa = run_quality_assurance(
        raw_dataframe,
        transformed,
        _successful_execution(raw_dataframe, transformed, plan, review),
        report,
        context,
        intent,
        plan,
        validation,
        review,
        human_approval(plan),
    )

    assert qa.status is QAStatus.FAIL
    assert qa.row_loss_pct == 75.0


def test_required_column_loss_fails_qa(raw_dataframe) -> None:
    transformed = raw_dataframe.drop(columns=["amount_text"])
    intent = UserIntent(
        intent_id="intent-required",
        required_columns=("amount_text",),
        acceptable_row_loss_pct=100.0,
    )
    report = diagnose_raw_dataframe(raw_dataframe)
    context = build_planning_context(report, intent, ())
    plan = _empty_plan(raw_dataframe)
    validation = validate_plan(context, plan)
    review = accepted_review(plan, validation)
    qa = run_quality_assurance(
        raw_dataframe,
        transformed,
        _successful_execution(raw_dataframe, transformed, plan, review),
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
        result.invariant_id == "required-amount_text" and result.status.value == "FAIL"
        for result in qa.invariant_results
    )
