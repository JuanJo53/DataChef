from __future__ import annotations

import pandas as pd
import pytest

from datachef.contracts import (
    AcceptedReviewEvidence,
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    ExecutionResult,
    InvariantKind,
    InvariantStatus,
    OperationExecutionRecord,
    OperationExecutionStatus,
    OperationType,
    PlanValidationResult,
    QAStatus,
    HumanDecision,
    DeduplicateByKeysParameters,
    DropDuplicateRowsParameters,
    KeepPolicy,
    NormalizeMissingTokensParameters,
    RenameColumnParameters,
    RiskLevel,
    TransformationOperation,
    TrimWhitespaceParameters,
    UserIntent,
)
from datachef.diagnostics import dataframe_fingerprint, diagnose_raw_dataframe
from datachef.planning import create_transformation_plan, validate_plan
from datachef.privacy import build_planning_context
from datachef.qa import run_quality_assurance
from datachef.transform.executor import (
    ApprovalFailure,
    ApprovalGateError,
    execute_approved_plan,
)
from support import accepted_review, human_approval


def _plan(source: pd.DataFrame, operation: TransformationOperation):
    intent = UserIntent(intent_id="r2-trust", acceptable_row_loss_pct=100)
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(operation,),
        summary="R2 adversarial plan.",
    )
    return intent, report, context, plan, validate_plan(context, plan)


def _trim(operation_id: str = "duplicate") -> TransformationOperation:
    return TransformationOperation(
        operation_id=operation_id,
        operation_type=OperationType.TRIM_WHITESPACE,
        target_columns=("label",),
        parameters=TrimWhitespaceParameters(),
        user_requirement_ids=("trim",),
        rationale="Normalize surrounding whitespace.",
        expected_effect="Trim the configured label.",
        risk=RiskLevel.LOW,
        requires_human_approval=False,
    )


def test_executor_recomputes_validation_instead_of_trusting_typed_claim() -> None:
    source = pd.DataFrame({"label": [" A "]})
    intent = UserIntent(intent_id="forged-validation", acceptable_row_loss_pct=100)
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    operation = _trim()
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(operation, operation),
        summary="Duplicate operation identifiers are invalid.",
    )
    authentic = validate_plan(context, plan)
    assert authentic.valid is False
    forged = PlanValidationResult(plan_id=plan.plan_id, valid=True)
    forged_review = AcceptedReviewEvidence(
        dataset_id=plan.dataset_id,
        dataset_fingerprint=plan.dataset_fingerprint,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        attempt=1,
        validation_plan_id=plan.plan_id,
    )

    with pytest.raises(ApprovalGateError):
        execute_approved_plan(
            source,
            report,
            context,
            intent,
            plan,
            forged,
            forged_review,
            human_approval(plan),
            expected_review_attempt=1,
        )


def test_qa_replay_rejects_forged_cast_effect_metadata() -> None:
    source = pd.DataFrame({"value": ["1", "bad"]})
    operation = TransformationOperation(
        operation_id="cast-value",
        operation_type=OperationType.CAST_COLUMN,
        target_columns=("value",),
        parameters=CastColumnParameters(
            target_type=CastTarget.NUMERIC,
            errors=CastErrorPolicy.COERCE,
        ),
        user_requirement_ids=("numeric",),
        rationale="Convert numeric text.",
        expected_effect="Inspect a coercive numeric conversion.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )
    intent, report, context, plan, validation = _plan(source, operation)
    review = accepted_review(plan, validation)
    approval = human_approval(plan)
    transformed = pd.DataFrame({"value": pd.to_numeric(source["value"], errors="coerce")})
    execution = ExecutionResult(
        execution_id="forged-cast-effects",
        dataset_id=plan.dataset_id,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        accepted_review_attempt=review.attempt,
        success=True,
        source_fingerprint=dataframe_fingerprint(source),
        result_fingerprint=dataframe_fingerprint(transformed),
        before_row_count=2,
        after_row_count=2,
        before_column_count=1,
        after_column_count=1,
        operation_records=(
            OperationExecutionRecord(
                operation_id="cast-value",
                status=OperationExecutionStatus.APPLIED,
                rows_before=2,
                rows_after=2,
                affected_cell_count=2,
                introduced_null_count=0,
            ),
        ),
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
        approval=approval,
    )

    assert qa.status is QAStatus.FAIL
    assert any(
        item.kind is InvariantKind.PROVENANCE
        and item.status is InvariantStatus.FAIL
        for item in qa.invariant_results
    )
    assert any(
        item.kind is InvariantKind.CAST_VALUE_PRESERVATION
        and item.status is InvariantStatus.FAIL
        for item in qa.invariant_results
    )


def _execute_cast(source: pd.DataFrame):
    operation = TransformationOperation(
        operation_id="cast-value",
        operation_type=OperationType.CAST_COLUMN,
        target_columns=("value",),
        parameters=CastColumnParameters(
            target_type=CastTarget.NUMERIC,
            errors=CastErrorPolicy.COERCE,
        ),
        user_requirement_ids=("numeric",),
        rationale="Convert numeric text.",
        expected_effect="Produce a numeric dtype.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )
    intent, report, context, plan, validation = _plan(source, operation)
    review = accepted_review(plan, validation)
    approval = human_approval(plan)
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
    return intent, report, context, plan, validation, review, approval, bundle


@pytest.mark.parametrize("tamper", ["missing_effect", "missing_record", "duplicate_record", "row_count"])
def test_qa_replay_rejects_tampered_execution_records(tamper: str) -> None:
    source = pd.DataFrame({"value": ["1", "2"]})
    intent, report, context, plan, validation, review, approval, bundle = _execute_cast(source)
    record = bundle.result.operation_records[0]
    if tamper == "missing_effect":
        records = (record.model_copy(update={"introduced_null_count": None}),)
    elif tamper == "missing_record":
        records = ()
    elif tamper == "duplicate_record":
        records = (record, record)
    else:
        records = (
            record.model_copy(update={"rows_before": 99, "rows_after": 99}),
        )
    forged = bundle.result.model_copy(update={"operation_records": records})

    qa = run_quality_assurance(
        source,
        bundle.dataframe,
        forged,
        report,
        context,
        intent,
        plan,
        validation,
        review,
        approval,
    )

    assert qa.status is QAStatus.FAIL
    assert any(
        item.invariant_id == "provenance-replay-records"
        and item.status is InvariantStatus.FAIL
        for item in qa.invariant_results
    )


@pytest.mark.parametrize(
    "approval_kind",
    ["missing", "rejected", "foreign", "partial", "stale_version"],
)
def test_qa_requires_matching_human_approval(approval_kind: str) -> None:
    source = pd.DataFrame({"value": ["1", "2"]})
    intent, report, context, plan, validation, review, approval, bundle = _execute_cast(source)
    if approval_kind == "missing":
        supplied = None
    elif approval_kind == "rejected":
        supplied = approval.model_copy(update={"decision": HumanDecision.REJECT})
    elif approval_kind == "foreign":
        supplied = approval.model_copy(update={"plan_id": "foreign-plan"})
    elif approval_kind == "partial":
        supplied = approval.model_copy(update={"approved_operation_ids": ()})
    else:
        supplied = approval.model_copy(update={"plan_version": plan.version + 1})

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
        supplied,
    )

    assert qa.status is QAStatus.FAIL
    assert any(
        item.invariant_id == "provenance-human-approval"
        and item.status is InvariantStatus.FAIL
        for item in qa.invariant_results
    )


def test_replay_attributes_null_loss_to_the_correct_cast_step() -> None:
    source = pd.DataFrame({"value": [" N/A ", "2", "bad"]})
    trim = TransformationOperation(
        operation_id="trim",
        operation_type=OperationType.TRIM_WHITESPACE,
        target_columns=("value",),
        parameters=TrimWhitespaceParameters(),
        user_requirement_ids=("trim",),
        rationale="Trim first.",
        expected_effect="Remove spaces.",
        risk=RiskLevel.LOW,
        requires_human_approval=False,
    )
    normalize = TransformationOperation(
        operation_id="normalize",
        operation_type=OperationType.NORMALIZE_MISSING_TOKENS,
        target_columns=("value",),
        parameters=NormalizeMissingTokensParameters(tokens=("N/A",)),
        user_requirement_ids=("missing",),
        rationale="Normalize one approved missing token.",
        expected_effect="Represent the token as null.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )
    cast_one = TransformationOperation(
        operation_id="cast-one",
        operation_type=OperationType.CAST_COLUMN,
        target_columns=("value",),
        parameters=CastColumnParameters(
            target_type=CastTarget.NUMERIC,
            errors=CastErrorPolicy.COERCE,
        ),
        user_requirement_ids=("numeric-one",),
        rationale="Convert numeric text.",
        expected_effect="Inspect invalid numeric values.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )
    cast_two = cast_one.model_copy(
        update={"operation_id": "cast-two", "user_requirement_ids": ("numeric-two",)}
    )
    intent = UserIntent(intent_id="cast-sequence", acceptable_row_loss_pct=100)
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(trim, normalize, cast_one, cast_two),
        summary="Trace null introduction per cast operation.",
    )
    validation = validate_plan(context, plan)
    review = accepted_review(plan, validation)
    approval = human_approval(plan)
    bundle = execute_approved_plan(
        source,
        report,
        context,
        intent,
        plan,
        validation,
        review,
        approval,
        expected_review_attempt=1,
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
    cast_results = {
        item.invariant_id: item
        for item in qa.invariant_results
        if item.kind is InvariantKind.CAST_VALUE_PRESERVATION
    }

    assert cast_results["cast-preservation-cast-one"].observed_value == 1
    assert cast_results["cast-preservation-cast-one"].status is InvariantStatus.FAIL
    assert cast_results["cast-preservation-cast-two"].observed_value == 0
    assert qa.status is QAStatus.FAIL

    reordered = bundle.result.model_copy(
        update={"operation_records": tuple(reversed(bundle.result.operation_records))}
    )
    reordered_qa = run_quality_assurance(
        source,
        bundle.dataframe,
        reordered,
        report,
        context,
        intent,
        plan,
        validation,
        review,
        approval,
    )
    assert next(
        item
        for item in reordered_qa.invariant_results
        if item.invariant_id == "provenance-replay-records"
    ).status is InvariantStatus.FAIL

    forged_shape = bundle.result.model_copy(
        update={"after_row_count": bundle.result.after_row_count + 1}
    )
    shape_qa = run_quality_assurance(
        source,
        bundle.dataframe,
        forged_shape,
        report,
        context,
        intent,
        plan,
        validation,
        review,
        approval,
    )
    assert shape_qa.status is QAStatus.FAIL
    assert next(
        item
        for item in shape_qa.invariant_results
        if item.invariant_id == "provenance-replay-shape"
    ).status is InvariantStatus.FAIL


def test_executor_revalidation_rejects_fabricated_valid_claims_for_all_rules() -> None:
    scenarios: list[tuple[pd.DataFrame, UserIntent, TransformationOperation]] = []

    protected_source = pd.DataFrame({"internal_code": [" A "]})
    protected_intent = UserIntent(
        intent_id="protected",
        protected_columns=("internal_code",),
    )
    protected_report = diagnose_raw_dataframe(protected_source)
    protected_context = build_planning_context(protected_report, protected_intent, ())
    protected_alias = protected_context.privacy_manifest.aliased_columns[0]
    scenarios.append((protected_source, protected_intent, _trim().model_copy(
        update={"target_columns": (protected_alias,), "operation_id": "protected"}
    )))

    privacy_source = pd.DataFrame({"email": ["a@example.test"]})
    privacy_intent = UserIntent(intent_id="privacy")
    privacy_report = diagnose_raw_dataframe(privacy_source)
    privacy_context = build_planning_context(privacy_report, privacy_intent, ())
    privacy_alias = privacy_context.privacy_manifest.aliased_columns[0]
    scenarios.append((privacy_source, privacy_intent, _trim().model_copy(
        update={"target_columns": (privacy_alias,), "operation_id": "privacy"}
    )))

    scenarios.extend(
        (
            (
                pd.DataFrame({"left": [1], "right": [2]}),
                UserIntent(intent_id="collision"),
                TransformationOperation(
                    operation_id="collision",
                    operation_type=OperationType.RENAME_COLUMN,
                    target_columns=("left",),
                    parameters=RenameColumnParameters(new_name="right"),
                    user_requirement_ids=("rename",),
                    rationale="Create a forbidden collision.",
                    expected_effect="Must be rejected.",
                    risk=RiskLevel.MEDIUM,
                    requires_human_approval=True,
                ),
            ),
            (
                pd.DataFrame({"value": [1, 1]}),
                UserIntent(intent_id="row-loss", acceptable_row_loss_pct=0),
                TransformationOperation(
                    operation_id="row-loss",
                    operation_type=OperationType.DROP_DUPLICATE_ROWS,
                    parameters=DropDuplicateRowsParameters(keep=KeepPolicy.FIRST),
                    user_requirement_ids=("dedup",),
                    rationale="Exceed the row-loss threshold.",
                    expected_effect="Must be rejected.",
                    risk=RiskLevel.HIGH,
                    requires_human_approval=True,
                ),
            ),
            (
                pd.DataFrame({"key": [1, None, 1]}),
                UserIntent(
                    intent_id="null-key",
                    selected_key_columns=("key",),
                    acceptable_row_loss_pct=100,
                ),
                TransformationOperation(
                    operation_id="null-key",
                    operation_type=OperationType.DEDUPLICATE_BY_KEYS,
                    target_columns=("key",),
                    parameters=DeduplicateByKeysParameters(
                        keys=("key",),
                        keep=KeepPolicy.FIRST,
                    ),
                    user_requirement_ids=("key",),
                    rationale="Deduplicate a forbidden nullable key.",
                    expected_effect="Must be rejected.",
                    risk=RiskLevel.HIGH,
                    requires_human_approval=True,
                ),
            ),
        )
    )

    for source, intent, operation in scenarios:
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
            summary="Fabricated validation claim.",
        )
        assert validate_plan(context, plan).valid is False
        forged = PlanValidationResult(plan_id=plan.plan_id, valid=True)
        forged_review = AcceptedReviewEvidence(
            dataset_id=plan.dataset_id,
            dataset_fingerprint=plan.dataset_fingerprint,
            plan_id=plan.plan_id,
            plan_version=plan.version,
            attempt=1,
            validation_plan_id=plan.plan_id,
        )
        with pytest.raises(ApprovalGateError):
            execute_approved_plan(
                source,
                report,
                context,
                intent,
                plan,
                forged,
                forged_review,
                human_approval(plan),
                expected_review_attempt=1,
            )

    source = pd.DataFrame({"label": [" A "]})
    intent, report, context, canonical, _ = _plan(source, _trim("canonical"))
    noncanonical = canonical.model_copy(update={"plan_id": "plan-00000000000000000000"})
    forged = PlanValidationResult(plan_id=noncanonical.plan_id, valid=True)
    forged_review = AcceptedReviewEvidence(
        dataset_id=noncanonical.dataset_id,
        dataset_fingerprint=noncanonical.dataset_fingerprint,
        plan_id=noncanonical.plan_id,
        plan_version=noncanonical.version,
        attempt=1,
        validation_plan_id=noncanonical.plan_id,
    )
    with pytest.raises(ApprovalGateError):
        execute_approved_plan(
            source,
            report,
            context,
            intent,
            noncanonical,
            forged,
            forged_review,
            human_approval(noncanonical),
            expected_review_attempt=1,
        )


def test_executor_recomputes_context_and_diagnostics_from_local_intent() -> None:
    protected_source = pd.DataFrame({"internal_code": [" A "]})
    protected_intent = UserIntent(
        intent_id="authoritative-protected",
        protected_columns=("internal_code",),
    )
    authentic_report = diagnose_raw_dataframe(protected_source)
    fabricated_context = build_planning_context(
        authentic_report,
        UserIntent(intent_id="fabricated-unprotected"),
        (),
    )
    operation = _trim("hidden-protection").model_copy(
        update={"target_columns": ("internal_code",)}
    )
    plan = create_transformation_plan(
        dataset_id=fabricated_context.dataset_identity.dataset_id,
        dataset_fingerprint=fabricated_context.dataset_identity.fingerprint,
        version=1,
        operations=(operation,),
        summary="A fabricated context omits the protected-column fact.",
    )
    fabricated_validation = validate_plan(fabricated_context, plan)
    assert fabricated_validation.valid is True

    with pytest.raises(ApprovalGateError) as protected_error:
        execute_approved_plan(
            protected_source,
            authentic_report,
            fabricated_context,
            protected_intent,
            plan,
            fabricated_validation,
            accepted_review(plan, fabricated_validation),
            human_approval(plan),
            expected_review_attempt=1,
        )
    assert ApprovalFailure.DIAGNOSTIC_CONTEXT_CHANGED in protected_error.value.failures

    fabricated_intent = UserIntent(intent_id="fabricated-unprotected")
    fabricated_bundle = execute_approved_plan(
        protected_source,
        authentic_report,
        fabricated_context,
        fabricated_intent,
        plan,
        fabricated_validation,
        accepted_review(plan, fabricated_validation),
        human_approval(plan),
        expected_review_attempt=1,
    )
    assert fabricated_bundle.dataframe is not None
    independent_qa = run_quality_assurance(
        protected_source,
        fabricated_bundle.dataframe,
        fabricated_bundle.result,
        authentic_report,
        fabricated_context,
        protected_intent,
        plan,
        fabricated_validation,
        accepted_review(plan, fabricated_validation),
        human_approval(plan),
    )
    assert independent_qa.status is QAStatus.FAIL
    assert next(
        item
        for item in independent_qa.invariant_results
        if item.invariant_id == "provenance-planning-context"
    ).status is InvariantStatus.FAIL

    null_key_source = pd.DataFrame({"key": [1, None, 1]})
    key_intent = UserIntent(
        intent_id="authoritative-null-key",
        selected_key_columns=("key",),
        acceptable_row_loss_pct=100,
    )
    fabricated_report = diagnose_raw_dataframe(null_key_source)
    fabricated_key_context = build_planning_context(
        fabricated_report,
        key_intent,
        (),
    )
    dedup = TransformationOperation(
        operation_id="hidden-null-key",
        operation_type=OperationType.DEDUPLICATE_BY_KEYS,
        target_columns=("key",),
        parameters=DeduplicateByKeysParameters(keys=("key",), keep=KeepPolicy.FIRST),
        user_requirement_ids=("key",),
        rationale="A fabricated report omits null-key evidence.",
        expected_effect="Must be rejected after authoritative diagnosis.",
        risk=RiskLevel.HIGH,
        requires_human_approval=True,
    )
    key_plan = create_transformation_plan(
        dataset_id=fabricated_key_context.dataset_identity.dataset_id,
        dataset_fingerprint=fabricated_key_context.dataset_identity.fingerprint,
        version=1,
        operations=(dedup,),
        summary="A fabricated diagnostic report omits selected-key metrics.",
    )
    fabricated_key_validation = validate_plan(fabricated_key_context, key_plan)
    assert fabricated_key_validation.valid is True

    with pytest.raises(ApprovalGateError) as key_error:
        execute_approved_plan(
            null_key_source,
            fabricated_report,
            fabricated_key_context,
            key_intent,
            key_plan,
            fabricated_key_validation,
            accepted_review(key_plan, fabricated_key_validation),
            human_approval(key_plan),
            expected_review_attempt=1,
        )
    assert ApprovalFailure.INVALID_PLAN in key_error.value.failures
