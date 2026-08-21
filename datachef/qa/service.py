"""Compare raw and transformed data using typed deterministic invariants."""

from __future__ import annotations

from hashlib import sha256

import pandas as pd

from datachef.contracts import (
    AcceptedReviewEvidence,
    CastColumnParameters,
    ComputeColumnParameters,
    ComputeOperator,
    DropColumnParameters,
    CastTarget,
    DiagnosticIssueComparison,
    DiagnosticIssueKind,
    DiagnosticReport,
    DiagnosticResolution,
    DTypeChange,
    ExecutionResult,
    HumanApproval,
    HumanDecision,
    InvariantKind,
    InvariantResult,
    InvariantStatus,
    NullCountChange,
    OperationExecutionStatus,
    OperationType,
    PlanValidationResult,
    PlanningContext,
    QAReport,
    QAStatus,
    QualityInvariant,
    RenameColumnParameters,
    TransformationPlan,
    UserIntent,
)
from datachef.diagnostics import dataframe_fingerprint
from datachef.planning.lineage import ColumnLineage
from datachef.planning.authoritative import (
    context_claim_matches,
    recompute_validation_facts,
)
from datachef.transform.runner import OperationRun, run_allowlisted_plan


def _duplicate_key_rows(dataframe: pd.DataFrame, keys: tuple[str, ...]) -> int | None:
    if not keys or any(key not in dataframe.columns for key in keys):
        return None
    return int(dataframe.duplicated(subset=list(keys), keep="first").sum())


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    normalized = expected.upper()
    if normalized == CastTarget.STRING.value:
        return pd.api.types.is_string_dtype(series.dtype)
    if normalized == CastTarget.NUMERIC.value:
        return pd.api.types.is_numeric_dtype(series.dtype)
    if normalized == CastTarget.BOOLEAN.value:
        return pd.api.types.is_bool_dtype(series.dtype)
    if normalized == CastTarget.DATETIME.value:
        return pd.api.types.is_datetime64_any_dtype(series.dtype)
    return str(series.dtype) == expected


def _result(
    invariant: QualityInvariant,
    passed: bool,
    explanation: str,
    *,
    observed: int | float | str | None = None,
    expected: int | float | str | None = None,
) -> InvariantResult:
    status = (
        InvariantStatus.PASS
        if passed
        else InvariantStatus.FAIL
        if invariant.mandatory
        else InvariantStatus.WARN
    )
    return InvariantResult(
        invariant_id=invariant.invariant_id,
        kind=invariant.kind,
        status=status,
        mandatory=invariant.mandatory,
        explanation=explanation,
        observed_value=observed,
        expected_value=expected,
    )


def _evaluate_invariant(
    invariant: QualityInvariant,
    source: pd.DataFrame,
    result: pd.DataFrame,
    row_loss_pct: float,
) -> InvariantResult:
    if invariant.kind is InvariantKind.REQUIRED_COLUMN:
        passed = bool(invariant.column and invariant.column in result.columns)
        return _result(invariant, passed, "Required column presence check.")
    if invariant.kind is InvariantKind.MAX_ROW_LOSS:
        maximum = invariant.maximum_pct if invariant.maximum_pct is not None else 0.0
        return _result(
            invariant,
            row_loss_pct <= maximum,
            "Observed row loss compared with the approved threshold.",
            observed=row_loss_pct,
            expected=maximum,
        )
    if invariant.kind is InvariantKind.DTYPE:
        passed = bool(
            invariant.column
            and invariant.column in result.columns
            and invariant.expected_dtype
            and _dtype_matches(result[invariant.column], invariant.expected_dtype)
        )
        observed = (
            str(result[invariant.column].dtype)
            if invariant.column and invariant.column in result.columns
            else "MISSING"
        )
        return _result(
            invariant,
            passed,
            "Requested dtype check.",
            observed=observed,
            expected=invariant.expected_dtype,
        )
    if invariant.kind is InvariantKind.KEY_UNIQUENESS:
        duplicates = _duplicate_key_rows(result, invariant.key_columns)
        return _result(
            invariant,
            duplicates == 0,
            "Selected key uniqueness check.",
            observed=duplicates,
            expected=0,
        )
    if invariant.kind is InvariantKind.NULL_PCT_MAX:
        maximum = invariant.maximum_pct if invariant.maximum_pct is not None else 0.0
        if not invariant.column or invariant.column not in result.columns:
            observed = 100.0
            passed = False
        else:
            observed = float(result[invariant.column].isna().mean() * 100) if len(result) else 0.0
            passed = observed <= maximum
        return _result(
            invariant,
            passed,
            "Null percentage compared with the user expectation.",
            observed=observed,
            expected=maximum,
        )
    if invariant.kind is InvariantKind.PROTECTED_COLUMN_UNCHANGED:
        column = invariant.column
        passed = bool(column and column in source.columns and column in result.columns)
        if passed:
            try:
                passed = result[column].equals(source.loc[result.index, column])
            except KeyError:
                passed = False
        return _result(invariant, passed, "Protected column value check.")
    return _result(invariant, False, "Unsupported invariant kind.")


def _default_invariants(
    intent: UserIntent,
    plan: TransformationPlan,
    selected_key_columns: tuple[str, ...],
) -> tuple[QualityInvariant, ...]:
    invariants: list[QualityInvariant] = [
        QualityInvariant(
            invariant_id="default-max-row-loss",
            kind=InvariantKind.MAX_ROW_LOSS,
            maximum_pct=intent.acceptable_row_loss_pct,
        )
    ]
    invariants.extend(
        QualityInvariant(
            invariant_id=f"required-{column}",
            kind=InvariantKind.REQUIRED_COLUMN,
            column=column,
        )
        for column in intent.required_columns
    )
    invariants.extend(
        QualityInvariant(
            invariant_id=f"protected-{column}",
            kind=InvariantKind.PROTECTED_COLUMN_UNCHANGED,
            column=column,
        )
        for column in intent.protected_columns
    )
    if selected_key_columns:
        invariants.append(
            QualityInvariant(
                invariant_id="selected-key-unique",
                kind=InvariantKind.KEY_UNIQUENESS,
                key_columns=selected_key_columns,
            )
        )
    invariants.extend(
        QualityInvariant(
            invariant_id=f"null-{expectation.column}",
            kind=InvariantKind.NULL_PCT_MAX,
            mandatory=expectation.mandatory,
            column=expectation.column,
            maximum_pct=expectation.maximum_null_pct,
        )
        for expectation in intent.null_expectations
    )
    for operation in plan.operations:
        if operation.operation_type is OperationType.CAST_COLUMN:
            parameters = operation.parameters
            assert isinstance(parameters, CastColumnParameters)
            invariants.append(
                QualityInvariant(
                    invariant_id=f"cast-{operation.operation_id}",
                    kind=InvariantKind.DTYPE,
                    column=operation.target_columns[0],
                    expected_dtype=parameters.target_type.value,
                )
            )
    return tuple(invariants)


def _provenance_results(
    source: pd.DataFrame,
    transformed: pd.DataFrame,
    execution: ExecutionResult,
    report: DiagnosticReport,
    context: PlanningContext,
    plan: TransformationPlan,
    validation: PlanValidationResult,
    accepted_review: AcceptedReviewEvidence,
    approval: HumanApproval | None,
    replay: OperationRun,
    intent: UserIntent,
) -> tuple[InvariantResult, ...]:
    from datachef.planning.plan import expected_plan_id
    replay_frame = replay.dataframe
    planned_ids = tuple(operation.operation_id for operation in plan.operations)
    authoritative = recompute_validation_facts(source, intent, plan)
    recomputed_validation = authoritative.validation
    checks = (
        (
            "provenance-source-diagnostic",
            report == authoritative.diagnostic_report,
        ),
        (
            "provenance-plan-dataset",
            plan.dataset_id == report.dataset_identity.dataset_id
            and plan.dataset_fingerprint == report.dataset_identity.fingerprint,
        ),
        (
            "provenance-planning-context",
            context.dataset_identity.dataset_id == report.dataset_identity.dataset_id
            and context.dataset_identity.fingerprint
            == report.dataset_identity.fingerprint
            and context_claim_matches(context, authoritative.planning_context),
        ),
        (
            "provenance-canonical-plan",
            plan.plan_id == expected_plan_id(plan),
        ),
        (
            "provenance-plan-validation",
            validation.valid
            and validation.plan_id == plan.plan_id
            and validation == recomputed_validation,
        ),
        (
            "provenance-accepted-review",
            accepted_review.plan_id == plan.plan_id
            and accepted_review.plan_version == plan.version
            and accepted_review.dataset_id == plan.dataset_id
            and accepted_review.dataset_fingerprint == plan.dataset_fingerprint
            and accepted_review.validation_plan_id == validation.plan_id
            and accepted_review.validation_valid,
        ),
        (
            "provenance-human-approval",
            approval is not None
            and approval.decision is HumanDecision.APPROVE
            and approval.dataset_id == plan.dataset_id
            and approval.dataset_fingerprint == plan.dataset_fingerprint
            and approval.plan_id == plan.plan_id
            and approval.plan_version == plan.version
            and approval.approved_operation_ids == planned_ids,
        ),
        (
            "provenance-execution-plan",
            execution.dataset_id == plan.dataset_id
            and execution.plan_id == plan.plan_id
            and execution.plan_version == plan.version
            and execution.accepted_review_attempt == accepted_review.attempt,
        ),
        (
            "provenance-execution-source",
            execution.source_fingerprint == dataframe_fingerprint(source),
        ),
        (
            "provenance-execution-result",
            execution.result_fingerprint == dataframe_fingerprint(transformed),
        ),
        (
            "provenance-recorded-shape",
            execution.before_row_count == len(source)
            and execution.after_row_count == len(transformed)
            and execution.before_column_count == source.shape[1]
            and execution.after_column_count == transformed.shape[1],
        ),
        (
            "provenance-execution-success",
            execution.success
            and execution.error_code is None
            and all(
                record.status is OperationExecutionStatus.APPLIED
                for record in execution.operation_records
            ),
        ),
        (
            "provenance-operation-records",
            tuple(record.operation_id for record in execution.operation_records)
            == tuple(operation.operation_id for operation in plan.operations),
        ),
        (
            "provenance-replay-success",
            replay.success and replay_frame is not None and replay.error_code is None,
        ),
        (
            "provenance-replay-result",
            replay_frame is not None
            and dataframe_fingerprint(replay_frame) == dataframe_fingerprint(transformed)
            and execution.result_fingerprint == dataframe_fingerprint(replay_frame),
        ),
        (
            "provenance-replay-records",
            replay.operation_records == execution.operation_records,
        ),
        (
            "provenance-replay-shape",
            replay_frame is not None
            and len(replay_frame) == execution.after_row_count
            and replay_frame.shape[1] == execution.after_column_count,
        ),
        (
            "provenance-empty-plan-no-change",
            bool(plan.operations)
            or execution.result_fingerprint == execution.source_fingerprint,
        ),
    )
    return tuple(
        InvariantResult(
            invariant_id=invariant_id,
            kind=InvariantKind.PROVENANCE,
            status=InvariantStatus.PASS if passed else InvariantStatus.FAIL,
            mandatory=True,
            explanation="Typed execution provenance check.",
        )
        for invariant_id, passed in checks
    )


def _cast_preservation_results(
    plan: TransformationPlan,
    execution: ExecutionResult,
    replay: OperationRun,
) -> tuple[InvariantResult, ...]:
    records = {record.operation_id: record for record in execution.operation_records}
    replay_records = {
        record.operation_id: record for record in replay.operation_records
    }
    results: list[InvariantResult] = []
    for operation in plan.operations:
        if operation.operation_type is not OperationType.CAST_COLUMN:
            continue
        record = records.get(operation.operation_id)
        replay_record = replay_records.get(operation.operation_id)
        introduced = (
            replay_record.introduced_null_count
            if replay_record is not None
            else None
        )
        supplied_matches = (
            record is not None
            and replay_record is not None
            and record.introduced_null_count is not None
            and record.introduced_null_count == replay_record.introduced_null_count
        )
        passed = (
            supplied_matches
            and replay_record is not None
            and replay_record.status is OperationExecutionStatus.APPLIED
            and introduced == 0
        )
        results.append(
            InvariantResult(
                invariant_id=f"cast-preservation-{operation.operation_id}",
                kind=InvariantKind.CAST_VALUE_PRESERVATION,
                status=InvariantStatus.PASS if passed else InvariantStatus.FAIL,
                mandatory=True,
                explanation="Cast must not turn a previously non-null value into null.",
                observed_value=introduced if introduced is not None else "MISSING",
                expected_value=0,
            )
        )
    return tuple(results)



def _operation_preservation_results(
    plan: TransformationPlan,
    execution: ExecutionResult,
    replay: OperationRun,
    source: pd.DataFrame,
    transformed: pd.DataFrame,
) -> tuple[InvariantResult, ...]:
    """Per-operation assertions for the three value-rewriting operations.

    Modelled on _cast_preservation_results: the handler measures, the replay
    measures again, and the invariant passes only when both agree and the
    measurement is acceptable. Every one is mandatory and fails closed, so a
    missing measurement is a failure rather than a pass.
    """

    records = {record.operation_id: record for record in execution.operation_records}
    replay_records = {record.operation_id: record for record in replay.operation_records}
    results: list[InvariantResult] = []

    for operation in plan.operations:
        record = records.get(operation.operation_id)
        replay_record = replay_records.get(operation.operation_id)
        applied = (
            replay_record is not None
            and replay_record.status is OperationExecutionStatus.APPLIED
        )
        rows_held = (
            replay_record is not None
            and replay_record.rows_before == replay_record.rows_after
        )

        if operation.operation_type is OperationType.IMPUTE_MISSING:
            changed = replay_record.changed_non_null_count if replay_record else None
            filled = replay_record.filled_null_count if replay_record else None
            agrees = (
                record is not None
                and replay_record is not None
                and record.changed_non_null_count == replay_record.changed_non_null_count
                and record.filled_null_count == replay_record.filled_null_count
            )
            passed = bool(
                applied
                and rows_held
                and agrees
                and changed == 0
                and filled is not None
                and filled > 0
            )
            results.append(
                InvariantResult(
                    invariant_id=f"impute-preservation-{operation.operation_id}",
                    kind=InvariantKind.IMPUTATION_VALUE_PRESERVATION,
                    status=InvariantStatus.PASS if passed else InvariantStatus.FAIL,
                    mandatory=True,
                    explanation=(
                        "Imputation may only fill nulls: every already-populated "
                        "value must be unchanged and the null count must fall."
                    ),
                    observed_value=changed if changed is not None else "MISSING",
                    expected_value=0,
                )
            )
        elif operation.operation_type is OperationType.NORMALIZE_NUMERIC_TEXT:
            introduced = replay_record.introduced_null_count if replay_record else None
            agrees = (
                record is not None
                and replay_record is not None
                and record.introduced_null_count == replay_record.introduced_null_count
            )
            passed = bool(applied and rows_held and agrees and introduced == 0)
            results.append(
                InvariantResult(
                    invariant_id=f"normalize-no-nulls-{operation.operation_id}",
                    kind=InvariantKind.NUMERIC_TEXT_NO_NULLS,
                    status=InvariantStatus.PASS if passed else InvariantStatus.FAIL,
                    mandatory=True,
                    explanation=(
                        "Stripping numeric noise must not turn any value null."
                    ),
                    observed_value=introduced if introduced is not None else "MISSING",
                    expected_value=0,
                )
            )
        elif operation.operation_type is OperationType.DROP_COLUMN:
            assert isinstance(operation.parameters, DropColumnParameters)

            dropped = tuple(operation.target_columns)
            # Every column any drop in this plan removes. A per-operation
            # invariant is handed the whole-plan frames, so a second drop must
            # not make the first one look like a violation.
            all_dropped = {
                column
                for other in plan.operations
                if other.operation_type is OperationType.DROP_COLUMN
                for column in other.target_columns
            }
            absent = all(column not in transformed.columns for column in dropped)
            # Count arithmetic rather than comparing name sets: a RENAME_COLUMN
            # elsewhere in the plan changes a name without removing a column, and
            # must not read as a column having vanished.
            expected_count = len(source.columns) - len(all_dropped)
            actual_count = len(transformed.columns)
            held = bool(
                applied
                and rows_held
                and absent
                and actual_count == expected_count
            )

            results.append(
                InvariantResult(
                    invariant_id=f"drop-structure-{operation.operation_id}",
                    kind=InvariantKind.DROPPED_COLUMN_STRUCTURE,
                    status=InvariantStatus.PASS if held else InvariantStatus.FAIL,
                    mandatory=True,
                    explanation=(
                        "Dropping columns must preserve the row count, remove every "
                        "named dropped column, and preserve the surviving column structure."
                    ),
                    observed_value=actual_count,
                    expected_value=expected_count,
                )
            )
    return tuple(results)


def _computed_column_results(
    plan: TransformationPlan,
    execution: ExecutionResult,
    replay: OperationRun,
) -> tuple[InvariantResult, ...]:
    """Verify each computation from replayed pre-operation inputs."""

    records = {record.operation_id: record for record in execution.operation_records}
    replay_records = {record.operation_id: record for record in replay.operation_records}
    evidence = {item.operation_id: item for item in replay.computation_evidence}
    results: list[InvariantResult] = []
    for operation in plan.operations:
        if operation.operation_type is not OperationType.COMPUTE_COLUMN:
            continue
        parameters = operation.parameters
        assert isinstance(parameters, ComputeColumnParameters)
        step = evidence.get(operation.operation_id)
        supplied_record = records.get(operation.operation_id)
        replay_record = replay_records.get(operation.operation_id)
        expected = None
        if step is not None:
            if parameters.operator is ComputeOperator.ADD:
                expected = step.left_values + step.right_values
            elif parameters.operator is ComputeOperator.SUBTRACT:
                expected = step.left_values - step.right_values
            elif parameters.operator is ComputeOperator.MULTIPLY:
                expected = step.left_values * step.right_values
            elif parameters.operator is ComputeOperator.DIVIDE:
                if not bool(step.right_values.eq(0).fillna(False).any()):
                    expected = step.left_values / step.right_values
        structure_held = bool(
            step is not None
            and step.rows_before == step.rows_after
            and step.existing_columns_preserved
            and step.after_columns == step.before_columns + (parameters.output_column,)
        )
        passed = bool(
            structure_held
            and expected is not None
            and step is not None
            and expected.equals(step.output_values)
            and supplied_record is not None
            and replay_record is not None
            and supplied_record == replay_record
        )
        results.append(
            InvariantResult(
                invariant_id=f"compute-isolation-{operation.operation_id}",
                kind=InvariantKind.COMPUTED_COLUMN_ISOLATION,
                status=InvariantStatus.PASS if passed else InvariantStatus.FAIL,
                mandatory=True,
                explanation=(
                    "Computed columns must add exactly the approved output, preserve "
                    "all existing columns and rows, and equal closed arithmetic replay."
                ),
            )
        )
    return tuple(results)


def _column_lineage(
    report: DiagnosticReport,
    plan: TransformationPlan,
) -> tuple[ColumnLineage, dict[str, tuple[str, ...]]]:
    lineage = ColumnLineage.from_columns(
        tuple(profile.name for profile in report.column_profiles)
    )
    casts: dict[str, list[str]] = {
        profile.name: [] for profile in report.column_profiles
    }
    for operation in plan.operations:
        if operation.operation_type is OperationType.CAST_COLUMN and operation.target_columns:
            target = operation.target_columns[0]
            original = lineage.original_for_current(target)
            if original is not None:
                casts[original].append(operation.operation_id)
        elif operation.operation_type is OperationType.RENAME_COLUMN and operation.target_columns:
            parameters = operation.parameters
            assert isinstance(parameters, RenameColumnParameters)
            source_name = operation.target_columns[0]
            lineage.rename(source_name, parameters.new_name)
    return lineage, {name: tuple(operation_ids) for name, operation_ids in casts.items()}


def _diagnostic_comparisons(
    report: DiagnosticReport,
    before: pd.DataFrame,
    after: pd.DataFrame,
    plan: TransformationPlan,
    execution: ExecutionResult,
) -> tuple[DiagnosticIssueComparison, ...]:
    comparisons: list[DiagnosticIssueComparison] = []
    lineage, cast_lineage = _column_lineage(report, plan)
    execution_records = {
        record.operation_id: record for record in execution.operation_records
    }
    for issue in report.issues:
        before_value: int | float | str | None = None
        after_value: int | float | str | None = None
        status = DiagnosticResolution.NOT_APPLICABLE
        explanation = "Issue is not measurable with the Phase 1A deterministic checks."
        if issue.kind is DiagnosticIssueKind.NULL_VALUES and issue.affected_columns:
            original_column = issue.affected_columns[0]
            current_column = lineage.current_for_original(original_column)
            if original_column in before and current_column and current_column in after:
                before_value = int(before[original_column].isna().sum())
                after_value = int(after[current_column].isna().sum())
            else:
                explanation = "Column lineage is unavailable for deterministic comparison."
        elif issue.kind is DiagnosticIssueKind.DUPLICATE_ROWS:
            before_value = int(before.duplicated().sum())
            after_value = int(after.duplicated().sum())
        elif issue.kind is DiagnosticIssueKind.DUPLICATE_KEYS:
            before_value = _duplicate_key_rows(before, issue.affected_columns)
            current_keys = lineage.currents_for_original(issue.affected_columns)
            after_value = (
                _duplicate_key_rows(after, current_keys)
                if current_keys is not None
                else None
            )
        elif issue.kind is DiagnosticIssueKind.CANDIDATE_TYPE_CONVERSION and issue.affected_columns:
            original_column = issue.affected_columns[0]
            current_column = lineage.current_for_original(original_column)
            before_value = str(before[original_column].dtype) if original_column in before else "MISSING"
            after_value = (
                str(after[current_column].dtype)
                if current_column and current_column in after
                else "MISSING"
            )
            cast_records = tuple(
                execution_records[operation_id]
                for operation_id in cast_lineage.get(original_column, ())
                if operation_id in execution_records
            )
            if current_column is None or current_column not in after:
                status = DiagnosticResolution.NOT_APPLICABLE
                explanation = "Column lineage is unavailable for deterministic comparison."
            elif any(record.introduced_null_count for record in cast_records):
                status = DiagnosticResolution.WORSENED
                explanation = "The cast introduced null values from non-null source values."
            elif before_value == after_value or not cast_records:
                status = DiagnosticResolution.UNCHANGED
                explanation = "Candidate conversion remains unresolved."
            else:
                status = DiagnosticResolution.RESOLVED
                explanation = "An approved cast changed dtype without introducing nulls."
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            if after_value == 0 and before_value > 0:
                status = DiagnosticResolution.RESOLVED
            elif after_value < before_value:
                status = DiagnosticResolution.IMPROVED
            elif after_value == before_value:
                status = DiagnosticResolution.UNCHANGED
            else:
                status = DiagnosticResolution.WORSENED
            explanation = "Deterministic before/after metric comparison."
        comparisons.append(
            DiagnosticIssueComparison(
                issue_id=issue.issue_id,
                status=status,
                before_value=before_value,
                after_value=after_value,
                explanation=explanation,
            )
        )
    return tuple(comparisons)


def run_quality_assurance(
    source: pd.DataFrame,
    transformed: pd.DataFrame,
    execution: ExecutionResult,
    report: DiagnosticReport,
    context: PlanningContext,
    intent: UserIntent,
    plan: TransformationPlan,
    validation: PlanValidationResult,
    accepted_review: AcceptedReviewEvidence,
    approval: HumanApproval | None,
    *,
    user_invariants: tuple[QualityInvariant, ...] = (),
) -> QAReport:
    before_rows = len(source)
    after_rows = len(transformed)
    row_loss_pct = ((before_rows - after_rows) / before_rows * 100) if before_rows else 0.0
    row_loss_pct = max(0.0, float(row_loss_pct))
    before_columns = tuple(str(column) for column in source.columns)
    after_columns = tuple(str(column) for column in transformed.columns)
    common = set(before_columns).intersection(after_columns)
    dtype_changes = tuple(
        DTypeChange(
            column=column,
            before=str(source[column].dtype),
            after=str(transformed[column].dtype),
        )
        for column in before_columns
        if column in common and str(source[column].dtype) != str(transformed[column].dtype)
    )
    null_changes = tuple(
        NullCountChange(
            column=column,
            before=int(source[column].isna().sum()),
            after=int(transformed[column].isna().sum()),
        )
        for column in before_columns
        if column in common
        and int(source[column].isna().sum()) != int(transformed[column].isna().sum())
    )
    renamed = tuple(
        f"{operation.target_columns[0]}->{operation.parameters.new_name}"
        for operation in plan.operations
        if operation.operation_type is OperationType.RENAME_COLUMN
        and isinstance(operation.parameters, RenameColumnParameters)
    )
    final_lineage, _ = _column_lineage(report, plan)
    current_selected_keys = (
        final_lineage.currents_for_original(intent.selected_key_columns)
        if intent.selected_key_columns
        else ()
    )
    invariants = _default_invariants(
        intent,
        plan,
        current_selected_keys or intent.selected_key_columns,
    ) + user_invariants
    ordinary_results = tuple(
        _evaluate_invariant(invariant, source, transformed, row_loss_pct)
        for invariant in invariants
    )
    replay = run_allowlisted_plan(source, plan)
    invariant_results = (
        _provenance_results(
            source,
            transformed,
            execution,
            report,
            context,
            plan,
            validation,
            accepted_review,
            approval,
            replay,
            intent,
        )
        + _cast_preservation_results(plan, execution, replay)
        + _operation_preservation_results(
            plan, execution, replay, source, transformed
        )
        + _computed_column_results(plan, execution, replay)
        + ordinary_results
    )
    execution_failures = tuple(
        record.operation_id
        for record in execution.operation_records
        if record.status is OperationExecutionStatus.FAILED
    )
    comparisons = _diagnostic_comparisons(
        report,
        source,
        transformed,
        plan,
        execution,
    )
    if not execution.success or execution_failures or any(
        item.mandatory and item.status is InvariantStatus.FAIL
        for item in invariant_results
    ):
        status = QAStatus.FAIL
    elif any(item.status is InvariantStatus.WARN for item in invariant_results) or any(
        item.status is DiagnosticResolution.WORSENED for item in comparisons
    ):
        status = QAStatus.WARN
    else:
        status = QAStatus.PASS
    material = f"{execution.execution_id}|{status.value}|{row_loss_pct:.8f}"
    qa_hash = sha256(material.encode("utf-8")).hexdigest()
    return QAReport(
        qa_report_id=f"qa-{qa_hash[:16]}",
        dataset_id=execution.dataset_id,
        plan_id=execution.plan_id,
        status=status,
        before_row_count=before_rows,
        after_row_count=after_rows,
        before_column_count=source.shape[1],
        after_column_count=transformed.shape[1],
        row_loss_pct=row_loss_pct,
        added_columns=tuple(column for column in after_columns if column not in before_columns),
        removed_columns=tuple(column for column in before_columns if column not in after_columns),
        renamed_columns=renamed,
        dtype_changes=dtype_changes,
        null_count_changes=null_changes,
        duplicate_rows_before=int(source.duplicated().sum()),
        duplicate_rows_after=int(transformed.duplicated().sum()),
        duplicate_keys_before=_duplicate_key_rows(source, intent.selected_key_columns),
        duplicate_keys_after=_duplicate_key_rows(
            transformed,
            current_selected_keys or intent.selected_key_columns,
        ),
        invariant_results=invariant_results,
        diagnostic_comparisons=comparisons,
        execution_failures=execution_failures,
    )
