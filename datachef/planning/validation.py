"""Deterministic, all-findings transformation-plan validation."""

from __future__ import annotations

from datachef.contracts import (
    CastColumnParameters,
    DeduplicateByKeysParameters,
    DiagnosticIssueKind,
    ImputeMissingParameters,
    ImputeStrategy,
    OperationType,
    PIIHandling,
    PlanValidationFinding,
    PlanValidationResult,
    PlanningContext,
    RenameColumnParameters,
    RowLossEstimate,
    Severity,
    TransformationPlan,
)
from datachef.transform.operations import OPERATION_CATALOGUE
from datachef.planning.plan import expected_plan_id
from datachef.planning.lineage import ColumnLineage



# Dtype families, read off the provider-safe column schema. Prefix tests rather
# than an exhaustive list of pandas dtype spellings, so a widening dtype name
# does not silently fall through to "not numeric".
_NUMERIC_DTYPE_PREFIXES = ("int", "uint", "float", "number", "decimal")
_TEXT_DTYPES = frozenset({"str", "string", "object"})


def _is_numeric_dtype(dtype: str) -> bool:
    return dtype.lower().startswith(_NUMERIC_DTYPE_PREFIXES)


def _is_text_dtype(dtype: str) -> bool:
    return dtype.lower() in _TEXT_DTYPES


def _declared_dtypes(context: PlanningContext) -> dict[str, str]:
    return {column.name: column.dtype for column in context.column_schema}


def _null_counts(context: PlanningContext) -> dict[str, int]:
    """Return measured per-column null counts from the privacy-safe context."""

    return context.null_counts

def _constant_matches_dtype(value: object, dtype: str) -> bool:
    if _is_numeric_dtype(dtype):
        # bool is an int subclass in Python; a Boolean is not a number here.
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if _is_text_dtype(dtype):
        return isinstance(value, str)
    if dtype.lower().startswith("bool"):
        return isinstance(value, bool)
    return False


def validate_plan(
    context: PlanningContext,
    plan: TransformationPlan,
) -> PlanValidationResult:
    findings: list[PlanValidationFinding] = []
    row_loss_estimates: list[RowLossEstimate] = []
    cumulative_estimated_rows = 0

    def add(code: str, message: str, operation_id: str | None = None) -> None:
        findings.append(
            PlanValidationFinding(
                code=code,
                message=message,
                operation_id=operation_id,
                severity=Severity.HIGH,
            )
        )

    if plan.dataset_id != context.dataset_identity.dataset_id:
        add("DATASET_ID_MISMATCH", "Plan dataset ID does not match planning context.")
    if plan.dataset_fingerprint != context.dataset_identity.fingerprint:
        add("DATASET_FINGERPRINT_MISMATCH", "Plan fingerprint does not match context.")
    if plan.plan_id != expected_plan_id(plan):
        add("PLAN_ID_MISMATCH", "Plan identity does not match its declarative contents.")
    if context.user_intent.pii_handling is not PIIHandling.NONE:
        add(
            "UNSUPPORTED_PII_OPERATION",
            "Phase 1A cannot satisfy the requested PII masking or removal operation.",
        )
    operation_ids = [operation.operation_id for operation in plan.operations]
    for operation_id in sorted({item for item in operation_ids if operation_ids.count(item) > 1}):
        add("DUPLICATE_OPERATION_ID", "Operation IDs must be unique.", operation_id)

    lineage = ColumnLineage.from_columns(
        tuple(column.name for column in context.column_schema)
    )
    protected = set(context.user_intent.protected_columns)
    required = set(context.user_intent.required_columns)
    declared_dtypes = _declared_dtypes(context)
    null_counts = _null_counts(context)
    issue_ids = {issue.issue_id for issue in context.diagnostic_report.issues}

    for operation in plan.operations:
        definition = OPERATION_CATALOGUE.get(operation.operation_type)
        if definition is None:
            add("UNSUPPORTED_OPERATION", "Operation is not allow-listed.", operation.operation_id)
            continue
        if not isinstance(operation.parameters, definition.parameter_type):
            add("PARAMETER_TYPE_MISMATCH", "Parameters do not match operation type.", operation.operation_id)
        current_columns = lineage.current_columns
        missing = [column for column in operation.target_columns if column not in current_columns]
        for column in missing:
            add("MISSING_COLUMN", f"Referenced column does not exist at this step: {column}.", operation.operation_id)
        for issue_id in operation.diagnostic_issue_ids:
            if issue_id not in issue_ids:
                add("UNKNOWN_DIAGNOSTIC_ISSUE", "Operation references an unknown diagnostic issue.", operation.operation_id)
        if protected.intersection(operation.target_columns):
            add("PROTECTED_COLUMN", "Protected columns cannot be transformed.", operation.operation_id)
        if set(operation.target_columns).intersection(
            context.privacy_manifest.aliased_columns
        ):
            add(
                "ALIASED_COLUMN_NOT_EXECUTABLE",
                "Privacy-aliased columns are not executable in Phase 1A.",
                operation.operation_id,
            )
        if definition.material and not operation.requires_human_approval:
            add("MATERIAL_APPROVAL_REQUIRED", "Material operation must require human approval.", operation.operation_id)
        if (
            operation.operation_type is not OperationType.DROP_DUPLICATE_ROWS
            and not operation.target_columns
        ):
            add(
                "EMPTY_TARGET_COLUMNS",
                "Column-oriented operation requires at least one target column.",
                operation.operation_id,
            )

        if operation.operation_type is OperationType.RENAME_COLUMN:
            parameters = operation.parameters
            assert isinstance(parameters, RenameColumnParameters)
            if len(operation.target_columns) != 1:
                add("RENAME_TARGET_COUNT", "Rename requires exactly one source column.", operation.operation_id)
            elif parameters.new_name in current_columns and parameters.new_name != operation.target_columns[0]:
                add("RENAME_COLLISION", "Rename target already exists.", operation.operation_id)
            elif operation.target_columns[0] in required:
                add("REQUIRED_COLUMN_RENAME", "Required columns cannot be renamed.", operation.operation_id)
            elif not missing:
                lineage.rename(operation.target_columns[0], parameters.new_name)
        elif operation.operation_type is OperationType.CAST_COLUMN:
            if len(operation.target_columns) != 1:
                add("CAST_TARGET_COUNT", "Cast requires exactly one target column.", operation.operation_id)
            if not isinstance(operation.parameters, CastColumnParameters):
                add("UNSUPPORTED_CAST", "Cast target is unsupported.", operation.operation_id)
        elif operation.operation_type is OperationType.DEDUPLICATE_BY_KEYS:
            parameters = operation.parameters
            assert isinstance(parameters, DeduplicateByKeysParameters)
            if not parameters.keys:
                add("EMPTY_KEYS", "Deduplication keys cannot be empty.", operation.operation_id)
            if tuple(operation.target_columns) != tuple(parameters.keys):
                add("KEY_TARGET_MISMATCH", "Target columns must equal approved deduplication keys.", operation.operation_id)
            for key in parameters.keys:
                if key not in current_columns:
                    add("INVALID_KEY", f"Deduplication key does not exist: {key}.", operation.operation_id)
            original_keys = lineage.originals_for_current(parameters.keys)
            metric = next(
                (
                    item
                    for item in context.diagnostic_report.key_duplicate_metrics
                    if original_keys is not None
                    and item.key_columns == original_keys
                ),
                None,
            )
            if metric and metric.null_key_row_count:
                add(
                    "NULL_KEYS_UNSAFE",
                    "Key deduplication is not allowed when any approved key is null.",
                    operation.operation_id,
                )

        elif operation.operation_type is OperationType.DROP_COLUMN:
            if operation.target_columns:
                dropped_required = sorted(required.intersection(operation.target_columns))
                for column in dropped_required:
                    add(
                        "REQUIRED_COLUMN_DROP",
                        f"Required column cannot be dropped: {column}.",
                        operation.operation_id,
                    )
                surviving = [
                    column
                    for column in current_columns
                    if column not in set(operation.target_columns)
                ]
                if not surviving:
                    add(
                        "DROP_ALL_COLUMNS",
                        "A plan may not drop every column.",
                        operation.operation_id,
                    )
                if not missing and not dropped_required and surviving:
                    for column in operation.target_columns:
                        lineage.drop(column)
        elif operation.operation_type is OperationType.IMPUTE_MISSING:
            parameters = operation.parameters
            assert isinstance(parameters, ImputeMissingParameters)
            if len(operation.target_columns) != 1:
                add(
                    "IMPUTE_TARGET_COUNT",
                    "Imputation requires exactly one target column.",
                    operation.operation_id,
                )
            elif not missing:
                column = operation.target_columns[0]
                original = lineage.original_for_current(column)
                dtype = declared_dtypes.get(original or column, "")
                nulls = null_counts.get(original or column, 0)
                if not nulls:
                    add(
                        "IMPUTE_NO_MISSING_VALUES",
                        "Column has no missing values, so there is nothing to impute.",
                        operation.operation_id,
                    )
                if parameters.strategy in (ImputeStrategy.MEAN, ImputeStrategy.MEDIAN):
                    if not _is_numeric_dtype(dtype):
                        add(
                            "IMPUTE_STRATEGY_DTYPE_MISMATCH",
                            "Mean and median imputation require a numeric column.",
                            operation.operation_id,
                        )
                elif parameters.strategy is ImputeStrategy.MODE:
                    # A column that is entirely null has no most-frequent value.
                    if nulls and nulls >= context.dataset_identity.row_count:
                        add(
                            "IMPUTE_NO_MODE",
                            "Column is entirely null, so no mode exists.",
                            operation.operation_id,
                        )
                elif parameters.strategy is ImputeStrategy.CONSTANT:
                    if parameters.constant_value is None:
                        add(
                            "IMPUTE_CONSTANT_TYPE_MISMATCH",
                            "Constant imputation requires a constant value.",
                            operation.operation_id,
                        )
                    elif not _constant_matches_dtype(parameters.constant_value, dtype):
                        add(
                            "IMPUTE_CONSTANT_TYPE_MISMATCH",
                            "Constant value type does not match the column dtype.",
                            operation.operation_id,
                        )
        elif operation.operation_type is OperationType.NORMALIZE_NUMERIC_TEXT:
            if not missing:
                for column in operation.target_columns:
                    original = lineage.original_for_current(column)
                    dtype = declared_dtypes.get(original or column, "")
                    if not _is_text_dtype(dtype):
                        add(
                            "NORMALIZE_NON_TEXT_COLUMN",
                            f"Numeric-text normalization requires a text column: {column}.",
                            operation.operation_id,
                        )

        if definition.may_drop_rows and context.dataset_identity.row_count:
            estimated_rows = 0
            if operation.operation_type is OperationType.DROP_DUPLICATE_ROWS:
                estimated_rows = context.diagnostic_report.duplicate_row_count
            elif operation.operation_type is OperationType.DEDUPLICATE_BY_KEYS:
                parameters = operation.parameters
                assert isinstance(parameters, DeduplicateByKeysParameters)
                original_keys = lineage.originals_for_current(parameters.keys)
                metric = next(
                    (
                        item
                        for item in context.diagnostic_report.key_duplicate_metrics
                        if original_keys is not None
                        and item.key_columns == original_keys
                    ),
                    None,
                )
                estimated_rows = metric.duplicate_row_count if metric else 0
            estimated_pct = estimated_rows / context.dataset_identity.row_count * 100
            estimated_pct = min(100.0, max(0.0, estimated_pct))
            row_loss_estimates.append(
                RowLossEstimate(
                    operation_id=operation.operation_id,
                    estimated_rows=estimated_rows,
                    estimated_pct=estimated_pct,
                )
            )
            cumulative_estimated_rows = min(
                context.dataset_identity.row_count,
                cumulative_estimated_rows + estimated_rows,
            )
            if estimated_pct > context.user_intent.acceptable_row_loss_pct:
                add(
                    "ROW_LOSS_THRESHOLD",
                    "Estimated row loss exceeds the user's approved threshold.",
                    operation.operation_id,
                )

    # The provider-safe context intentionally has no raw frame. Phase 1A therefore
    # adds independently measured row-loss estimates and caps them at source size.
    # This may reject overlapping removals conservatively, but cannot underestimate
    # the sum of known removal risks.
    row_count = context.dataset_identity.row_count
    cumulative_pct = (
        min(100.0, cumulative_estimated_rows / row_count * 100)
        if row_count
        else 0.0
    )
    if cumulative_pct > context.user_intent.acceptable_row_loss_pct:
        add(
            "CUMULATIVE_ROW_LOSS_THRESHOLD",
            "Conservative cumulative row-loss estimate exceeds the approved threshold.",
        )

    return PlanValidationResult(
        plan_id=plan.plan_id,
        valid=not findings,
        findings=tuple(findings),
        row_loss_estimates=tuple(row_loss_estimates),
        cumulative_estimated_row_loss_pct=cumulative_pct,
    )
