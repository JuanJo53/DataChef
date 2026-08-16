from __future__ import annotations

import pandas as pd

from datachef.contracts import (
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    DownstreamUse,
    DropDuplicateRowsParameters,
    KeepPolicy,
    NormalizeMissingTokensParameters,
    OperationType,
    RenameColumnParameters,
    RiskLevel,
    TransformationOperation,
    TrimWhitespaceParameters,
    UserIntent,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.planning import create_transformation_plan, validate_plan
from datachef.privacy import build_planning_context
from datachef.transform.executor import execute_approved_plan
from support import accepted_review, human_approval


def _operation(
    operation_id: str,
    operation_type: OperationType,
    targets: tuple[str, ...],
    parameters,
    *,
    material: bool,
) -> TransformationOperation:
    return TransformationOperation(
        operation_id=operation_id,
        operation_type=operation_type,
        target_columns=targets,
        parameters=parameters,
        user_requirement_ids=(f"intent.{operation_id}",),
        rationale="Synthetic catalogue contract test.",
        expected_effect="Apply exactly the configured allow-listed operation.",
        risk=RiskLevel.MEDIUM if material else RiskLevel.LOW,
        requires_human_approval=material,
    )


def test_column_operation_catalogue_executes_declared_parameters() -> None:
    source = pd.DataFrame(
        {
            "label": [" A ", " B "],
            "missing_token": ["N/A", "present"],
            "numeric_text": ["10", "20"],
            "boolean_text": ["yes", "no"],
            "date_text": ["2026-01-01", "2026-01-02"],
            "number_as_string": [1, 2],
        }
    )
    intent = UserIntent(
        intent_id="intent-catalogue",
        downstream_use=DownstreamUse.ANALYSIS,
        acceptable_row_loss_pct=0.0,
    )
    operations = (
        _operation(
            "trim",
            OperationType.TRIM_WHITESPACE,
            ("label",),
            TrimWhitespaceParameters(),
            material=False,
        ),
        _operation(
            "missing",
            OperationType.NORMALIZE_MISSING_TOKENS,
            ("missing_token",),
            NormalizeMissingTokensParameters(tokens=("N/A",)),
            material=True,
        ),
        _operation(
            "numeric",
            OperationType.CAST_COLUMN,
            ("numeric_text",),
            CastColumnParameters(
                target_type=CastTarget.NUMERIC,
                errors=CastErrorPolicy.RAISE,
            ),
            material=True,
        ),
        _operation(
            "boolean",
            OperationType.CAST_COLUMN,
            ("boolean_text",),
            CastColumnParameters(target_type=CastTarget.BOOLEAN),
            material=True,
        ),
        _operation(
            "datetime",
            OperationType.CAST_COLUMN,
            ("date_text",),
            CastColumnParameters(
                target_type=CastTarget.DATETIME,
                datetime_format="%Y-%m-%d",
            ),
            material=True,
        ),
        _operation(
            "string",
            OperationType.CAST_COLUMN,
            ("number_as_string",),
            CastColumnParameters(target_type=CastTarget.STRING),
            material=True,
        ),
        _operation(
            "rename",
            OperationType.RENAME_COLUMN,
            ("label",),
            RenameColumnParameters(new_name="clean_label"),
            material=True,
        ),
    )
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=operations,
        summary="Exercise each column-oriented Phase 1A operation.",
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
        expected_review_attempt=review.attempt,
    )

    assert validation.valid is True
    assert bundle.dataframe is not None
    result = bundle.dataframe
    assert result["clean_label"].tolist() == ["A", "B"]
    assert pd.isna(result.loc[0, "missing_token"])
    assert pd.api.types.is_numeric_dtype(result["numeric_text"])
    assert pd.api.types.is_bool_dtype(result["boolean_text"])
    assert pd.api.types.is_datetime64_any_dtype(result["date_text"])
    assert pd.api.types.is_string_dtype(result["number_as_string"])


def test_complete_row_deduplication_uses_explicit_keep_policy() -> None:
    source = pd.DataFrame({"category": ["A", "A", "B"], "measure": [1, 1, 2]})
    intent = UserIntent(
        intent_id="intent-row-dedup",
        acceptable_row_loss_pct=40.0,
    )
    operation = _operation(
        "drop-complete-duplicates",
        OperationType.DROP_DUPLICATE_ROWS,
        (),
        DropDuplicateRowsParameters(keep=KeepPolicy.FIRST),
        material=True,
    )
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(operation,),
        summary="Remove complete duplicate rows.",
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
        expected_review_attempt=review.attempt,
    )

    assert validation.valid is True
    assert bundle.dataframe is not None
    assert len(bundle.dataframe) == 2
