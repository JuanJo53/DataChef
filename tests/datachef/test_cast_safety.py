from __future__ import annotations

import pandas as pd
import pytest

from datachef.contracts import (
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    OperationType,
    QAStatus,
    RenameColumnParameters,
    RiskLevel,
    TransformationOperation,
    UserIntent,
    WorkflowStage,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.planning import (
    RuleBasedReviewer,
    SequencePlanner,
    create_transformation_plan,
    validate_plan,
)
from datachef.privacy import build_planning_context
from datachef.qa import run_quality_assurance
from datachef.transform.executor import execute_approved_plan
from datachef.workflow import execute_workflow, prepare_workflow
from support import accepted_review, human_approval


def _cast(operation_id: str, column: str, target: CastTarget) -> TransformationOperation:
    return TransformationOperation(
        operation_id=operation_id,
        operation_type=OperationType.CAST_COLUMN,
        target_columns=(column,),
        parameters=CastColumnParameters(
            target_type=target,
            errors=CastErrorPolicy.COERCE,
        ),
        user_requirement_ids=(f"requirement-{operation_id}",),
        rationale="Synthetic cast-safety test.",
        expected_effect="Convert only values accepted by the declared target type.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )


def _run(source: pd.DataFrame, operations: tuple[TransformationOperation, ...]):
    intent = UserIntent(intent_id="cast-safety", acceptable_row_loss_pct=0)
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=operations,
        summary="Cast-safety plan.",
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
    return bundle, qa, context, plan


@pytest.mark.parametrize(
    ("values", "target"),
    [
        (["1", "2", "bad"], CastTarget.NUMERIC),
        (["yes", "no", "unexpected"], CastTarget.BOOLEAN),
        (["2026-01-01", "invalid-date"], CastTarget.DATETIME),
    ],
)
def test_coercive_cast_loss_is_a_mandatory_qa_failure(values, target) -> None:
    bundle, qa, _, _ = _run(
        pd.DataFrame({"value": values}),
        (_cast("cast-value", "value", target),),
    )

    assert bundle.result.operation_records[0].introduced_null_count == 1
    assert qa.status is QAStatus.FAIL
    assert any(
        result.invariant_id == "cast-preservation-cast-value"
        and result.status.value == "FAIL"
        for result in qa.invariant_results
    )


@pytest.mark.parametrize(
    "values",
    [(["1", None, "2"]), (["1", "2", "3"])],
)
def test_valid_coercion_does_not_misclassify_existing_nulls(values) -> None:
    bundle, qa, _, _ = _run(
        pd.DataFrame({"value": values}),
        (_cast("cast-value", "value", CastTarget.NUMERIC),),
    )

    assert bundle.result.operation_records[0].introduced_null_count == 0
    assert qa.status is QAStatus.PASS


def test_multiple_casts_and_renamed_target_retain_operation_evidence() -> None:
    source = pd.DataFrame(
        {
            "amount": ["1", "2", None],
            "observed_on": ["2026-01-01", "2026-01-02", None],
        }
    )
    rename = TransformationOperation(
        operation_id="rename-amount",
        operation_type=OperationType.RENAME_COLUMN,
        target_columns=("amount",),
        parameters=RenameColumnParameters(new_name="amount_clean"),
        user_requirement_ids=("rename",),
        rationale="Trace the converted column through a rename.",
        expected_effect="Retain the converted values under a new label.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )
    operations = (
        _cast("cast-amount", "amount", CastTarget.NUMERIC),
        rename,
        _cast("cast-date", "observed_on", CastTarget.DATETIME),
    )

    bundle, qa, _, _ = _run(source, operations)

    assert bundle.dataframe is not None
    assert "amount_clean" in bundle.dataframe
    cast_results = {
        result.invariant_id: result.status.value
        for result in qa.invariant_results
        if result.invariant_id.startswith("cast-preservation-")
    }
    assert cast_results == {
        "cast-preservation-cast-amount": "PASS",
        "cast-preservation-cast-date": "PASS",
    }


def test_cast_loss_blocks_workflow_gold_promotion() -> None:
    source = pd.DataFrame({"value": ["1", "2", "bad"]})
    intent = UserIntent(intent_id="cast-workflow", acceptable_row_loss_pct=0)
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(_cast("cast-value", "value", CastTarget.NUMERIC),),
        summary="Coercive workflow regression.",
    )
    runtime = prepare_workflow(
        source,
        intent,
        SequencePlanner((plan,)),
        RuleBasedReviewer(),
    )

    completed = execute_workflow(runtime, human_approval(plan))

    assert completed.state.stage is WorkflowStage.QA_FAILED
    assert completed.gold_dataframe is None
