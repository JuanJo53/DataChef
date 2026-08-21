from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from datachef.application.request_compiler import compile_objective
from datachef.contracts import (
    ComputeColumnParameters,
    ComputeOperator,
    DownstreamUse,
    InvariantKind,
    InvariantStatus,
    OperationType,
    QAStatus,
    RiskLevel,
    TransformationOperation,
    UserIntent,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.planning import create_transformation_plan, validate_plan
from datachef.privacy import build_planning_context
from datachef.qa import run_quality_assurance
from datachef.transform.executor import execute_approved_plan
from support import accepted_review, human_approval


def _operation(
    *,
    left: str = "quantity",
    right: str = "unit_price",
    output: str = "total",
    operator: ComputeOperator = ComputeOperator.MULTIPLY,
) -> TransformationOperation:
    return TransformationOperation(
        operation_id="op-compute-total",
        operation_type=OperationType.COMPUTE_COLUMN,
        target_columns=(left, right),
        parameters=ComputeColumnParameters(
            left_column=left,
            right_column=right,
            output_column=output,
            operator=operator,
        ),
        user_requirement_ids=("request-compute-total",),
        rationale="The user requested a closed arithmetic derivation.",
        expected_effect="Add exactly one deterministic numeric column.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )


def _facts(source: pd.DataFrame, operation=None, *, intent=None):
    operation = operation or _operation()
    intent = intent or UserIntent(
        intent_id="intent-compute",
        downstream_use=DownstreamUse.ML,
        acceptable_row_loss_pct=0,
    )
    report = diagnose_raw_dataframe(source)
    context = build_planning_context(report, intent, ())
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(operation,),
        summary="Compute one approved column.",
    )
    validation = validate_plan(context, plan)
    return intent, report, context, plan, validation


def test_compute_contract_is_closed_and_forbids_expression_fields() -> None:
    parameters = ComputeColumnParameters(
        left_column="quantity",
        right_column="unit_price",
        output_column="total",
        operator=ComputeOperator.MULTIPLY,
    )
    assert parameters.operator is ComputeOperator.MULTIPLY
    with pytest.raises(ValidationError):
        ComputeColumnParameters(
            left_column="quantity",
            right_column="unit_price",
            output_column="total",
            operator=ComputeOperator.MULTIPLY,
            expression="quantity * unit_price",
        )
    with pytest.raises(ValidationError):
        ComputeColumnParameters(
            left_column="quantity",
            right_column="unit_price",
            output_column=" ",
            operator=ComputeOperator.MULTIPLY,
        )


@pytest.mark.parametrize(
    ("source", "operation", "code"),
    [
        (
            pd.DataFrame({"quantity": [1], "unit_price": [2]}),
            _operation(left="missing"),
            "MISSING_COLUMN",
        ),
        (
            pd.DataFrame({"quantity": ["one"], "unit_price": [2]}),
            _operation(),
            "COMPUTE_NON_NUMERIC_INPUT",
        ),
        (
            pd.DataFrame({"quantity": [1], "unit_price": [2], "total": [2]}),
            _operation(),
            "COMPUTE_OUTPUT_COLLISION",
        ),
        (
            pd.DataFrame({"quantity": [1], "unit_price": [0]}),
            _operation(operator=ComputeOperator.DIVIDE),
            "COMPUTE_ZERO_DENOMINATOR",
        ),
    ],
)
def test_compute_validation_fails_closed(source, operation, code) -> None:
    *_, validation = _facts(source, operation)
    assert not validation.valid
    assert code in {item.code for item in validation.findings}


def test_compute_executes_on_copy_and_passes_dedicated_qa_invariant() -> None:
    source = pd.DataFrame(
        {
            "quantity": [2, 3],
            "unit_price": [4.5, 10.0],
            "protected_note": ["x", "y"],
        }
    )
    original = source.copy(deep=True)
    intent = UserIntent(
        intent_id="intent-compute-protected",
        downstream_use=DownstreamUse.ML,
        protected_columns=("protected_note",),
        required_columns=("quantity", "unit_price"),
        acceptable_row_loss_pct=0,
    )
    intent, report, context, plan, validation = _facts(source, intent=intent)
    assert validation.valid
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
    assert bundle.dataframe["total"].tolist() == [9.0, 30.0]
    assert list(bundle.dataframe.columns) == [
        "quantity", "unit_price", "protected_note", "total"
    ]
    assert_frame_equal(source, original)
    computed = [
        item for item in qa.invariant_results
        if item.kind is InvariantKind.COMPUTED_COLUMN_ISOLATION
    ]
    assert len(computed) == 1
    assert computed[0].status is InvariantStatus.PASS


def test_compute_tampered_execution_metadata_fails_provenance_and_invariant() -> None:
    source = pd.DataFrame({"quantity": [2, 3], "unit_price": [4.5, 10.0]})
    intent, report, context, plan, validation = _facts(source)
    review = accepted_review(plan, validation)
    approval = human_approval(plan)
    bundle = execute_approved_plan(
        source, report, context, intent, plan, validation, review, approval,
        expected_review_attempt=review.attempt,
    )
    record = bundle.result.operation_records[0]
    forged = bundle.result.model_copy(
        update={
            "operation_records": (
                record.model_copy(update={"affected_cell_count": 0}),
            )
        }
    )

    qa = run_quality_assurance(
        source, bundle.dataframe, forged, report, context, intent, plan,
        validation, review, approval,
    )

    assert qa.status is QAStatus.FAIL
    assert any(
        item.kind is InvariantKind.PROVENANCE
        and item.status is InvariantStatus.FAIL
        for item in qa.invariant_results
    )
    assert any(
        item.kind is InvariantKind.COMPUTED_COLUMN_ISOLATION
        and item.status is InvariantStatus.FAIL
        for item in qa.invariant_results
    )


@pytest.mark.parametrize(
    ("text", "operator"),
    [
        ("compute total as quantity times unit_price", ComputeOperator.MULTIPLY),
        ("calculate total as quantity plus unit_price", ComputeOperator.ADD),
        ("derive margin as quantity minus unit_price", ComputeOperator.SUBTRACT),
        ("create ratio as quantity divided by unit_price", ComputeOperator.DIVIDE),
    ],
)
def test_compute_request_compiler_supports_only_closed_binary_phrases(text, operator) -> None:
    source = pd.DataFrame({"quantity": [2], "unit_price": [4]})
    result = compile_objective(text, source, diagnose_raw_dataframe(source))

    assert result.findings == ()
    assert len(result.requests) == 1
    request = result.requests[0]
    assert request.operation_type is OperationType.COMPUTE_COLUMN
    assert request.target_columns == ("quantity", "unit_price")
    assert request.parameters.operator is operator


def test_ambiguous_compute_expression_is_visible_and_unexecutable() -> None:
    source = pd.DataFrame({"quantity": [2], "unit_price": [4]})
    result = compile_objective(
        "compute total using a clever formula",
        source,
        diagnose_raw_dataframe(source),
    )
    assert result.requests == ()
    assert any(
        item.code == "COMPUTE_COLUMN_UNSUPPORTED" and item.blocking
        for item in result.findings
    )


def test_compute_agent_tool_has_closed_arguments_and_builds_valid_plan() -> None:
    from datachef.agents.tools import (
        ComputeColumnArgs,
        PlanDraft,
        apply_operation_args,
        build_operation_specs,
    )

    source = pd.DataFrame({"quantity": [2], "unit_price": [4]})
    intent, report, context, *_ = _facts(source)
    draft = PlanDraft(context=context)
    args = ComputeColumnArgs(
        left_column="quantity",
        right_column="unit_price",
        output_column="total",
        operator=ComputeOperator.MULTIPLY,
        diagnostic_issue_ids=[context.diagnostic_report.issues[0].issue_id]
        if context.diagnostic_report.issues else [],
        rationale="Compute the requested total.",
        expected_effect="Add total.",
    )
    # Tool-grounded compute may use a typed request instead of a diagnosis.
    if not args.diagnostic_issue_ids:
        args = args.model_copy(update={"user_requirement_ids": ["request-total"]})
    result = apply_operation_args(draft, "propose_compute_column", args)

    assert result["accepted"] is True
    assert any(name == "propose_compute_column" for name, _, _ in build_operation_specs())
    assert validate_plan(context, draft.build_plan()).valid
    with pytest.raises(ValidationError):
        ComputeColumnArgs(**args.model_dump(), expression="quantity * unit_price")
