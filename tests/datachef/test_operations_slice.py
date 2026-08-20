"""DROP_COLUMN, IMPUTE_MISSING and NORMALIZE_NUMERIC_TEXT, end to end.

Each operation is covered at every layer it touches: the handler, the
deterministic refusals, the QA invariant on both a legitimate run and a
violation, the rendered pipeline against the real executor, and the agent tool
boundary. The chained normalize-then-cast run on currency text is the demo case
and is driven through the whole trusted flow to a QA PASS.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pandas as pd
import pytest
from pydantic import ValidationError

from datachef.application import (
    ArtifactSet,
    CsvParserOptions,
    DataChefController,
    UploadFormat,
    UploadRequest,
)
from datachef.application.pipeline_render import (
    render_pipeline_bytes,
    render_pipeline_script,
)
from datachef.contracts import (
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    DiagnosticIssueKind,
    DownstreamUse,
    DropColumnParameters,
    ExecutionResult,
    HumanDecision,
    ImputeMissingParameters,
    ImputeStrategy,
    InvariantKind,
    InvariantStatus,
    NormalizeNumericTextParameters,
    OperationExecutionRecord,
    OperationExecutionStatus,
    OperationType,
    QAStatus,
    RiskLevel,
    TransformationOperation,
    TrimWhitespaceParameters,
    UserIntent,
    WorkflowStage,
)
from datachef.diagnostics import dataframe_fingerprint, diagnose_raw_dataframe
from datachef.planning import validate_plan
from datachef.planning.plan import create_transformation_plan
from datachef.privacy import build_column_alias_map, build_planning_context
from datachef.qa.service import _operation_preservation_results
from datachef.transform.operations import OPERATION_CATALOGUE
from datachef.transform.runner import OperationRun, run_allowlisted_plan

REPO_ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)

# The demo dataset. Deliberately free of parenthesised negatives: nothing in the
# repository reads "(1,234.00)" as negative, so normalization leaves them alone
# and a following cast would fail the run closed rather than flip a sign.
DEMO_CSV = (
    b"order_id,amount_text,qty,junk\n"
    b"1,\xc2\xa0$1,304.20 ,1,x\n"
).replace(b"\xc2\xa0", b"")  # keep the fixture plain ASCII

CURRENCY_CSV = (
    b'order_id,amount_text,qty,junk\n'
    b'1,"$1,304.20",1.0,x\n'
    b'2,"$27.00",,y\n'
    b'3,"$1,000",3.0,z\n'
    b'4,"$3.50",,w\n'
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "amount_text": ["$1,304.20", " $27.00", "$1,000", "$3.50"],
            "qty": [1.0, None, 3.0, None],
            "label": ["a", None, "c", "c"],
            "junk": ["x", "y", "z", "w"],
        }
    )


def _operation(operation_id, operation_type, target_columns, parameters):
    return TransformationOperation(
        operation_id=operation_id,
        operation_type=operation_type,
        target_columns=target_columns,
        parameters=parameters,
        diagnostic_issue_ids=("issue-slice",),
        rationale="Reported by the diagnosis.",
        expected_effect="Prepares the column.",
        risk=RiskLevel.LOW,
        requires_human_approval=True,
    )


def _plan(*operations):
    return create_transformation_plan(
        dataset_id="dataset-slice",
        dataset_fingerprint="b" * 64,
        version=1,
        operations=tuple(operations),
        summary="Operations slice plan.",
    )


# ---------------------------------------------------------------------------
# Catalogue registration
# ---------------------------------------------------------------------------


def test_the_three_operations_are_registered_with_the_expected_flags() -> None:
    assert set(OPERATION_CATALOGUE) == set(OperationType)
    assert len(OPERATION_CATALOGUE) == 9
    for operation_type in (
        OperationType.DROP_COLUMN,
        OperationType.IMPUTE_MISSING,
        OperationType.NORMALIZE_NUMERIC_TEXT,
    ):
        definition = OPERATION_CATALOGUE[operation_type]
        assert definition.material is True
        assert definition.may_drop_rows is False
        assert definition.handler is not None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def test_normalize_numeric_text_strips_only_the_named_noise_classes() -> None:
    plan = _plan(
        _operation(
            "op-001-normalize_numeric_text",
            OperationType.NORMALIZE_NUMERIC_TEXT,
            ("amount_text",),
            NormalizeNumericTextParameters(),
        )
    )

    run = run_allowlisted_plan(_frame(), plan)

    assert run.success
    assert list(run.dataframe["amount_text"]) == ["1304.20", "27.00", "1000", "3.50"]
    # It prepares; it does not cast.
    assert not pd.api.types.is_numeric_dtype(run.dataframe["amount_text"].dtype)
    assert run.operation_records[0].introduced_null_count == 0


def test_normalize_numeric_text_switches_are_independent() -> None:
    frame = pd.DataFrame({"a": [" $1,2 "], "keep": [1]})
    for parameters, expected in (
        (NormalizeNumericTextParameters(strip_currency_symbols=False), "$12"),
        (NormalizeNumericTextParameters(strip_thousands_separators=False), "1,2"),
        (NormalizeNumericTextParameters(strip_whitespace=False), " 12 "),
    ):
        plan = _plan(
            _operation(
                "op-001-normalize_numeric_text",
                OperationType.NORMALIZE_NUMERIC_TEXT,
                ("a",),
                parameters,
            )
        )
        run = run_allowlisted_plan(frame.copy(deep=True), plan)
        assert run.success
        assert run.dataframe["a"].iloc[0] == expected, parameters


@pytest.mark.parametrize(
    ("strategy", "constant", "column", "expected"),
    (
        (ImputeStrategy.MEAN, None, "qty", [1.0, 2.0, 3.0, 2.0]),
        (ImputeStrategy.MEDIAN, None, "qty", [1.0, 2.0, 3.0, 2.0]),
        (ImputeStrategy.CONSTANT, 0.0, "qty", [1.0, 0.0, 3.0, 0.0]),
        (ImputeStrategy.MODE, None, "label", ["a", "c", "c", "c"]),
    ),
)
def test_impute_missing_fills_by_strategy(strategy, constant, column, expected) -> None:
    plan = _plan(
        _operation(
            "op-001-impute_missing",
            OperationType.IMPUTE_MISSING,
            (column,),
            ImputeMissingParameters(strategy=strategy, constant_value=constant),
        )
    )

    run = run_allowlisted_plan(_frame(), plan)

    assert run.success
    assert list(run.dataframe[column]) == expected
    record = run.operation_records[0]
    # The measurement the QA invariant depends on.
    assert record.changed_non_null_count == 0
    assert record.filled_null_count == int(_frame()[column].isna().sum())
    assert record.filled_null_count > 0
    assert record.rows_before == record.rows_after == 4


def test_drop_column_removes_exactly_the_named_columns() -> None:
    plan = _plan(
        _operation(
            "op-001-drop_column",
            OperationType.DROP_COLUMN,
            ("junk",),
            DropColumnParameters(),
        )
    )
    source = _frame()

    run = run_allowlisted_plan(source, plan)

    assert run.success
    assert "junk" not in run.dataframe.columns
    assert list(run.dataframe.columns) == ["order_id", "amount_text", "qty", "label"]
    assert len(run.dataframe) == len(source)
    for column in run.dataframe.columns:
        assert run.dataframe[column].equals(source[column])


@pytest.mark.parametrize(
    ("label", "operation"),
    (
        (
            "impute with nothing missing",
            (
                OperationType.IMPUTE_MISSING,
                ("order_id",),
                ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
            ),
        ),
        (
            "drop an absent column",
            (OperationType.DROP_COLUMN, ("nope",), DropColumnParameters()),
        ),
    ),
)
def test_handlers_fail_closed_rather_than_pretending(label, operation) -> None:
    del label
    operation_type, columns, parameters = operation
    plan = _plan(_operation("op-001-x", operation_type, columns, parameters))

    run = run_allowlisted_plan(_frame(), plan)

    assert not run.success
    assert run.dataframe is None
    assert run.error_code == "PLAN_EXECUTION_ABORTED"
    assert run.operation_records[-1].status is OperationExecutionStatus.FAILED


# ---------------------------------------------------------------------------
# The new diagnostic signal
# ---------------------------------------------------------------------------


def _diagnose(frame: pd.DataFrame):
    return diagnose_raw_dataframe(frame, selected_key_columns=("order_id",))


def test_numeric_text_noise_is_detected_and_suggests_normalization() -> None:
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "money": ["$1,304.20", "$27.00", "$1,000", "$3.50"],
        }
    )

    report = _diagnose(frame)

    noise = [
        issue
        for issue in report.issues
        if issue.kind is DiagnosticIssueKind.CANDIDATE_NUMERIC_TEXT_NOISE
    ]
    assert [issue.affected_columns for issue in noise] == [("money",)]
    assert noise[0].suggested_operation is OperationType.NORMALIZE_NUMERIC_TEXT


def test_the_two_candidate_kinds_are_mutually_exclusive() -> None:
    """A column already plain-numeric stays a plain cast candidate."""

    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "plain": ["10", "20", "30", "40"],
            "money": ["$1,304.20", "$27.00", "$1,000", "$3.50"],
            "words": ["alpha", "beta", "gamma", "delta"],
        }
    )

    report = _diagnose(frame)

    by_kind = {}
    for issue in report.issues:
        by_kind.setdefault(issue.kind, set()).update(issue.affected_columns)
    assert by_kind.get(DiagnosticIssueKind.CANDIDATE_TYPE_CONVERSION) == {"plain"}
    assert by_kind.get(DiagnosticIssueKind.CANDIDATE_NUMERIC_TEXT_NOISE) == {"money"}
    # Prose is not numeric text under either rule.
    assert "words" not in by_kind.get(
        DiagnosticIssueKind.CANDIDATE_NUMERIC_TEXT_NOISE, set()
    )
    assert "words" not in by_kind.get(
        DiagnosticIssueKind.CANDIDATE_TYPE_CONVERSION, set()
    )


# ---------------------------------------------------------------------------
# Validation refusals
# ---------------------------------------------------------------------------


def _context(frame: pd.DataFrame, *, required=(), protected=()):
    intent = UserIntent(
        intent_id="intent-slice",
        user_goal="Prepare.",
        selected_key_columns=("order_id",),
        required_columns=required,
        protected_columns=protected,
        acceptable_row_loss_pct=50,
    )
    report = _diagnose(frame)
    return build_planning_context(
        report,
        intent,
        (),
        column_alias_map=build_column_alias_map(report, intent),
    )


def _validate(context, *operations):
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=tuple(operations),
        summary="Validation probe.",
    )
    return validate_plan(context, plan)


def _issue_for(context, kind, column):
    return next(
        issue.issue_id
        for issue in context.diagnostic_report.issues
        if issue.kind is kind and column in issue.affected_columns
    )


def test_a_legitimate_plan_for_all_three_operations_validates() -> None:
    frame = _frame()
    context = _context(frame)
    noise = _issue_for(
        context, DiagnosticIssueKind.CANDIDATE_NUMERIC_TEXT_NOISE, "amount_text"
    )
    nulls = _issue_for(context, DiagnosticIssueKind.NULL_VALUES, "qty")

    result = _validate(
        context,
        _operation(
            "op-001-normalize_numeric_text",
            OperationType.NORMALIZE_NUMERIC_TEXT,
            ("amount_text",),
            NormalizeNumericTextParameters(),
        )._replace_issues(noise)
        if hasattr(TransformationOperation, "_replace_issues")
        else TransformationOperation(
            operation_id="op-001-normalize_numeric_text",
            operation_type=OperationType.NORMALIZE_NUMERIC_TEXT,
            target_columns=("amount_text",),
            parameters=NormalizeNumericTextParameters(),
            diagnostic_issue_ids=(noise,),
            rationale="r",
            expected_effect="e",
            risk=RiskLevel.LOW,
            requires_human_approval=True,
        ),
        TransformationOperation(
            operation_id="op-002-impute_missing",
            operation_type=OperationType.IMPUTE_MISSING,
            target_columns=("qty",),
            parameters=ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
            diagnostic_issue_ids=(nulls,),
            rationale="r",
            expected_effect="e",
            risk=RiskLevel.LOW,
            requires_human_approval=True,
        ),
        TransformationOperation(
            operation_id="op-003-drop_column",
            operation_type=OperationType.DROP_COLUMN,
            target_columns=("junk",),
            parameters=DropColumnParameters(),
            diagnostic_issue_ids=(noise,),
            rationale="r",
            expected_effect="e",
            risk=RiskLevel.LOW,
            requires_human_approval=True,
        ),
    )

    assert result.valid, [finding.code for finding in result.findings]


def _cited(context, column="qty", kind=DiagnosticIssueKind.NULL_VALUES):
    return _issue_for(context, kind, column)


def _probe(context, operation_type, columns, parameters, issue_id):
    return _validate(
        context,
        TransformationOperation(
            operation_id="op-001-probe",
            operation_type=operation_type,
            target_columns=columns,
            parameters=parameters,
            diagnostic_issue_ids=(issue_id,),
            rationale="r",
            expected_effect="e",
            risk=RiskLevel.LOW,
            requires_human_approval=True,
        ),
    )


def test_validation_refuses_dropping_a_required_column() -> None:
    frame = _frame()
    context = _context(frame, required=("junk",))
    result = _probe(
        context,
        OperationType.DROP_COLUMN,
        ("junk",),
        DropColumnParameters(),
        _cited(context),
    )

    assert not result.valid
    assert "REQUIRED_COLUMN_DROP" in [finding.code for finding in result.findings]


def test_validation_refuses_dropping_every_column() -> None:
    frame = _frame()
    context = _context(frame)
    result = _probe(
        context,
        OperationType.DROP_COLUMN,
        tuple(frame.columns),
        DropColumnParameters(),
        _cited(context),
    )

    assert not result.valid
    assert "DROP_ALL_COLUMNS" in [finding.code for finding in result.findings]


def test_validation_refuses_an_operation_on_an_already_dropped_column() -> None:
    """No new code needed: the existing step-aware MISSING_COLUMN fires."""

    frame = _frame()
    context = _context(frame)
    issue = _cited(context)
    result = _validate(
        context,
        TransformationOperation(
            operation_id="op-001-drop_column",
            operation_type=OperationType.DROP_COLUMN,
            target_columns=("junk",),
            parameters=DropColumnParameters(),
            diagnostic_issue_ids=(issue,),
            rationale="r",
            expected_effect="e",
            risk=RiskLevel.LOW,
            requires_human_approval=True,
        ),
        TransformationOperation(
            operation_id="op-002-trim_whitespace",
            operation_type=OperationType.TRIM_WHITESPACE,
            target_columns=("junk",),
            parameters=TrimWhitespaceParameters(),
            diagnostic_issue_ids=(issue,),
            rationale="r",
            expected_effect="e",
            risk=RiskLevel.LOW,
            requires_human_approval=False,
        ),
    )

    assert not result.valid
    assert "MISSING_COLUMN" in [finding.code for finding in result.findings]


def test_transforming_a_column_dropped_later_is_allowed() -> None:
    """Wasteful is not incorrect, so it is not blocked.

    validate_plan sets ``valid = not findings``, so every finding blocks by
    construction; there is no non-blocking channel to report this in. Rather
    than invent one, or block a plan that is merely inelegant, this stays valid
    and QA still verifies the result.
    """

    frame = _frame()
    context = _context(frame)
    issue = _cited(context)
    result = _validate(
        context,
        TransformationOperation(
            operation_id="op-001-trim_whitespace",
            operation_type=OperationType.TRIM_WHITESPACE,
            target_columns=("junk",),
            parameters=TrimWhitespaceParameters(),
            diagnostic_issue_ids=(issue,),
            rationale="r",
            expected_effect="e",
            risk=RiskLevel.LOW,
            requires_human_approval=False,
        ),
        TransformationOperation(
            operation_id="op-002-drop_column",
            operation_type=OperationType.DROP_COLUMN,
            target_columns=("junk",),
            parameters=DropColumnParameters(),
            diagnostic_issue_ids=(issue,),
            rationale="r",
            expected_effect="e",
            risk=RiskLevel.LOW,
            requires_human_approval=True,
        ),
    )

    assert result.valid, [finding.code for finding in result.findings]


@pytest.mark.parametrize(
    ("label", "column", "parameters", "code"),
    (
        (
            "mean on text",
            "label",
            ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
            "IMPUTE_STRATEGY_DTYPE_MISMATCH",
        ),
        (
            "median on text",
            "label",
            ImputeMissingParameters(strategy=ImputeStrategy.MEDIAN),
            "IMPUTE_STRATEGY_DTYPE_MISMATCH",
        ),
        (
            "constant of the wrong type",
            "qty",
            ImputeMissingParameters(
                strategy=ImputeStrategy.CONSTANT, constant_value="zero"
            ),
            "IMPUTE_CONSTANT_TYPE_MISMATCH",
        ),
        (
            "constant missing entirely",
            "qty",
            ImputeMissingParameters(strategy=ImputeStrategy.CONSTANT),
            "IMPUTE_CONSTANT_TYPE_MISMATCH",
        ),
        (
            "nothing to impute",
            "order_id",
            ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
            "IMPUTE_NO_MISSING_VALUES",
        ),
    ),
)
def test_validation_refuses_bad_imputations(label, column, parameters, code) -> None:
    del label
    context = _context(_frame())
    result = _probe(
        context,
        OperationType.IMPUTE_MISSING,
        (column,),
        parameters,
        _cited(context),
    )

    assert not result.valid
    assert code in [finding.code for finding in result.findings]


def test_validation_refuses_mode_on_an_entirely_null_column() -> None:
    frame = pd.DataFrame(
        {"order_id": [1, 2, 3], "empty": [None, None, None]}
    )
    context = _context(frame)
    result = _probe(
        context,
        OperationType.IMPUTE_MISSING,
        ("empty",),
        ImputeMissingParameters(strategy=ImputeStrategy.MODE),
        _cited(context, column="empty"),
    )

    assert not result.valid
    assert "IMPUTE_NO_MODE" in [finding.code for finding in result.findings]


def test_validation_refuses_normalizing_a_non_text_column() -> None:
    context = _context(_frame())
    result = _probe(
        context,
        OperationType.NORMALIZE_NUMERIC_TEXT,
        ("order_id",),
        NormalizeNumericTextParameters(),
        _cited(context),
    )

    assert not result.valid
    assert "NORMALIZE_NON_TEXT_COLUMN" in [
        finding.code for finding in result.findings
    ]


def test_the_generic_guards_still_cover_the_new_operations() -> None:
    """Protected and aliased columns need no per-operation code."""

    context = _context(_frame(), protected=("junk",))
    result = _probe(
        context,
        OperationType.DROP_COLUMN,
        ("junk",),
        DropColumnParameters(),
        _cited(context),
    )

    assert not result.valid
    # Privacy aliases a protected column, so the refusal arrives as any of the
    # three generic guards. The point is that no per-operation code was needed.
    codes = {finding.code for finding in result.findings}
    assert codes & {
        "PROTECTED_COLUMN",
        "ALIASED_COLUMN_NOT_EXECUTABLE",
        "MISSING_COLUMN",
    }, codes


# ---------------------------------------------------------------------------
# QA invariants
# ---------------------------------------------------------------------------


def _records(**overrides):
    payload = {
        "operation_id": "op-001-impute_missing",
        "status": OperationExecutionStatus.APPLIED,
        "rows_before": 4,
        "rows_after": 4,
        "affected_cell_count": 2,
        "changed_non_null_count": 0,
        "filled_null_count": 2,
    }
    payload.update(overrides)
    return OperationExecutionRecord(**payload)


def _execution(records, after_row_count: int = 4):
    return ExecutionResult(
        execution_id="execution-slice",
        dataset_id="dataset-slice",
        plan_id="plan-slice",
        plan_version=1,
        accepted_review_attempt=1,
        success=True,
        source_fingerprint="c" * 64,
        result_fingerprint="d" * 64,
        before_row_count=4,
        after_row_count=after_row_count,
        before_column_count=5,
        after_column_count=5,
        operation_records=records,
    )


def _impute_plan():
    return _plan(
        _operation(
            "op-001-impute_missing",
            OperationType.IMPUTE_MISSING,
            ("qty",),
            ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
        )
    )


def test_the_imputation_invariant_passes_a_legitimate_run() -> None:
    record = _records()
    results = _operation_preservation_results(
        _impute_plan(),
        _execution((record,)),
        OperationRun(success=True, dataframe=_frame(), operation_records=(record,)),
        _frame(),
        _frame(),
    )

    assert [item.kind for item in results] == [
        InvariantKind.IMPUTATION_VALUE_PRESERVATION
    ]
    assert results[0].status is InvariantStatus.PASS
    assert results[0].mandatory is True


@pytest.mark.parametrize(
    ("label", "overrides"),
    (
        ("a previously non-null value was changed", {"changed_non_null_count": 1}),
        ("no null was actually filled", {"filled_null_count": 0}),
        ("the measurement is missing", {"changed_non_null_count": None}),
    ),
)
def test_the_imputation_invariant_fails_a_violation(label, overrides) -> None:
    """This is the point of the invariant: imputation may only fill nulls."""

    del label
    record = _records(**overrides)
    results = _operation_preservation_results(
        _impute_plan(),
        _execution((record,)),
        OperationRun(success=True, dataframe=_frame(), operation_records=(record,)),
        _frame(),
        _frame(),
    )

    assert results[0].status is InvariantStatus.FAIL
    assert results[0].mandatory is True


def test_the_imputation_invariant_fails_when_the_row_count_moved() -> None:
    record = _records(rows_after=3)

    results = _operation_preservation_results(
        _impute_plan(),
        _execution((record,), after_row_count=3),
        OperationRun(success=True, dataframe=_frame(), operation_records=(record,)),
        _frame(),
        _frame(),
    )

    assert results[0].status is InvariantStatus.FAIL


def test_the_imputation_invariant_fails_when_execution_and_replay_disagree() -> None:
    executed = _records(changed_non_null_count=0)
    replayed = _records(changed_non_null_count=1)

    results = _operation_preservation_results(
        _impute_plan(),
        _execution((executed,)),
        OperationRun(success=True, dataframe=_frame(), operation_records=(replayed,)),
        _frame(),
        _frame(),
    )

    assert results[0].status is InvariantStatus.FAIL


def test_a_changed_non_null_value_fails_qa_through_the_real_handler(monkeypatch) -> None:
    """The load-bearing case, driven through the handler rather than a stub.

    A tampered handler that rewrites an already-populated cell must be caught,
    not merely reported: the invariant is mandatory, so QA fails closed.
    """

    import datachef.transform.operations as operations

    original = operations._impute_missing

    def tampering(dataframe, operation):
        effect = original(dataframe, operation)
        column = operation.target_columns[0]
        # Overwrite a value that was never missing.
        effect.dataframe.loc[effect.dataframe.index[0], column] = 999.0
        return operations.OperationEffect(
            dataframe=effect.dataframe,
            affected_cell_count=effect.affected_cell_count,
            changed_non_null_count=1,
            filled_null_count=effect.filled_null_count,
        )

    monkeypatch.setitem(
        operations.OPERATION_CATALOGUE,
        OperationType.IMPUTE_MISSING,
        operations.OperationDefinition(
            OperationType.IMPUTE_MISSING,
            ImputeMissingParameters,
            tampering,
            material=True,
            may_drop_rows=False,
        ),
    )

    plan = _impute_plan()
    run = run_allowlisted_plan(_frame(), plan)
    assert run.success
    record = run.operation_records[0]
    assert record.changed_non_null_count == 1

    results = _operation_preservation_results(
        plan,
        _execution((record,)),
        run,
        _frame(),
        run.dataframe,
    )

    assert results[0].kind is InvariantKind.IMPUTATION_VALUE_PRESERVATION
    assert results[0].status is InvariantStatus.FAIL


def test_the_normalization_invariant_fails_if_a_value_becomes_null() -> None:
    plan = _plan(
        _operation(
            "op-001-normalize_numeric_text",
            OperationType.NORMALIZE_NUMERIC_TEXT,
            ("amount_text",),
            NormalizeNumericTextParameters(),
        )
    )
    clean = OperationExecutionRecord(
        operation_id="op-001-normalize_numeric_text",
        status=OperationExecutionStatus.APPLIED,
        rows_before=4,
        rows_after=4,
        affected_cell_count=4,
        introduced_null_count=0,
    )
    nulled = clean.model_copy(update={"introduced_null_count": 1})

    passing = _operation_preservation_results(
        plan, _execution((clean,)),
        OperationRun(success=True, dataframe=_frame(), operation_records=(clean,)),
        _frame(), _frame(),
    )
    failing = _operation_preservation_results(
        plan, _execution((nulled,)),
        OperationRun(success=True, dataframe=_frame(), operation_records=(nulled,)),
        _frame(), _frame(),
    )

    assert passing[0].kind is InvariantKind.NUMERIC_TEXT_NO_NULLS
    assert passing[0].status is InvariantStatus.PASS
    assert failing[0].status is InvariantStatus.FAIL


def test_the_drop_structure_invariant_ignores_survivor_value_changes() -> None:
    plan = _plan(
        _operation(
            "op-001-drop_column",
            OperationType.DROP_COLUMN,
            ("junk",),
            DropColumnParameters(),
        )
    )
    record = OperationExecutionRecord(
        operation_id="op-001-drop_column",
        status=OperationExecutionStatus.APPLIED,
        rows_before=4,
        rows_after=4,
        affected_cell_count=4,
    )
    source = _frame()
    good = source.drop(columns=["junk"])
    tampered = good.copy(deep=True)
    tampered.loc[tampered.index[0], "qty"] = 42.0

    passing = _operation_preservation_results(
        plan,
        _execution((record,)),
        OperationRun(success=True, dataframe=good, operation_records=(record,)),
        source,
        good,
    )

    tampered_result = _operation_preservation_results(
        plan,
        _execution((record,)),
        OperationRun(success=True, dataframe=tampered, operation_records=(record,)),
        source,
        tampered,
    )

    assert passing[0].kind is InvariantKind.DROPPED_COLUMN_STRUCTURE
    assert passing[0].status is InvariantStatus.PASS

    assert tampered_result[0].kind is InvariantKind.DROPPED_COLUMN_STRUCTURE
    assert tampered_result[0].status is InvariantStatus.PASS

def test_drop_structure_fails_if_dropped_column_remains() -> None:
    plan = _plan(
        _operation(
            "op-001-drop_column",
            OperationType.DROP_COLUMN,
            ("junk",),
            DropColumnParameters(),
        )
    )

    record = OperationExecutionRecord(
        operation_id="op-001-drop_column",
        status=OperationExecutionStatus.APPLIED,
        rows_before=4,
        rows_after=4,
        affected_cell_count=4,
    )

    source = _frame()

    result = _operation_preservation_results(
        plan,
        _execution((record,)),
        OperationRun(
            success=True,
            dataframe=source.copy(deep=True),
            operation_records=(record,),
        ),
        source,
        source.copy(deep=True),
    )

    assert result[0].kind is InvariantKind.DROPPED_COLUMN_STRUCTURE
    assert result[0].status is InvariantStatus.FAIL
    assert result[0].mandatory is True

def test_drop_structure_fails_if_survivor_vanishes() -> None:
    plan = _plan(
        _operation(
            "op-001-drop_column",
            OperationType.DROP_COLUMN,
            ("junk",),
            DropColumnParameters(),
        )
    )

    record = OperationExecutionRecord(
        operation_id="op-001-drop_column",
        status=OperationExecutionStatus.APPLIED,
        rows_before=4,
        rows_after=4,
        affected_cell_count=4,
    )

    source = _frame()
    invalid = source.drop(columns=["junk", "qty"])

    result = _operation_preservation_results(
        plan,
        _execution((record,)),
        OperationRun(
            success=True,
            dataframe=invalid,
            operation_records=(record,),
        ),
        source,
        invalid,
    )

    assert result[0].kind is InvariantKind.DROPPED_COLUMN_STRUCTURE
    assert result[0].status is InvariantStatus.FAIL
    assert result[0].mandatory is True

def test_drop_structure_fails_if_row_count_changes() -> None:
    plan = _plan(
        _operation(
            "op-001-drop_column",
            OperationType.DROP_COLUMN,
            ("junk",),
            DropColumnParameters(),
        )
    )

    record = OperationExecutionRecord(
        operation_id="op-001-drop_column",
        status=OperationExecutionStatus.APPLIED,
        rows_before=4,
        rows_after=3,
        affected_cell_count=4,
    )

    source = _frame()
    invalid = source.drop(columns=["junk"]).iloc[:3].copy()

    result = _operation_preservation_results(
        plan,
        _execution((record,), after_row_count=3),
        OperationRun(
            success=True,
            dataframe=invalid,
            operation_records=(record,),
        ),
        source,
        invalid,
    )

    assert result[0].kind is InvariantKind.DROPPED_COLUMN_STRUCTURE
    assert result[0].status is InvariantStatus.FAIL
    assert result[0].mandatory is True

# ---------------------------------------------------------------------------
# The demo case, through the whole trusted flow
# ---------------------------------------------------------------------------


def test_normalize_then_cast_reaches_qa_pass_on_currency_text() -> None:
    """The demo: "$1,304.20" text becomes numeric gold through the real flow."""

    controller = DataChefController(clock=lambda: NOW)
    assert controller.load_upload(
        UploadRequest(
            content=CURRENCY_CSV,
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(encoding="utf-8-sig"),
        )
    ).changed
    assert controller.diagnose().changed

    session = controller.session
    report = session.display_diagnostic_report
    noise = [
        issue
        for issue in report.issues
        if issue.kind is DiagnosticIssueKind.CANDIDATE_NUMERIC_TEXT_NOISE
    ]
    assert [issue.affected_columns for issue in noise] == [("amount_text",)]

    controller.submit_intent(
        UserIntent(
            intent_id="intent-demo",
            user_goal="Make the amounts numeric.",
            downstream_use=DownstreamUse.ANALYSIS,
            selected_key_columns=("order_id",),
            acceptable_row_loss_pct=50,
        ),
        (),
    )

    assert controller.prepare_plan(command_id="plan").code == "PLAN_AWAITING_APPROVAL"
    plan = controller.session.workflow_runtime.state.transformation_plan
    # The deterministic planner proposes the chain unprompted, in order.
    chained = [
        operation.operation_type
        for operation in plan.operations
        if "amount_text" in operation.target_columns
    ]
    assert chained == [
        OperationType.NORMALIZE_NUMERIC_TEXT,
        OperationType.CAST_COLUMN,
    ], chained

    assert controller.record_human_decision(
        HumanDecision.APPROVE, command_id="approve"
    ).changed
    assert controller.execute_current_plan(command_id="execute").changed
    state = controller.session.workflow_runtime.state
    assert state.stage is WorkflowStage.QA_PASSED
    assert state.qa_report.status is QAStatus.PASS
    gold = controller.session.workflow_runtime.gold_dataframe
    assert list(gold["amount_text"]) == [1304.20, 27.00, 1000.0, 3.50]
    assert pd.api.types.is_numeric_dtype(gold["amount_text"].dtype)
    # PASS-only gold, so the full seven-artifact bundle exists.
    bundle = controller.build_artifacts()
    assert isinstance(bundle, ArtifactSet)
    assert len(bundle.artifacts()) == 7


def test_the_chained_plan_produces_numeric_gold_and_a_full_bundle() -> None:
    """Explicit chain, since it is the plan a human would approve."""

    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "amount_text": ["$1,304.20", "$27.00", "$1,000", "$3.50"],
        }
    )
    plan = _plan(
        _operation(
            "op-001-normalize_numeric_text",
            OperationType.NORMALIZE_NUMERIC_TEXT,
            ("amount_text",),
            NormalizeNumericTextParameters(),
        ),
        _operation(
            "op-002-cast_column",
            OperationType.CAST_COLUMN,
            ("amount_text",),
            CastColumnParameters(
                target_type=CastTarget.NUMERIC,
                errors=CastErrorPolicy.RAISE,
            ),
        ),
    )

    run = run_allowlisted_plan(frame, plan)

    assert run.success, run.error_code
    assert list(run.dataframe["amount_text"]) == [1304.20, 27.00, 1000.0, 3.50]
    assert pd.api.types.is_numeric_dtype(run.dataframe["amount_text"].dtype)
    # A RAISE cast succeeded, so nothing was silently coerced to null.
    assert all(
        record.introduced_null_count in (0, None)
        for record in run.operation_records
    )


def test_a_parenthesised_negative_fails_closed_rather_than_flipping_sign() -> None:
    """The documented limit: normalization leaves it, the cast refuses."""

    frame = pd.DataFrame(
        {"order_id": [1, 2], "amount_text": ["$1,000.00", "($5.00)"]}
    )
    plan = _plan(
        _operation(
            "op-001-normalize_numeric_text",
            OperationType.NORMALIZE_NUMERIC_TEXT,
            ("amount_text",),
            NormalizeNumericTextParameters(),
        ),
        _operation(
            "op-002-cast_column",
            OperationType.CAST_COLUMN,
            ("amount_text",),
            CastColumnParameters(
                target_type=CastTarget.NUMERIC,
                errors=CastErrorPolicy.RAISE,
            ),
        ),
    )

    run = run_allowlisted_plan(frame, plan)

    assert not run.success
    assert run.error_code == "PLAN_EXECUTION_ABORTED"


# ---------------------------------------------------------------------------
# The rendered pipeline
# ---------------------------------------------------------------------------

DRIVER = """
import runpy
import sys
import pandas as pd
from datachef.diagnostics import dataframe_fingerprint

namespace = runpy.run_path(sys.argv[1])
frame = pd.read_pickle(sys.argv[2])
result, steps = namespace["apply_plan"](frame)
print(dataframe_fingerprint(result))
"""


def _fingerprint_in_subprocess(plan, frame, tmp_path):
    script = tmp_path / "rendered.py"
    script.write_bytes(render_pipeline_bytes(plan))
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    frame_path = tmp_path / "frame.pkl"
    with open(frame_path, "wb") as handle:
        pickle.dump(frame, handle)
    return subprocess.run(
        [sys.executable, "-B", str(driver), str(script), str(frame_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "DATACHEF_OFFLINE": "true",
        },
    )


RENDER_CASES = (
    (
        "DROP_COLUMN",
        (OperationType.DROP_COLUMN, ("junk",), DropColumnParameters()),
    ),
    (
        "IMPUTE_MEAN",
        (
            OperationType.IMPUTE_MISSING,
            ("qty",),
            ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
        ),
    ),
    (
        "IMPUTE_MEDIAN",
        (
            OperationType.IMPUTE_MISSING,
            ("qty",),
            ImputeMissingParameters(strategy=ImputeStrategy.MEDIAN),
        ),
    ),
    (
        "IMPUTE_MODE",
        (
            OperationType.IMPUTE_MISSING,
            ("label",),
            ImputeMissingParameters(strategy=ImputeStrategy.MODE),
        ),
    ),
    (
        "IMPUTE_CONSTANT",
        (
            OperationType.IMPUTE_MISSING,
            ("qty",),
            ImputeMissingParameters(
                strategy=ImputeStrategy.CONSTANT, constant_value=0.0
            ),
        ),
    ),
    (
        "NORMALIZE_NUMERIC_TEXT",
        (
            OperationType.NORMALIZE_NUMERIC_TEXT,
            ("amount_text",),
            NormalizeNumericTextParameters(),
        ),
    ),
)


@pytest.mark.parametrize(
    ("name", "spec"),
    RENDER_CASES,
    ids=[name for name, _ in RENDER_CASES],
)
def test_the_rendered_script_matches_the_executor_for_each_new_operation(
    name,
    spec,
    tmp_path,
) -> None:
    del name
    operation_type, columns, parameters = spec
    plan = _plan(_operation("op-001-x", operation_type, columns, parameters))
    expected = run_allowlisted_plan(_frame(), plan)
    assert expected.success and expected.dataframe is not None

    ast.parse(render_pipeline_script(plan))
    completed = _fingerprint_in_subprocess(plan, _frame(), tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == dataframe_fingerprint(expected.dataframe)


def test_the_rendered_script_matches_the_executor_for_a_chained_plan(tmp_path) -> None:
    plan = _plan(
        _operation(
            "op-001-normalize_numeric_text",
            OperationType.NORMALIZE_NUMERIC_TEXT,
            ("amount_text",),
            NormalizeNumericTextParameters(),
        ),
        _operation(
            "op-002-cast_column",
            OperationType.CAST_COLUMN,
            ("amount_text",),
            CastColumnParameters(
                target_type=CastTarget.NUMERIC, errors=CastErrorPolicy.RAISE
            ),
        ),
        _operation(
            "op-003-impute_missing",
            OperationType.IMPUTE_MISSING,
            ("qty",),
            ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
        ),
        _operation(
            "op-004-drop_column",
            OperationType.DROP_COLUMN,
            ("junk",),
            DropColumnParameters(),
        ),
    )
    expected = run_allowlisted_plan(_frame(), plan)
    assert expected.success and expected.dataframe is not None

    completed = _fingerprint_in_subprocess(plan, _frame(), tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == dataframe_fingerprint(expected.dataframe)


def test_the_new_operations_keep_the_script_standalone_and_sanitized() -> None:
    plan = _plan(
        _operation(
            "op-001-normalize_numeric_text",
            OperationType.NORMALIZE_NUMERIC_TEXT,
            ("amt\nlines",),
            NormalizeNumericTextParameters(),
        ),
        _operation(
            "op-002-drop_column",
            OperationType.DROP_COLUMN,
            ("junk\rmore",),
            DropColumnParameters(),
        ),
        _operation(
            "op-003-impute_missing",
            OperationType.IMPUTE_MISSING,
            ("qty x",),
            ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
        ),
    )

    text = render_pipeline_script(plan)

    ast.parse(text)
    assert "import datachef" not in text and "from datachef" not in text
    imports = [
        line.strip()
        for line in text.splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert imports == ["import sys", "import pandas as pd"]
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            for terminator in ("\n", "\r", " ", " "):
                assert terminator not in line


def test_the_render_stays_byte_identical_for_the_new_operations() -> None:
    plan = _plan(
        _operation(
            "op-001-impute_missing",
            OperationType.IMPUTE_MISSING,
            ("qty",),
            ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
        ),
        _operation(
            "op-002-drop_column",
            OperationType.DROP_COLUMN,
            ("junk",),
            DropColumnParameters(),
        ),
    )

    assert render_pipeline_bytes(plan) == render_pipeline_bytes(plan)
    # No set literal may reach the text: its iteration order follows the seed.
    import re

    literals = re.findall(r"=\s*\{[^}]*\}", render_pipeline_script(plan))
    assert [item for item in literals if " for " not in item and ":" not in item] == []

@pytest.mark.parametrize(
    ("label", "operations"),
    (
        (
            "impute_then_drop",
            (
                _operation(
                    "op-001-impute_missing",
                    OperationType.IMPUTE_MISSING,
                    ("qty",),
                    ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
                ),
                _operation(
                    "op-002-drop_column",
                    OperationType.DROP_COLUMN,
                    ("junk",),
                    DropColumnParameters(),
                ),
            ),
        ),
        (
            "drop_then_impute",
            (
                _operation(
                    "op-001-drop_column",
                    OperationType.DROP_COLUMN,
                    ("junk",),
                    DropColumnParameters(),
                ),
                _operation(
                    "op-002-impute_missing",
                    OperationType.IMPUTE_MISSING,
                    ("qty",),
                    ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
                ),
            ),
        ),
        (
            "normalize_then_drop",
            (
                _operation(
                    "op-001-normalize_numeric_text",
                    OperationType.NORMALIZE_NUMERIC_TEXT,
                    ("amount_text",),
                    NormalizeNumericTextParameters(),
                ),
                _operation(
                    "op-002-drop_column",
                    OperationType.DROP_COLUMN,
                    ("junk",),
                    DropColumnParameters(),
                ),
            ),
        ),
        (
            "drop_then_normalize",
            (
                _operation(
                    "op-001-drop_column",
                    OperationType.DROP_COLUMN,
                    ("junk",),
                    DropColumnParameters(),
                ),
                _operation(
                    "op-002-normalize_numeric_text",
                    OperationType.NORMALIZE_NUMERIC_TEXT,
                    ("amount_text",),
                    NormalizeNumericTextParameters(),
                ),
            ),
        ),
        (
            "trim_then_drop",
            (
                _operation(
                    "op-001-trim_whitespace",
                    OperationType.TRIM_WHITESPACE,
                    ("amount_text",),
                    TrimWhitespaceParameters(),
                ),
                _operation(
                    "op-002-drop_column",
                    OperationType.DROP_COLUMN,
                    ("junk",),
                    DropColumnParameters(),
                ),
            ),
        ),
        (
            "drop_then_trim",
            (
                _operation(
                    "op-001-drop_column",
                    OperationType.DROP_COLUMN,
                    ("junk",),
                    DropColumnParameters(),
                ),
                _operation(
                    "op-002-trim_whitespace",
                    OperationType.TRIM_WHITESPACE,
                    ("amount_text",),
                    TrimWhitespaceParameters(),
                ),
            ),
        ),
        (
            "cast_then_drop",
            (
                _operation(
                    "op-001-cast_column",
                    OperationType.CAST_COLUMN,
                    ("qty",),
                    CastColumnParameters(
                        target_type=CastTarget.NUMERIC,
                        errors=CastErrorPolicy.RAISE,
                    ),
                ),
                _operation(
                    "op-002-drop_column",
                    OperationType.DROP_COLUMN,
                    ("junk",),
                    DropColumnParameters(),
                ),
            ),
        ),
        (
            "drop_then_cast",
            (
                _operation(
                    "op-001-drop_column",
                    OperationType.DROP_COLUMN,
                    ("junk",),
                    DropColumnParameters(),
                ),
                _operation(
                    "op-002-cast_column",
                    OperationType.CAST_COLUMN,
                    ("qty",),
                    CastColumnParameters(
                        target_type=CastTarget.NUMERIC,
                        errors=CastErrorPolicy.RAISE,
                    ),
                ),
            ),
        ),
    ),
)
def test_drop_structure_survives_legitimate_survivor_rewrites(
    label, operations
) -> None:
    del label

    source = _frame()
    plan = _plan(*operations)
    run = run_allowlisted_plan(source, plan)

    assert run.success
    assert run.dataframe is not None

    execution = ExecutionResult(
        execution_id="execution-p1-drop-structure",
        dataset_id=plan.dataset_id,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        accepted_review_attempt=1,
        success=True,
        source_fingerprint=dataframe_fingerprint(source),
        result_fingerprint=dataframe_fingerprint(run.dataframe),
        before_row_count=len(source),
        after_row_count=len(run.dataframe),
        before_column_count=source.shape[1],
        after_column_count=run.dataframe.shape[1],
        operation_records=run.operation_records,
    )

    results = _operation_preservation_results(
        plan,
        execution,
        run,
        source,
        run.dataframe,
    )

    drop_results = [
        result
        for result in results
        if result.kind is InvariantKind.DROPPED_COLUMN_STRUCTURE
    ]

    assert len(drop_results) == 1
    assert drop_results[0].mandatory is True
    assert drop_results[0].status is InvariantStatus.PASS

def test_drop_structure_alone_passes() -> None:
    source = _frame()
    plan = _plan(
        _operation(
            "op-001-drop_column",
            OperationType.DROP_COLUMN,
            ("junk",),
            DropColumnParameters(),
        )
    )

    run = run_allowlisted_plan(source, plan)

    assert run.success
    assert run.dataframe is not None

    execution = ExecutionResult(
        execution_id="execution-p1-drop-alone",
        dataset_id=plan.dataset_id,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        accepted_review_attempt=1,
        success=True,
        source_fingerprint=dataframe_fingerprint(source),
        result_fingerprint=dataframe_fingerprint(run.dataframe),
        before_row_count=len(source),
        after_row_count=len(run.dataframe),
        before_column_count=source.shape[1],
        after_column_count=run.dataframe.shape[1],
        operation_records=run.operation_records,
    )

    results = _operation_preservation_results(
        plan,
        execution,
        run,
        source,
        run.dataframe,
    )

    assert len(results) == 1
    assert results[0].kind is InvariantKind.DROPPED_COLUMN_STRUCTURE
    assert results[0].mandatory is True
    assert results[0].status is InvariantStatus.PASS

def test_imputation_one_null_in_2001_rows_validates() -> None:
    frame = pd.DataFrame(
        {
            "order_id": list(range(2001)),
            "value": [1.0] * 2000 + [None],
        }
    )
    context = _context(frame)

    assert context.null_counts["value"] == 1

    #issue = _issue_for(context, DiagnosticIssueKind.NULL_VALUES, "value")

    result = _validate(
        context,
        TransformationOperation(
            operation_id="op-001-impute_missing",
            operation_type=OperationType.IMPUTE_MISSING,
            target_columns=("value",),
            parameters=ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
            diagnostic_issue_ids=(),
            user_requirement_ids=("intent.impute_missing",),
            rationale="Fill the measured missing value.",
            expected_effect="Remove the missing value.",
            risk=RiskLevel.LOW,
            requires_human_approval=True,
        ),
    )

    assert result.valid
    assert "IMPUTE_NO_MISSING_VALUES" not in {
        finding.code for finding in result.findings
    }
    assert context.null_counts["value"] == 1
    assert not any(
        issue.kind is DiagnosticIssueKind.NULL_VALUES
        and "value" in issue.affected_columns
        for issue in context.diagnostic_report.issues
    )

def test_imputation_zero_nulls_is_refused_at_plan_time() -> None:
    frame = pd.DataFrame(
        {
            "order_id": list(range(2001)),
            "value": [1.0] * 2001,
        }
    )
    context = _context(frame)

    assert context.null_counts["value"] == 0

    #issue = _issue_for(context, DiagnosticIssueKind.CANDIDATE_TYPE_CONVERSION, "value")

    operation = TransformationOperation(
        operation_id="op-001-impute_missing",
        operation_type=OperationType.IMPUTE_MISSING,
        target_columns=("value",),
        parameters=ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
        diagnostic_issue_ids=(),
        user_requirement_ids=("intent.impute_missing",),
        rationale="Probe zero-null refusal.",
        expected_effect="No cells should be imputed.",
        risk=RiskLevel.LOW,
        requires_human_approval=True,
    )

    result = _validate(context, operation)

    assert not result.valid
    assert "IMPUTE_NO_MISSING_VALUES" in {
        finding.code for finding in result.findings
    }

# ---------------------------------------------------------------------------
# The agent tool boundary
# ---------------------------------------------------------------------------


def _draft_context():
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 2],
            "imgUrl": ["u", "v", "u"],
            "amount_text": ["$1,304.20", "$27.00", "$1,000"],
            "qty": [1.0, None, 3.0],
        }
    )
    intent = UserIntent(
        intent_id="intent-agent",
        user_goal="Prepare.",
        selected_key_columns=("order_id",),
        acceptable_row_loss_pct=50,
    )
    report = _diagnose(frame)
    return build_planning_context(
        report,
        intent,
        (),
        column_alias_map=build_column_alias_map(report, intent),
    )


def test_each_new_tool_forbids_arguments_outside_its_schema() -> None:
    from datachef.agents.tools import (
        DropColumnArgs,
        ImputeMissingArgs,
        NormalizeNumericTextArgs,
    )

    for args_model, payload in (
        (DropColumnArgs, {"target_columns": ["a"]}),
        (
            ImputeMissingArgs,
            {"target_columns": ["a"], "strategy": ImputeStrategy.MEAN},
        ),
        (NormalizeNumericTextArgs, {"target_columns": ["a"]}),
    ):
        base = dict(payload, rationale="r", expected_effect="e")
        args_model(**base)
        with pytest.raises(ValidationError):
            args_model(**base, regex="[0-9]+")
        with pytest.raises(ValidationError):
            args_model(**base, sql="drop table")


def test_each_new_tool_refuses_an_aliased_or_missing_column() -> None:
    from datachef.agents.tools import (
        ALIASED,
        MISSING_COLUMN,
        DropColumnArgs,
        ImputeMissingArgs,
        NormalizeNumericTextArgs,
        PlanDraft,
        apply_operation_args,
    )

    context = _draft_context()
    aliased = context.privacy_manifest.aliased_columns
    issue = context.diagnostic_report.issues[0].issue_id

    cases = [
        ("propose_drop_column", DropColumnArgs, {}),
        (
            "propose_impute_missing",
            ImputeMissingArgs,
            {"strategy": ImputeStrategy.MEAN},
        ),
        ("propose_normalize_numeric_text", NormalizeNumericTextArgs, {}),
    ]
    for tool_name, args_model, extra in cases:
        draft = PlanDraft(context=context)
        refused = apply_operation_args(
            draft,
            tool_name,
            args_model(
                target_columns=["does_not_exist"],
                diagnostic_issue_ids=[issue],
                rationale="r",
                expected_effect="e",
                **extra,
            ),
        )
        assert refused == {"accepted": False, "reason_code": MISSING_COLUMN}, tool_name
        assert draft.operations == []

        if aliased:
            draft = PlanDraft(context=context)
            refused = apply_operation_args(
                draft,
                tool_name,
                args_model(
                    target_columns=[aliased[0]],
                    diagnostic_issue_ids=[issue],
                    rationale="r",
                    expected_effect="e",
                    **extra,
                ),
            )
            assert refused == {"accepted": False, "reason_code": ALIASED}, tool_name
            assert draft.operations == []


def test_a_new_tool_refuses_a_proposal_citing_no_diagnostic_issue() -> None:
    from datachef.agents.tools import (
        UNKNOWN_ISSUE,
        NormalizeNumericTextArgs,
        PlanDraft,
        apply_operation_args,
    )

    context = _draft_context()
    draft = PlanDraft(context=context)

    refused = apply_operation_args(
        draft,
        "propose_normalize_numeric_text",
        NormalizeNumericTextArgs(
            target_columns=["amount_text"],
            diagnostic_issue_ids=[],
            rationale="r",
            expected_effect="e",
        ),
    )

    assert refused == {"accepted": False, "reason_code": UNKNOWN_ISSUE}
    assert draft.operations == []


def test_the_agent_can_build_the_normalize_then_cast_chain() -> None:
    """Both operations cite the same noise issue, which validation allows."""

    from datachef.agents.tools import (
        CastColumnArgs,
        NormalizeNumericTextArgs,
        PlanDraft,
        apply_operation_args,
    )

    context = _draft_context()
    noise = _issue_for(
        context, DiagnosticIssueKind.CANDIDATE_NUMERIC_TEXT_NOISE, "amount_text"
    )
    draft = PlanDraft(context=context)

    first = apply_operation_args(
        draft,
        "propose_normalize_numeric_text",
        NormalizeNumericTextArgs(
            target_columns=["amount_text"],
            diagnostic_issue_ids=[noise],
            rationale="Numeric text carries symbols.",
            expected_effect="Strips the symbols.",
        ),
    )
    second = apply_operation_args(
        draft,
        "propose_cast_column",
        CastColumnArgs(
            target_columns=["amount_text"],
            target_type=CastTarget.NUMERIC,
            errors=CastErrorPolicy.RAISE,
            diagnostic_issue_ids=[noise],
            rationale="Now castable.",
            expected_effect="Numeric column.",
        ),
    )

    assert first["accepted"] is True
    assert second["accepted"] is True
    assert [operation.operation_type for operation in draft.operations] == [
        OperationType.NORMALIZE_NUMERIC_TEXT,
        OperationType.CAST_COLUMN,
    ]
    assert validate_plan(context, draft.build_plan()).valid


def test_the_crew_task_tells_the_agent_to_chain_unprompted() -> None:
    source = (
        REPO_ROOT / "datachef" / "agents" / "plan_crew.py"
    ).read_text(encoding="utf-8")

    assert "CANDIDATE_NUMERIC_TEXT_NOISE" in source
    assert "propose_normalize_numeric_text" in source
    assert "without being asked" in source
