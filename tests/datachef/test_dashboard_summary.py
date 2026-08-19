"""The dashboard MVP: a deterministic summary of a verified run, and its screen.

Every number the dashboard shows is read from evidence that already passed the
gate -- the QA report, the execution record, the approved plan and verified gold.
These tests pin that: the summary is gated exactly like gold, it carries no cell
values, and it answers the questions the run leaves the user holding.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from datachef.application import (
    ColumnReadiness,
    CsvParserOptions,
    DashboardFailure,
    DashboardSummary,
    DataChefController,
    UploadFormat,
    UploadRequest,
    build_dashboard_summary,
)
from datachef.contracts import (
    HumanDecision,
    OperationType,
    QAStatus,
    UserIntent,
    WorkflowStage,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

ML_OBJECTIVE = (
    "Prepare this table for ML modelling, the objective is to use this table to "
    "train a model to predict the price column based on the other columns. For "
    "the missing values, check if the missing values in the column title is over "
    "40% and there's no mode drop it, otherwise if the mode exists for the column "
    "title use it to impute all null values, impute the missing values of the "
    "column stars using the mean, and impute the column price using the median, "
    "drop the category_id column, check the distribution in boughtInLastMonth if "
    "it has over 40% of null and 0s as values drop the column. finally drop the "
    "duplicate values based on the asin column"
)

CSV = (
    b"asin,title,stars,price,category_id,boughtInLastMonth\n"
    b"a1,t1,4.5,10.0,104,0\n"
    b"a2,,,20.0,104,0\n"
    b"a3,,4.0,,104,\n"
    b"a4,,3.5,40.0,104,\n"
    b"a1,t1,4.5,10.0,104,\n"
)


def _controller(objective: str = ML_OBJECTIVE, *, execute: bool = True):
    controller = DataChefController()
    controller.load_upload(
        UploadRequest(
            content=CSV,
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(encoding="utf-8-sig"),
        )
    )
    controller.diagnose()
    controller.submit_intent(
        UserIntent(
            intent_id="intent-dash",
            user_goal=objective,
            selected_key_columns=("asin",),
            acceptable_row_loss_pct=50,
        ),
        (),
    )
    controller.prepare_plan(command_id="plan")
    if execute:
        controller.record_human_decision(HumanDecision.APPROVE, command_id="approve")
        controller.execute_current_plan(command_id="execute")
    return controller


@pytest.fixture(scope="module")
def summary() -> DashboardSummary:
    controller = _controller()
    assert (
        controller.session.workflow_runtime.state.stage is WorkflowStage.QA_PASSED
    )
    built = controller.build_dashboard_summary()
    assert isinstance(built, DashboardSummary)
    return built


# ---------------------------------------------------------------------------
# 1. did it succeed  /  2. how much changed
# ---------------------------------------------------------------------------


def test_the_summary_reports_the_qa_outcome(summary) -> None:
    assert summary.qa_status is QAStatus.PASS
    assert summary.modelling_ready is True
    assert "no missing values remain" in summary.readiness_headline


def test_the_summary_reports_row_and_column_movement(summary) -> None:
    assert (summary.rows_before, summary.rows_after) == (5, 4)
    assert summary.rows_removed == 1
    assert summary.row_loss_pct == pytest.approx(20.0)
    assert (summary.columns_before, summary.columns_after) == (6, 4)


# ---------------------------------------------------------------------------
# 3. which columns went  /  4. what happened to missing values
# ---------------------------------------------------------------------------


def test_the_summary_names_removed_and_retained_columns(summary) -> None:
    assert set(summary.removed_columns) == {"category_id", "boughtInLastMonth"}
    assert summary.retained_columns == ("asin", "title", "stars", "price")
    assert set(summary.removed_columns).isdisjoint(summary.retained_columns)


def test_the_summary_reports_missingness_before_and_after_per_column(summary) -> None:
    by_column = {item.column: item for item in summary.columns}

    assert by_column["title"].nulls_before == 3
    assert by_column["title"].nulls_after == 0
    assert by_column["stars"].nulls_before == 1
    assert by_column["stars"].nulls_after == 0
    assert by_column["asin"].nulls_before == 0
    assert summary.nulls_before_total == 5
    assert summary.nulls_after_total == 0
    assert summary.nulls_filled == 5
    assert all(item.complete for item in summary.columns)


# ---------------------------------------------------------------------------
# 5. duplicates and rows  /  6. which transformations ran
# ---------------------------------------------------------------------------


def test_the_summary_reports_duplicate_movement(summary) -> None:
    assert summary.duplicate_keys_before == 1
    assert summary.duplicate_keys_after == 0
    assert summary.duplicate_rows_after == 0


def test_the_summary_lists_the_approved_operations_in_order(summary) -> None:
    assert [
        (item.operation_type, item.target_columns) for item in summary.operations
    ] == [
        # The order the objective states, ending with "finally drop the
        # duplicate values based on the asin column".
        (OperationType.IMPUTE_MISSING, ("title",)),
        (OperationType.IMPUTE_MISSING, ("stars",)),
        (OperationType.IMPUTE_MISSING, ("price",)),
        (OperationType.DROP_COLUMN, ("category_id",)),
        (OperationType.DROP_COLUMN, ("boughtInLastMonth",)),
        (OperationType.DEDUPLICATE_BY_KEYS, ("asin",)),
    ]
    details = {item.operation_type: item.detail for item in summary.operations}
    assert details[OperationType.IMPUTE_MISSING].startswith("strategy ")
    assert details[OperationType.DEDUPLICATE_BY_KEYS] == "keys asin"
    dedup = next(
        item
        for item in summary.operations
        if item.operation_type is OperationType.DEDUPLICATE_BY_KEYS
    )
    assert (dedup.rows_before, dedup.rows_after) == (5, 4)


def test_the_operation_detail_never_carries_agent_prose(summary) -> None:
    """Only closed enum values and column names, never a rationale."""

    for item in summary.operations:
        assert "requested" not in item.detail.lower()
        assert "objective" not in item.detail.lower()
        assert len(item.detail) < 60


# ---------------------------------------------------------------------------
# 7. is it ready  /  8. the target column
# ---------------------------------------------------------------------------


def test_the_target_column_is_read_from_the_objective(summary) -> None:
    assert summary.target_column == "price"
    assert summary.target_is_usable is True
    target = [item for item in summary.columns if item.is_target]
    assert [item.column for item in target] == ["price"]


def test_no_target_is_claimed_when_the_objective_names_none() -> None:
    controller = _controller("drop the category_id column")
    built = controller.build_dashboard_summary()

    assert built.target_column is None
    assert built.target_is_usable is False
    # This objective only drops a column, so the nulls it never asked to impute
    # are still there. Readiness follows the table, not the missing target.
    assert built.nulls_after_total > 0
    assert built.modelling_ready is False
    assert "missing value(s) remain" in built.readiness_headline
    # An otherwise-clean table with no named target is still ready.
    clean = built.model_copy(update={"nulls_after_total": 0, "columns": ()})
    assert clean.modelling_ready is True


def test_a_target_that_was_dropped_is_not_reported_as_usable() -> None:
    """The target must be resolved against gold, not against the objective."""

    controller = _controller(
        "predict the category_id column, drop the category_id column"
    )
    built = controller.build_dashboard_summary()

    assert "category_id" not in built.retained_columns
    assert built.target_column is None  # it does not survive, so it is not claimed
    assert built.target_is_usable is False


def test_residual_missing_values_block_modelling_readiness() -> None:
    """Readiness describes the table, so leftover nulls make it not ready."""

    controller = _controller("drop the category_id column")
    built = controller.build_dashboard_summary()
    with_nulls = built.model_copy(
        update={
            "nulls_after_total": 3,
            "columns": (
                ColumnReadiness(
                    column="stars",
                    dtype="float64",
                    nulls_before=4,
                    nulls_after=3,
                    distinct_count=2,
                ),
            ),
        }
    )

    assert with_nulls.modelling_ready is False
    assert "missing value(s) remain" in with_nulls.readiness_headline


# ---------------------------------------------------------------------------
# Gating and privacy
# ---------------------------------------------------------------------------


def test_the_summary_is_refused_before_approval_and_execution() -> None:
    controller = _controller(execute=False)

    built = controller.build_dashboard_summary()

    assert isinstance(built, DashboardFailure)
    assert built.code.value in {"GOLD_UNAVAILABLE", "EVIDENCE_INCOMPLETE"}


def test_the_summary_is_refused_on_a_raw_only_session() -> None:
    controller = DataChefController()
    controller.load_upload(
        UploadRequest(
            content=CSV,
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(encoding="utf-8-sig"),
        )
    )

    assert isinstance(controller.build_dashboard_summary(), DashboardFailure)


def test_a_tampered_gold_frame_is_refused() -> None:
    """Gated on the same evidence as every other gold consumer."""

    controller = _controller()
    runtime = controller.session.workflow_runtime
    tampered = runtime.gold_dataframe.copy(deep=True)
    tampered.loc[tampered.index[0], "price"] = 999.0

    built = build_dashboard_summary(
        dataclasses.replace(runtime, gold_dataframe=tampered),
        controller.session.intent,
    )

    assert isinstance(built, DashboardFailure)


def test_the_summary_carries_no_cell_values(summary) -> None:
    """Counts, dtypes, column names and operation metadata only."""

    serialized = summary.model_dump_json()

    for value in ("t1", "a1", "a2", "4.5", "10.0", "104"):
        assert f'"{value}"' not in serialized, value
    # And none of the objective prose travels with it.
    assert "modelling" not in serialized
    assert "impute the missing values" not in serialized


def test_the_summary_is_deterministic() -> None:
    first = _controller().build_dashboard_summary()
    second = _controller().build_dashboard_summary()

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


# ---------------------------------------------------------------------------
# The real screen
# ---------------------------------------------------------------------------


def test_the_results_screen_renders_the_readiness_dashboard() -> None:
    from streamlit.testing.v1 import AppTest

    from ui import state as ui_state

    def widget(at, kind, key):
        for element in getattr(at, kind):
            if getattr(element, "key", None) == key:
                return element
        raise KeyError(f"{kind}:{key}")

    at = AppTest.from_file(str(REPO_ROOT / "ui" / "app.py"), default_timeout=180)
    at.run()
    at.file_uploader[0].set_value(("amazon.csv", CSV, "text/csv"))
    at.run()
    widget(at, "button", ui_state.DIAGNOSE_WIDGET).click()
    at.run()
    widget(at, "text_area", ui_state.GOAL_WIDGET).set_value(ML_OBJECTIVE)
    widget(at, "multiselect", ui_state.KEY_COLUMNS_WIDGET).set_value(["asin"])
    widget(at, "slider", ui_state.ROW_LOSS_WIDGET).set_value(50.0)
    widget(at, "button", ui_state.SUBMIT_INTENT_WIDGET).click()
    at.run()
    widget(at, "button", ui_state.PREPARE_PLAN_WIDGET).click()
    at.run()
    widget(at, "button", ui_state.APPROVE_WIDGET).click()
    at.run()
    widget(at, "button", ui_state.EXECUTE_WIDGET).click()
    at.run()

    markdown = " ".join(element.value for element in at.markdown)
    assert "Data readiness" in markdown
    assert "What changed" in markdown
    assert "Removed columns" in markdown

    # The four headline metrics.
    labels = [element.label for element in at.metric]
    for label in ("Rows", "Columns", "Missing values", "Duplicate rows"):
        assert label in labels, labels
    values = {element.label: element.value for element in at.metric}
    assert values["Rows"] == "4"
    assert values["Columns"] == "4"
    assert values["Missing values"] == "0"

    success = " ".join(element.value for element in at.success)
    assert "no missing values remain" in success
    captions = " ".join(element.value for element in at.caption)
    assert "Target `price`" in captions
    # The per-column and per-operation tables both rendered.
    assert len(at.dataframe) >= 2
    assert len(at.exception) == 0, [item.value for item in at.exception]
