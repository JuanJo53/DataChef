from __future__ import annotations

import ast
from datetime import datetime, timezone
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from datachef.application import DataChefController, ScreenId
from datachef.contracts import OperationType, QAStatus, WorkflowStage
from datachef.workflow import WorkflowRuntime, execute_workflow
from ui import state as ui_state


REPO_ROOT = Path(__file__).parents[2]
APP_PATH = str(REPO_ROOT / "ui" / "app.py")
NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
TIMEOUT = 120

CSV = (
    b"order_id,region,amount,ordered_on\n"
    b"1,North,10,2026-01-01\n"
    b"2,South,20,2026-01-02\n"
    b"2,South,20,2026-01-02\n"
    b"3,North,30,2026-01-03\n"
)
CLEAN_CSV = b"category,amount\na,1\nb,2\nc,3\n"
# 9 rows; `category_id` is constant, so the deterministic planner proposes a
# key deduplication that would remove 8 of 9 rows (88.89%).
CONSTANT_KEY_CSV = (
    b"asin,title,imgUrl,productURL,stars,reviews,price,listPrice,category_id,isBestSeller,boughtInLastMonth\n"
    b"B014TMV5YE,,https://m.media-amazon.com/images/I/815dLQKYIYL._AC_UL320_.jpg,https://www.amazon.com/dp/B014TMV5YE,4.5,0,139.99,0,104,False,2000\n"
    b"B07GDLCQXV,,https://m.media-amazon.com/images/I/81bQlm7vf6L._AC_UL320_.jpg,https://www.amazon.com/dp/B07GDLCQXV,4.5,0,169.99,209.99,104,False,1000\n"
    b"B07XSCCZYG,,https://m.media-amazon.com/images/I/71EA35zvJBL._AC_UL320_.jpg,https://www.amazon.com/dp/B07XSCCZYG,4.6,0,365.49,429.99,104,False,300\n"
    b"B08MVFKGJM,,https://m.media-amazon.com/images/I/91k6NYLQyIL._AC_UL320_.jpg,https://www.amazon.com/dp/B08MVFKGJM,4.6,0,291.59,354.37,104,False,400\n"
    b"B01DJLKZBA,,https://m.media-amazon.com/images/I/61NJoaZcP9L._AC_UL320_.jpg,https://www.amazon.com/dp/B01DJLKZBA,4.5,0,174.99,309.99,104,False,400\n"
    b"B07XSCD2R4,,https://m.media-amazon.com/images/I/61LnBNsSBSL._AC_UL320_.jpg,https://www.amazon.com/dp/B07XSCD2R4,4.5,0,144.49,0,104,False,500\n"
    b"B07MXF4G8K,,https://m.media-amazon.com/images/I/71CghLYrnAL._AC_UL320_.jpg,https://www.amazon.com/dp/B07MXF4G8K,4.5,0,169.99,0,104,False,400\n"
    b"B07H515VCZ,,https://m.media-amazon.com/images/I/81f3h+YHOXL._AC_UL320_.jpg,https://www.amazon.com/dp/B07H515VCZ,4.5,0,299.99,0,104,False,100\n"
    b"B014TMV5YE,,https://m.media-amazon.com/images/I/815dLQKYIYL._AC_UL320_.jpg,https://www.amazon.com/dp/B014TMV5YE,4.5,0,139.99,0,104,False,2000\n"
)
CAST_FAILURE_JSON = (
    b'[{"order_id":1,"amount_text":"1"},{"order_id":2,"amount_text":"2"},'
    b'{"order_id":3,"amount_text":"3"},{"order_id":4,"amount_text":"4"},'
    b'{"order_id":5,"amount_text":"bad"}]'
)


def _app(controller: DataChefController | None = None) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if controller is not None:
        at.session_state[ui_state.CONTROLLER] = controller
    return at


def _widget(at: AppTest, kind: str, key: str):
    for element in getattr(at, kind):
        if getattr(element, "key", None) == key:
            return element
    raise KeyError(f"{kind}:{key}")


def _has(at: AppTest, kind: str, key: str) -> bool:
    return any(getattr(element, "key", None) == key for element in getattr(at, kind))


def _controller(at: AppTest) -> DataChefController:
    return at.session_state[ui_state.CONTROLLER]


def _upload_and_diagnose(
    at: AppTest,
    content: bytes = CSV,
    filename: str = "orders.csv",
    mime: str = "text/csv",
) -> AppTest:
    at.run()
    at.file_uploader[0].set_value((filename, content, mime))
    at.run()
    _widget(at, "button", ui_state.DIAGNOSE_WIDGET).click()
    at.run()
    # Diagnosing lands on Diagnostics, screen 2. The objective is screen 3.
    _widget(at, "button", ui_state.CONTINUE_TO_INTENT_WIDGET).click()
    at.run()
    return at


def _submit_intent(
    at: AppTest,
    *,
    key_columns: list[str] | None = None,
    cast_columns: list[str] | None = None,
    row_loss: float = 50.0,
    goal: str | None = None,
    questions: str | None = None,
) -> AppTest:
    if goal is not None:
        _widget(at, "text_area", ui_state.GOAL_WIDGET).set_value(goal)
    if questions is not None:
        _widget(at, "text_area", ui_state.QUESTIONS_WIDGET).set_value(questions)
    if key_columns:
        _widget(at, "multiselect", ui_state.KEY_COLUMNS_WIDGET).set_value(key_columns)
    if cast_columns:
        _widget(at, "multiselect", ui_state.CAST_REQUEST_WIDGET).set_value(cast_columns)
    _widget(at, "slider", ui_state.ROW_LOSS_WIDGET).set_value(row_loss)
    _widget(at, "button", ui_state.SUBMIT_INTENT_WIDGET).click()
    at.run()
    return at


def _prepare(at: AppTest) -> AppTest:
    _widget(at, "button", ui_state.PREPARE_PLAN_WIDGET).click()
    at.run()
    return at


def _approve_and_execute(at: AppTest) -> AppTest:
    _widget(at, "button", ui_state.APPROVE_WIDGET).click()
    at.run()
    _widget(at, "button", ui_state.EXECUTE_WIDGET).click()
    at.run()
    return at


def _pass_run(at: AppTest | None = None) -> AppTest:
    at = at or _app()
    _upload_and_diagnose(at)
    _submit_intent(at, key_columns=["order_id"])
    _prepare(at)
    _approve_and_execute(at)
    return at


def _assert_no_traceback(at: AppTest) -> None:
    assert len(at.exception) == 0, [item.value for item in at.exception]


def _assert_locked_down(at: AppTest) -> None:
    """No download control and no dashboard may be rendered."""

    assert len(at.download_button) == 0
    assert "DataChef Dashboard" not in [header.value for header in at.header]
    _assert_no_traceback(at)


def test_happy_path_reaches_pass_with_the_full_bundle_on_results() -> None:
    at = _pass_run()

    session = _controller(at).session
    assert session.workflow_runtime.state.stage is WorkflowStage.QA_PASSED
    assert session.screen.value == "RESULTS"
    assert [button.key for button in at.download_button] == [
        "datachef_w_download_CLEANED_CSV",
        "datachef_w_download_CLEANED_PARQUET",
        "datachef_w_download_TRANSFORMATION_PLAN_JSON",
        "datachef_w_download_QA_REPORT_JSON",
        "datachef_w_download_EXECUTION_CHANGE_LOG_JSON",
        "datachef_w_download_PIPELINE_SCRIPT_PY",
        "datachef_w_download_MANIFEST_JSON",
    ]
    # The dashboard is its own screen now, so Results does not draw one.
    assert "DataChef Dashboard" not in [header.value for header in at.header]
    assert "6 \u00b7 Results" in [header.value for header in at.header]
    assert tuple(item.kind.value for item in session.command_history) == (
        "PLAN_PREPARATION",
        "HUMAN_DECISION",
        "EXECUTION",
    )
    _assert_no_traceback(at)


def test_results_carries_the_quality_evidence_the_run_produced() -> None:
    """Quality assurance stopped being a screen; its evidence did not move."""

    at = _pass_run()

    text = _all_text(at)
    report = _controller(at).session.workflow_runtime.state.qa_report
    assert report.status is QAStatus.PASS
    assert "Quality report" in text
    assert report.qa_report_id in text
    assert "Execution" in text
    labels = {item.label: item.value for item in at.metric}
    assert labels["Rows before"] == str(report.before_row_count)
    assert labels["Rows after"] == str(report.after_row_count)
    assert labels["Columns after"] == str(report.after_column_count)
    _assert_no_traceback(at)


def test_a_failed_quality_run_reports_itself_on_results() -> None:
    """The failing verdict has to be readable somewhere, and Results is it."""

    at = _app()
    _upload_and_diagnose(at, CAST_FAILURE_JSON, "records.json", "application/json")
    _submit_intent(at, cast_columns=["amount_text"], row_loss=40.0)
    _prepare(at)
    _approve_and_execute(at)

    session = _controller(at).session
    assert session.workflow_runtime.state.stage is WorkflowStage.QA_FAILED
    assert session.screen.value == "RESULTS"
    text = _all_text(at)
    assert "Quality assurance failed" in text
    assert "Gold was withheld" in text
    assert "Quality report" in text
    _assert_locked_down(at)


def test_the_dashboard_is_its_own_screen_reached_from_results() -> None:
    at = _pass_run()
    assert _controller(at).session.screen.value == "RESULTS"

    _widget(at, "button", ui_state.CONTINUE_TO_DASHBOARD_WIDGET).click()
    at.run()

    session = _controller(at).session
    assert session.screen.value == "DASHBOARD"
    headers = [header.value for header in at.header]
    assert "7 \u00b7 Dashboard" in headers
    assert "DataChef Dashboard" in headers
    # Results owns the bundle; the dashboard screen carries no download.
    assert len(at.download_button) == 0
    # Nothing was re-run to get here.
    assert tuple(item.kind.value for item in session.command_history) == (
        "PLAN_PREPARATION",
        "HUMAN_DECISION",
        "EXECUTION",
    )
    _assert_no_traceback(at)


def test_the_dashboard_refuses_to_draw_without_a_passing_quality_gate() -> None:
    """Navigating straight to screen 7 grants nothing the controller withheld."""

    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")
    _submit_intent(at, key_columns=["order_id"], row_loss=10.0)
    _prepare(at)
    controller = _controller(at)
    assert controller.session.workflow_runtime.state.stage is WorkflowStage.AWAITING_APPROVAL

    controller.navigate(ScreenId.DASHBOARD)
    at.run()

    session = _controller(at).session
    assert session.screen.value == "DASHBOARD"
    assert session.pending_approval is None
    assert session.workflow_runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert session.workflow_runtime.gold_dataframe is None
    assert "GOLD_UNAVAILABLE" in _all_text(at)
    _assert_locked_down(at)


def test_results_offers_no_dashboard_shortcut_without_a_bundle() -> None:
    at = _app()
    _upload_and_diagnose(at, CAST_FAILURE_JSON, "records.json", "application/json")
    _submit_intent(at, cast_columns=["amount_text"], row_loss=40.0)
    _prepare(at)
    _approve_and_execute(at)

    assert _controller(at).session.screen.value == "RESULTS"
    assert not _has(at, "button", ui_state.CONTINUE_TO_DASHBOARD_WIDGET)
    _assert_locked_down(at)


def test_page_refresh_after_pass_performs_no_further_work() -> None:
    at = _pass_run()
    before = _controller(at).session

    at.run()
    at.run()

    after = _controller(at).session
    assert after.revision == before.revision
    assert after.command_history == before.command_history
    assert len(at.download_button) == 7
    _assert_no_traceback(at)


def test_same_action_driven_twice_replays_and_performs_no_work() -> None:
    at = _app()
    _upload_and_diagnose(at)
    _submit_intent(at, key_columns=["order_id"])
    _prepare(at)
    minted = at.session_state[ui_state.PLAN_COMMAND]
    _widget(at, "button", ui_state.REJECT_WIDGET).click()
    at.run()
    _widget(at, "button", ui_state.EXECUTE_WIDGET).click()
    at.run()
    before = _controller(at).session
    assert before.workflow_runtime.state.stage is WorkflowStage.PLAN_REJECTED
    assert _has(at, "button", ui_state.PREPARE_PLAN_WIDGET)

    _prepare(at)

    after = _controller(at).session
    assert at.session_state[ui_state.PLAN_COMMAND] == minted
    assert at.session_state[ui_state.LAST_RESULT].code == "PLAN_COMMAND_REPLAYED"
    assert after.revision == before.revision
    assert after.command_history == before.command_history
    _assert_locked_down(at)


def test_plan_rejection_offers_no_downloads_or_dashboard() -> None:
    at = _app()
    _upload_and_diagnose(at)
    _submit_intent(at, key_columns=["order_id"])
    _prepare(at)
    _widget(at, "button", ui_state.REJECT_WIDGET).click()
    at.run()
    _widget(at, "button", ui_state.EXECUTE_WIDGET).click()
    at.run()

    session = _controller(at).session
    assert session.workflow_runtime.state.stage is WorkflowStage.PLAN_REJECTED
    assert session.workflow_runtime.gold_dataframe is None
    _assert_locked_down(at)


def test_genuine_quality_failure_offers_no_downloads_or_dashboard() -> None:
    at = _app()
    _upload_and_diagnose(at, CAST_FAILURE_JSON, "records.json", "application/json")
    _submit_intent(at, cast_columns=["amount_text"], row_loss=40.0)
    _prepare(at)
    _approve_and_execute(at)

    session = _controller(at).session
    assert session.workflow_runtime.state.stage is WorkflowStage.QA_FAILED
    assert session.workflow_runtime.gold_dataframe is None
    assert session.screen.value == "RESULTS"
    _assert_locked_down(at)


def test_fabricated_warning_result_is_refused_without_downloads() -> None:
    def warning_execute(runtime, approval):
        completed = execute_workflow(runtime, approval)
        assert completed.state.qa_report is not None
        return WorkflowRuntime(
            state=completed.state.model_copy(
                update={
                    "stage": WorkflowStage.QA_WARNING,
                    "qa_report": completed.state.qa_report.model_copy(
                        update={"status": QAStatus.WARN}
                    ),
                }
            ),
            raw_dataframe=completed.raw_dataframe,
            transformed_dataframe=completed.transformed_dataframe,
            gold_dataframe=None,
            user_intent=completed.user_intent,
            column_alias_map=completed.column_alias_map,
        )

    at = _app(DataChefController(clock=lambda: NOW, execute_service=warning_execute))
    _upload_and_diagnose(at)
    _submit_intent(at, key_columns=["order_id"])
    _prepare(at)
    _approve_and_execute(at)

    session = _controller(at).session
    assert at.session_state[ui_state.LAST_RESULT].code == "EXECUTION_EVIDENCE_INVALID"
    assert session.workflow_runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert session.workflow_runtime.gold_dataframe is None
    _assert_locked_down(at)


def test_execution_service_failure_is_sanitized_without_downloads() -> None:
    def failing_execute(runtime, approval):
        del runtime, approval
        raise RuntimeError("private execution detail C:\\secret\\path.csv")

    at = _app(DataChefController(clock=lambda: NOW, execute_service=failing_execute))
    _upload_and_diagnose(at)
    _submit_intent(at, key_columns=["order_id"])
    _prepare(at)
    _approve_and_execute(at)

    assert at.session_state[ui_state.LAST_RESULT].code == "EXECUTION_SERVICE_FAILURE"
    rendered = " ".join(
        [item.value for item in at.error]
        + [item.value for item in at.warning]
        + [item.value for item in at.markdown]
        + [item.value for item in at.caption]
    )
    assert "private execution detail" not in rendered
    assert "C:\\secret" not in rendered
    _assert_locked_down(at)


def test_raw_only_session_offers_no_downloads_or_dashboard() -> None:
    at = _app()
    at.run()
    at.file_uploader[0].set_value(("orders.csv", CSV, "text/csv"))
    at.run()

    session = _controller(at).session
    assert session.source is not None
    assert session.workflow_runtime is None
    _assert_locked_down(at)


def test_empty_plan_reaching_pass_still_offers_the_full_bundle() -> None:
    at = _app()
    _upload_and_diagnose(at, CLEAN_CSV, "clean.csv")
    _submit_intent(at, goal="Use the already clean table.", row_loss=0.0)
    _prepare(at)
    session = _controller(at).session
    assert session.workflow_runtime.state.transformation_plan.operations == ()

    _approve_and_execute(at)

    session = _controller(at).session
    assert session.workflow_runtime.state.stage is WorkflowStage.QA_PASSED
    assert len(at.download_button) == 7
    _assert_no_traceback(at)


def test_blocking_finding_disables_the_approve_control() -> None:
    at = _app()
    _upload_and_diagnose(at)
    _submit_intent(at, cast_columns=["region"])
    _prepare(at)

    session = _controller(at).session
    assert any(finding.blocking for finding in session.findings)
    assert session.screen.value == "APPROVAL"
    approve = _widget(at, "button", ui_state.APPROVE_WIDGET)
    assert approve.proto.disabled is True
    reject = _widget(at, "button", ui_state.REJECT_WIDGET)
    assert reject.proto.disabled is False
    _assert_locked_down(at)


def test_already_numeric_cast_is_visible_approvable_and_runs_as_an_empty_plan() -> None:
    at = _app()
    _upload_and_diagnose(at, CLEAN_CSV, "clean.csv")
    _submit_intent(at, cast_columns=["amount"])
    _prepare(at)

    session = _controller(at).session
    plan = session.workflow_runtime.state.transformation_plan
    assert plan is not None
    assert plan.operations == ()
    assert any(
        finding.code == "REQUEST_ALREADY_SATISFIED" and not finding.blocking
        for finding in session.findings
    )
    assert not any(finding.code == "REQUEST_NOT_PLANNED" for finding in session.findings)
    assert "REQUEST_ALREADY_SATISFIED" in _all_text(at)
    assert _widget(at, "button", ui_state.APPROVE_WIDGET).proto.disabled is False

    _approve_and_execute(at)

    session = _controller(at).session
    assert session.workflow_runtime.state.stage is WorkflowStage.QA_PASSED
    assert session.workflow_runtime.gold_dataframe is not None
    assert session.workflow_runtime.state.execution_result.operation_records == ()
    assert "REQUEST_ALREADY_SATISFIED" in _all_text(at)
    _assert_no_traceback(at)


def test_walmart_drop_requests_and_already_numeric_cast_reach_results() -> None:
    content = (
        b"Store,Date,Weekly_Sales,Fuel_Price,CPI\n"
        b"1,05-02-2010,1000.0,2.572,211.096\n"
        b"1,12-02-2010,1100.0,2.548,211.242\n"
        b"2,19-02-2010,1200.0,2.514,211.289\n"
    )
    at = _app()
    _upload_and_diagnose(at, content, "walmart.csv")
    _submit_intent(
        at,
        goal="Drop Date and CPI.",
        cast_columns=["Fuel_Price"],
        row_loss=0,
    )
    _prepare(at)

    session = _controller(at).session
    assert session.workflow_runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert [
        (operation.operation_type, operation.target_columns)
        for operation in session.workflow_runtime.state.transformation_plan.operations
    ] == [
        (OperationType.DROP_COLUMN, ("Date",)),
        (OperationType.DROP_COLUMN, ("CPI",)),
    ]
    assert any(
        finding.code == "REQUEST_ALREADY_SATISFIED" and not finding.blocking
        for finding in session.findings
    )
    assert not any(finding.blocking for finding in session.findings)

    _approve_and_execute(at)

    runtime = _controller(at).session.workflow_runtime
    assert runtime.state.stage is WorkflowStage.QA_PASSED
    assert "Date" not in runtime.gold_dataframe
    assert "CPI" not in runtime.gold_dataframe
    assert len(at.download_button) == 7
    _assert_no_traceback(at)


def test_accounting_currency_normalization_reaches_pass_and_artifacts() -> None:
    content = (
        b'Profit,Sales,Postal Code\n'
        b'"($5)","$16",10001\n'
        b'"($2,574)","$2,574",10002\n'
        b'"-$65","$0",10003\n'
    )
    at = _app()
    _upload_and_diagnose(at, content, "superstore.csv")
    _submit_intent(
        at,
        goal="Convert Profit and Sales to numeric. Drop Postal Code.",
        cast_columns=["Profit", "Sales"],
        row_loss=0,
    )
    _prepare(at)
    _approve_and_execute(at)

    runtime = _controller(at).session.workflow_runtime
    assert runtime.state.stage is WorkflowStage.QA_PASSED
    assert list(runtime.gold_dataframe["Profit"]) == [-5, -2574, -65]
    assert list(runtime.gold_dataframe["Sales"]) == [16, 2574, 0]
    assert "Postal Code" not in runtime.gold_dataframe
    assert len(at.download_button) == 7
    _assert_no_traceback(at)


def test_dashboard_renders_question_grounded_charts_before_legacy_recommendations() -> None:
    at = _app()
    _upload_and_diagnose(at, CSV, "orders.csv")
    _submit_intent(
        at,
        key_columns=["order_id"],
        row_loss=50.0,
        questions="Show the trend of amount over ordered_on",
    )
    _prepare(at)
    _approve_and_execute(at)
    _widget(at, "button", ui_state.CONTINUE_TO_DASHBOARD_WIDGET).click()
    at.run()

    text = _all_text(at)
    assert "Charts answering your questions" in text
    assert "Other recommended charts" in text
    assert "could not be charted" not in text
    handoff = _controller(at).build_dashboard_handoff()
    assert handoff.context.question_resolutions[0].chart.chart_type.value == "LINE"
    assert len(at.get("plotly_chart")) >= 1
    _assert_no_traceback(at)


def test_dashboard_keeps_an_ambiguous_business_question_visible() -> None:
    at = _app()
    _upload_and_diagnose(at, CSV, "orders.csv")
    _submit_intent(
        at,
        key_columns=["order_id"],
        row_loss=50.0,
        questions="What should I know?",
    )
    _prepare(at)
    _approve_and_execute(at)
    _widget(at, "button", ui_state.CONTINUE_TO_DASHBOARD_WIDGET).click()
    at.run()

    text = _all_text(at)
    assert "Question 1 could not be answered" in text
    assert "State a trend, distribution, relationship, comparison, or aggregation" in text
    assert "What should I know?" in [item.value for item in at.text]
    assert "Other recommended charts" in text
    _assert_no_traceback(at)


def test_ml_dashboard_separates_authored_questions_and_resolves_ranking_scatter() -> None:
    rows = [
        "asin,title,stars,price,category_id,isBestSeller,boughtInLastMonth\n"
    ]
    rows.extend(
        (
            f"A{index:04d},"
            f"{'' if index < 250 else 'Common title'},"
            f"{'' if index % 10 == 0 else 3.0 + index % 3},"
            f"{'' if index % 11 == 0 else 10.0 + index},"
            f"104,False,{'' if index < 250 else 0}\n"
        )
        for index in range(500)
    )
    at = _app()
    _upload_and_diagnose(at, "".join(rows).encode("utf-8"), "ml-products.csv")
    suggestions = _widget(at, "multiselect", ui_state.SUGGESTED_QUESTIONS_WIDGET)
    missingness = next(
        option for option in suggestions.options if "missing" in option.casefold()
    )
    suggestions.set_value([missingness])
    _submit_intent(
        at,
        goal=(
            "Prepare this table for ML modelling to predict the price column. "
            "If title is over 40% missing and there is no mode drop it, otherwise "
            "use the mode to impute title, impute stars using the mean, impute "
            "price using the median, drop category_id, if boughtInLastMonth is "
            "over 40% null and has zero values drop it, finally drop duplicate "
            "values based on asin, if there are columns with one distinct value "
            "drop them"
        ),
        questions=(
            "What titles have the highest prices?\n"
            "What's the relationship between price and stars?"
        ),
    )
    _prepare(at)
    plan = _controller(at).session.workflow_runtime.state.transformation_plan
    assert [
        (operation.operation_type, operation.target_columns)
        for operation in plan.operations
    ] == [
        (OperationType.IMPUTE_MISSING, ("title",)),
        (OperationType.IMPUTE_MISSING, ("stars",)),
        (OperationType.IMPUTE_MISSING, ("price",)),
        (OperationType.DROP_COLUMN, ("category_id",)),
        (OperationType.DROP_COLUMN, ("boughtInLastMonth",)),
        (OperationType.DROP_COLUMN, ("isBestSeller",)),
        (OperationType.DEDUPLICATE_BY_KEYS, ("asin",)),
    ]
    _approve_and_execute(at)
    assert len(at.download_button) == 7
    _widget(at, "button", ui_state.CONTINUE_TO_DASHBOARD_WIDGET).click()
    at.run()

    handoff = _controller(at).build_dashboard_handoff()
    authored = handoff.context.authored_question_resolutions
    recommended = handoff.context.recommended_question_resolutions
    assert len(authored) == 2
    assert len(recommended) == 1
    assert authored[0].chart.chart_type.value == "BAR"
    assert authored[0].chart.x_column == "title"
    assert authored[0].chart.y_column == "price"
    assert authored[0].chart.ranking.value == "DESCENDING"
    assert authored[1].chart.chart_type.value == "SCATTER"
    assert (authored[1].chart.x_column, authored[1].chart.y_column) == (
        "price",
        "stars",
    )
    text = _all_text(at)
    assert "Charts answering your questions" in text
    assert "Other recommended charts / diagnostics" in text
    assert "Question 1 · Bar" in text
    assert "Question 2 · Scatter" in text
    assert "Question 3" not in text
    assert "Recommendation 1" in text
    _assert_no_traceback(at)


def test_keep_only_control_produces_explicit_approved_drops_and_exact_gold_schema() -> None:
    at = _app()
    content = b"metric_a,metric_b,metric_c,metric_d\n1,2,3,4\n5,6,7,8\n"
    _upload_and_diagnose(at, content, "metrics.csv")
    _widget(at, "multiselect", ui_state.KEEP_ONLY_COLUMNS_WIDGET).set_value(
        ["metric_a", "metric_b"]
    )
    _submit_intent(at)
    _prepare(at)

    session = _controller(at).session
    plan = session.workflow_runtime.state.transformation_plan
    assert plan is not None
    assert [
        (operation.operation_type, operation.target_columns)
        for operation in plan.operations
        if operation.operation_type is OperationType.DROP_COLUMN
    ] == [
        (OperationType.DROP_COLUMN, ("metric_c",)),
        (OperationType.DROP_COLUMN, ("metric_d",)),
    ]
    assert _widget(at, "button", ui_state.APPROVE_WIDGET).proto.disabled is False

    _approve_and_execute(at)

    runtime = _controller(at).session.workflow_runtime
    assert runtime.state.stage is WorkflowStage.QA_PASSED
    assert list(runtime.gold_dataframe.columns) == ["metric_a", "metric_b"]
    _assert_no_traceback(at)


def test_compute_column_objective_runs_through_approval_to_verified_gold() -> None:
    at = _app()
    content = b"quantity,unit_price,category\n2,4.5,A\n3,10.0,B\n"
    _upload_and_diagnose(at, content, "sales.csv")
    _submit_intent(
        at,
        goal="Create a new column called total by multiplying quantity by unit_price.",
    )
    _prepare(at)

    plan = _controller(at).session.workflow_runtime.state.transformation_plan
    assert plan is not None
    compute = [
        operation
        for operation in plan.operations
        if operation.operation_type is OperationType.COMPUTE_COLUMN
    ]
    assert len(compute) == 1
    assert compute[0].parameters.output_column == "total"
    assert _widget(at, "button", ui_state.APPROVE_WIDGET).proto.disabled is False

    _approve_and_execute(at)

    runtime = _controller(at).session.workflow_runtime
    assert runtime.state.stage is WorkflowStage.QA_PASSED
    assert runtime.gold_dataframe["total"].tolist() == [9.0, 30.0]
    assert len(at.download_button) == 7
    _assert_no_traceback(at)


def test_keep_only_with_currency_casts_retains_profit_and_sales() -> None:
    content = (
        b"Postal Code,Profit,Sales,Profit  From Sales w/o discount,Segment\n"
        b"100,$1,$10,9,A\n"
        b"200,($2),$20,18,B\n"
    )
    at = _app()
    _upload_and_diagnose(at, content, "compound-sales.csv")
    _widget(at, "multiselect", ui_state.KEEP_ONLY_COLUMNS_WIDGET).set_value(
        ["Profit", "Sales", "Segment"]
    )
    _submit_intent(
        at,
        goal="Turn Profit and Sales to numeric",
        cast_columns=["Profit", "Sales"],
        row_loss=0,
    )
    _prepare(at)

    plan = _controller(at).session.workflow_runtime.state.transformation_plan
    drops = [
        operation.target_columns
        for operation in plan.operations
        if operation.operation_type is OperationType.DROP_COLUMN
    ]
    assert drops == [
        ("Postal Code",),
        ("Profit  From Sales w/o discount",),
    ]
    assert ("Profit",) not in drops and ("Sales",) not in drops

    _approve_and_execute(at)
    runtime = _controller(at).session.workflow_runtime
    assert runtime.state.stage is WorkflowStage.QA_PASSED
    assert list(runtime.gold_dataframe.columns) == ["Profit", "Sales", "Segment"]
    assert list(runtime.gold_dataframe["Profit"]) == [1, -2]
    assert list(runtime.gold_dataframe["Sales"]) == [10, 20]
    assert len(at.download_button) == 7
    _assert_no_traceback(at)


def test_generic_single_distinct_objective_drops_every_eligible_constant() -> None:
    content = (
        b"row_id,constant_a,constant_b,varying\n"
        b"1,x,7,10\n2,x,7,20\n3,x,7,30\n"
    )
    at = _app()
    _upload_and_diagnose(at, content, "constants.csv")
    _submit_intent(
        at,
        goal="If there are columns with one distinct value, drop them",
        row_loss=0,
    )
    _prepare(at)

    plan = _controller(at).session.workflow_runtime.state.transformation_plan
    assert [
        operation.target_columns
        for operation in plan.operations
        if operation.operation_type is OperationType.DROP_COLUMN
    ] == [("constant_a",), ("constant_b",)]

    _approve_and_execute(at)
    runtime = _controller(at).session.workflow_runtime
    assert runtime.state.stage is WorkflowStage.QA_PASSED
    assert list(runtime.gold_dataframe.columns) == ["row_id", "varying"]
    assert len(at.download_button) == 7
    _assert_no_traceback(at)


def test_walmart_ranking_and_weekday_questions_render_from_verified_gold() -> None:
    content = (
        b"Store,Date,Weekly_Sales,CPI\n"
        b"1,2026-08-17,100,200\n"
        b"1,2026-08-18,150,201\n"
        b"2,2026-08-24,300,202\n"
    )
    at = _app()
    _upload_and_diagnose(at, content, "walmart-questions.csv")
    _submit_intent(
        at,
        goal="Drop CPI",
        row_loss=0,
        questions=(
            "What stores have the most weekly sales?\n"
            "On what day of the week do we have the most sales?\n"
            "Which stores have the highest CPI?"
        ),
    )
    _prepare(at)
    _approve_and_execute(at)
    _widget(at, "button", ui_state.CONTINUE_TO_DASHBOARD_WIDGET).click()
    at.run()

    handoff = _controller(at).build_dashboard_handoff()
    authored = handoff.context.authored_question_resolutions
    assert [item.status.value for item in authored] == [
        "RESOLVED",
        "RESOLVED",
        "QUESTION_UNSUPPORTED",
    ]
    assert authored[0].chart.x_column == "Store"
    assert authored[0].chart.y_column == "Weekly_Sales"
    assert authored[0].chart.aggregation.value == "SUM"
    assert authored[1].chart.category_transform.value == "DAY_OF_WEEK"
    assert authored[2].reason_code == "QUESTION_COLUMN_UNAVAILABLE"
    assert len(at.get("plotly_chart")) >= 2
    _assert_no_traceback(at)


def test_full_reset_clears_state_and_rotates_the_uploader_key() -> None:
    at = _app()
    at.run()
    generation = _controller(at).session.uploader_generation
    assert _has(at, "file_uploader", ui_state.uploader_key(generation))
    _pass_run(at)
    assert _controller(at).session.uploader_generation == generation
    assert len(at.download_button) == 7

    _widget(at, "button", ui_state.RESET_WIDGET).click()
    at.run()

    session = _controller(at).session
    assert session.uploader_generation == generation + 1
    assert session.source is None
    assert session.display_diagnostic_report is None
    assert session.intent is None
    assert session.workflow_runtime is None
    assert session.findings == ()
    assert session.pending_approval is None
    assert session.command_history == ()
    assert session.screen.value == "UPLOAD"
    assert _has(at, "file_uploader", ui_state.uploader_key(generation + 1))
    assert not _has(at, "file_uploader", ui_state.uploader_key(generation))
    assert ui_state.PLAN_COMMAND not in at.session_state
    _assert_locked_down(at)


def test_preview_is_off_by_default_and_toggling_keeps_command_history() -> None:
    at = _pass_run()
    before = _controller(at).session
    assert before.preview_enabled is False
    preview = _widget(at, "toggle", ui_state.PREVIEW_WIDGET)
    assert preview.value is False

    preview.set_value(True)
    at.run()

    after = _controller(at).session
    assert after.preview_enabled is True
    assert after.command_history == before.command_history
    assert after.workflow_runtime.state.stage is before.workflow_runtime.state.stage
    assert len(at.download_button) == 7
    _assert_no_traceback(at)


@pytest.mark.parametrize(
    "drive",
    (
        pytest.param(lambda at: at.run(), id="cold-start"),
        pytest.param(
            lambda at: _upload_and_diagnose(at, b"order_id\n", "broken.csv"),
            id="empty-dataset",
        ),
        pytest.param(
            lambda at: _upload_and_diagnose(at, b"a,b\n1\n", "ragged.csv"),
            id="ragged-csv",
        ),
        pytest.param(
            lambda at: _upload_and_diagnose(at, b"not-parquet", "bad.parquet", "application/octet-stream"),
            id="malformed-parquet",
        ),
    ),
)
def test_refusal_routes_never_surface_a_traceback(drive) -> None:
    at = _app()

    try:
        drive(at)
    except KeyError:
        pass

    _assert_locked_down(at)


def _refused_plan(cast_columns: list[str] | None = None) -> AppTest:
    """Drive a genuine row-loss refusal: a real duplicate key over the threshold.

    `order_id` has one duplicate in twelve rows (8.33%), so an acceptable row
    loss of 5% is refused by validation.
    """

    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")
    _submit_intent(at, cast_columns=cast_columns, row_loss=5.0)
    _prepare(at)
    runtime = _controller(at).session.workflow_runtime
    assert runtime is not None
    assert runtime.state.stage is WorkflowStage.PLAN_REJECTED
    assert runtime.state.last_error_code == "PLAN_VALIDATION_ATTEMPTS_EXHAUSTED"
    return at


def _revise(
    at: AppTest,
    *,
    row_loss: float | None = None,
    cast_columns: list[str] | None = None,
    key_columns: list[str] | None = None,
) -> AppTest:
    if row_loss is not None:
        _widget(at, "slider", ui_state.REVISE_ROW_LOSS_WIDGET).set_value(row_loss)
    if cast_columns is not None:
        _widget(at, "multiselect", ui_state.REVISE_CAST_REQUEST_WIDGET).set_value(cast_columns)
    if key_columns is not None:
        _widget(at, "multiselect", ui_state.REVISE_KEY_COLUMNS_WIDGET).set_value(key_columns)
    _widget(at, "button", ui_state.REVISE_SUBMIT_WIDGET).click()
    at.run()
    return at


def _all_text(at: AppTest) -> str:
    return " ".join(
        [item.value for item in at.error]
        + [item.value for item in at.warning]
        + [item.value for item in at.info]
        + [item.value for item in at.success]
        + [item.value for item in at.markdown]
        + [item.value for item in at.caption]
        + [f"{item.label} {item.value}" for item in at.metric]
    )


def test_row_loss_refusal_shows_the_finding_operation_and_both_thresholds() -> None:
    at = _refused_plan(cast_columns=["region"])

    text = _all_text(at)
    assert "ROW_LOSS_THRESHOLD" in text
    assert "CUMULATIVE_ROW_LOSS_THRESHOLD" in text
    assert "Estimated row loss exceeds the user's approved threshold." in text
    assert "op-deduplicate-keys-order_id" in text
    assert "8.33%" in text
    assert "1 row(s)" in text
    labels = {item.label: item.value for item in at.metric}
    assert labels["Estimated cumulative row loss"] == "8.33%"
    assert labels["Your acceptable row loss"] == "5.00%"
    _assert_locked_down(at)


def test_refusal_message_names_the_code_and_never_blames_a_reviewer() -> None:
    at = _refused_plan()

    runtime = _controller(at).session.workflow_runtime
    assert runtime.state.review_history == ()
    message = " ".join(item.value for item in at.error)
    assert "PLAN_VALIDATION_ATTEMPTS_EXHAUSTED" in message
    assert "reviewer" not in message.lower()
    _assert_locked_down(at)


def test_revise_recovers_from_a_refusal_without_reset_or_reupload() -> None:
    at = _refused_plan(cast_columns=["region"])
    session = _controller(at).session
    fingerprint = session.source.identity.fingerprint
    generation = session.uploader_generation
    assert any(finding.code == "REQUEST_NOT_PLANNED" for finding in session.findings)

    _revise(at, row_loss=90.0, cast_columns=[])

    revised = _controller(at).session
    assert revised.source is not None
    assert revised.source.identity.fingerprint == fingerprint
    assert revised.uploader_generation == generation
    assert revised.display_diagnostic_report is not None
    assert revised.workflow_runtime is None
    assert revised.command_history == ()
    assert revised.intent.acceptable_row_loss_pct == 90.0
    assert revised.requested_transformations == ()

    _prepare(at)

    final = _controller(at).session
    assert at.session_state[ui_state.LAST_RESULT].code == "PLAN_AWAITING_APPROVAL"
    assert final.workflow_runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert final.source.identity.fingerprint == fingerprint
    _assert_no_traceback(at)


def test_identical_revise_submission_changes_nothing() -> None:
    at = _refused_plan()
    before = _controller(at).session

    _revise(at)

    after = _controller(at).session
    assert after.revision == before.revision
    assert after.command_history == before.command_history
    assert after.intent == before.intent
    assert after.workflow_runtime.state.stage is WorkflowStage.PLAN_REJECTED
    _assert_locked_down(at)


def test_prepare_after_a_material_revise_mints_a_fresh_command_id() -> None:
    at = _refused_plan()
    minted = at.session_state[ui_state.PLAN_COMMAND]

    _revise(at, row_loss=95.0)

    assert ui_state.PLAN_COMMAND not in at.session_state
    _prepare(at)

    result = at.session_state[ui_state.LAST_RESULT]
    assert result.code == "PLAN_AWAITING_APPROVAL"
    assert at.session_state[ui_state.PLAN_COMMAND] != minted
    history = _controller(at).session.command_history
    assert tuple(item.result_code for item in history) == ("PLAN_AWAITING_APPROVAL",)
    _assert_no_traceback(at)


def test_rerun_safety_holds_across_the_revise_action() -> None:
    at = _refused_plan()
    _revise(at, row_loss=95.0)
    before = _controller(at).session

    at.run()
    at.run()

    after = _controller(at).session
    assert after.revision == before.revision
    assert after.command_history == before.command_history
    assert after.intent == before.intent
    _assert_locked_down(at)


def test_reset_clears_the_revise_form_widget_state() -> None:
    at = _refused_plan()
    _widget(at, "slider", ui_state.REVISE_ROW_LOSS_WIDGET).set_value(42.0)
    at.run()
    assert at.session_state[ui_state.REVISE_ROW_LOSS_WIDGET] == 42.0

    _widget(at, "button", ui_state.RESET_WIDGET).click()
    at.run()

    for key in (
        ui_state.REVISE_ROW_LOSS_WIDGET,
        ui_state.REVISE_KEY_COLUMNS_WIDGET,
        ui_state.REVISE_CAST_REQUEST_WIDGET,
        ui_state.REVISE_DEDUP_REQUEST_WIDGET,
    ):
        assert key not in at.session_state
    session = _controller(at).session
    assert session.source is None
    assert session.screen.value == "UPLOAD"
    _assert_locked_down(at)


def test_validation_refusal_route_exposes_no_downloads_or_dashboard() -> None:
    at = _refused_plan(cast_columns=["region"])

    _assert_locked_down(at)
    assert _controller(at).session.workflow_runtime.gold_dataframe is None


DEMO_CSV = (
    b"order_id,region,product,unit_price,quantity\n"
    b"1001,North,Laptop Stand,45.50,2\n"
    b"1002,South,USB-C Hub,29.99,5\n"
    b"1003,North,Monitor Arm,89.00,1\n"
    b"1004,West,Keyboard,72.25,3\n"
    b"1005,South,Mouse Pad,12.00,10\n"
    b"1006,East,Webcam,64.99,2\n"
    b"1007,North,Desk Lamp,38.40,4\n"
    b"1008,West,Cable Kit,19.95,6\n"
    b"1009,East,Laptop Sleeve,27.30,3\n"
    b"1010,South,Docking Station,149.00,1\n"
    b"1011,North,Headset,88.75,2\n"
    b"1011,North,Headset,88.75,2\n"
)


def test_refused_plan_screen_shows_each_operation_risk_as_text() -> None:
    at = _refused_plan()

    text = _all_text(at)
    runtime = _controller(at).session.workflow_runtime
    operations = runtime.state.transformation_plan.operations
    assert operations
    for operation in operations:
        assert operation.operation_type.value in text
        assert f"risk **{operation.risk.value}**" in text
    assert "risk **HIGH**" in text


def test_plan_operation_risk_is_not_hidden_inside_a_collapsed_label() -> None:
    at = _refused_plan()

    visible = " ".join(item.value for item in at.markdown)
    assert "risk **HIGH**" in visible
    labels = " ".join(str(getattr(item, "label", "")) for item in at.expander)
    assert "risk" not in labels


def test_upload_screen_states_only_the_disk_claim_datachef_controls() -> None:
    at = _app()
    at.run()
    at.file_uploader[0].set_value(("orders.csv", DEMO_CSV, "text/csv"))
    at.run()

    captions = " ".join(item.value for item in at.caption)
    assert "DataChef never writes your file to disk" in captions
    assert "reads only its extension to pick a parser" in captions
    assert "contacts no provider" in captions
    assert "Nothing is written to disk" not in captions


def test_diagnosis_renders_on_the_screen_the_user_lands_on_after_diagnosing() -> None:
    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")

    session = _controller(at).session
    assert session.screen.value == "INTENT"
    labels = {item.label: item.value for item in at.metric}
    assert labels["Health score"] == "97"
    assert labels["Grade"] == "A"
    assert labels["Duplicate rows"] == "1"
    text = _all_text(at)
    assert "Deterministic diagnosis" in text
    assert "Duplicate rows detected" in text
    assert "Duplicate values detected for key columns: order_id" in text
    assert "columns: order_id" in text
    assert "`HIGH`" in text
    _assert_no_traceback(at)


def test_operation_risk_renders_as_text_on_the_approval_screen() -> None:
    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")
    _submit_intent(at, key_columns=["order_id"], row_loss=10.0)
    _prepare(at)

    session = _controller(at).session
    assert session.screen.value == "APPROVAL"
    operations = session.workflow_runtime.state.transformation_plan.operations
    assert operations
    text = _all_text(at)
    for operation in operations:
        assert f"risk **{operation.risk.value}**" in text
    assert "risk **HIGH**" in text
    _assert_no_traceback(at)


def test_stage_indicator_renders_all_seven_screens_and_marks_the_current_one() -> None:
    at = _app()
    at.run()

    sidebar_text = " ".join(item.value for item in at.sidebar.markdown)
    for label in (
        "1 · Upload",
        "2 · Diagnostics",
        "3 · Objective",
        "4 · Plan",
        "5 · Approve",
        "6 · Results",
        "7 · Dashboard",
    ):
        assert label in sidebar_text
    assert 'class="stage-current"' in sidebar_text
    assert "●" in sidebar_text
    assert 'class="stage-pending"' in sidebar_text
    # Quality assurance is not a place the user goes.
    assert "Quality" not in sidebar_text
    _assert_no_traceback(at)


def test_the_progress_rail_never_offers_a_quality_screen() -> None:
    at = _pass_run()

    sidebar_text = " ".join(item.value for item in at.sidebar.markdown)
    assert "Quality" not in sidebar_text
    navigation = {
        button.key
        for button in at.sidebar.button
        if str(button.key).startswith(ui_state.STAGE_NAV_WIDGET)
    }
    assert f"{ui_state.STAGE_NAV_WIDGET}_QA" not in navigation
    assert f"{ui_state.STAGE_NAV_WIDGET}_DIAGNOSE" in navigation
    _assert_no_traceback(at)


def test_diagnostics_is_screen_two_and_shows_the_deterministic_evidence() -> None:
    at = _app()
    at.run()
    at.file_uploader[0].set_value(("orders.csv", DEMO_CSV, "text/csv"))
    at.run()
    # Accepting a file does not move the user; running the diagnosis does.
    assert _controller(at).session.screen.value == "UPLOAD"

    _widget(at, "button", ui_state.DIAGNOSE_WIDGET).click()
    at.run()

    session = _controller(at).session
    assert session.screen.value == "DIAGNOSE"
    sidebar_text = " ".join(item.value for item in at.sidebar.markdown)
    assert 'class="stage-current"' in sidebar_text
    assert "2 · Diagnostics" in sidebar_text
    report = session.display_diagnostic_report
    text = _all_text(at)
    assert "Data health" in text
    assert "Detected issues" in text
    assert "View column details" in [item.label for item in at.expander]
    assert len(at.dataframe) == 1
    profiles = at.dataframe[0].value
    assert profiles["Column"].tolist() == [
        profile.name for profile in report.column_profiles
    ]
    # It is a read of existing evidence: no plan, no workflow, no command.
    assert session.workflow_runtime is None
    assert session.command_history == ()
    _assert_locked_down(at)


def test_the_objective_screen_is_reached_from_diagnostics_as_screen_three() -> None:
    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")

    session = _controller(at).session
    assert session.screen.value == "INTENT"
    assert "3 · Objective" in [header.value for header in at.header]
    _assert_no_traceback(at)


def test_plan_and_approve_are_screens_four_and_five() -> None:
    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")
    _submit_intent(at, key_columns=["order_id"], row_loss=10.0)

    assert _controller(at).session.screen.value == "PLAN"
    assert "4 · Plan" in [header.value for header in at.header]

    _prepare(at)

    assert _controller(at).session.screen.value == "APPROVAL"
    assert "5 · Approve" in [header.value for header in at.header]
    _assert_no_traceback(at)


def test_no_rendered_text_says_intent() -> None:
    at = _pass_run()

    for probe in (at, _refused_plan()):
        rendered = " ".join(
            [item.value for item in probe.markdown]
            + [item.value for item in probe.caption]
            + [item.value for item in probe.header]
            + [item.value for item in probe.title]
            + [item.value for item in probe.info]
            + [item.value for item in probe.success]
            + [item.value for item in probe.warning]
            + [item.value for item in probe.error]
            + [str(item.label) for item in probe.button]
            + [str(item.label) for item in probe.expander]
        )
        assert "Intent" not in rendered
        assert "intent" not in rendered


def test_upload_preview_is_opt_in_capped_and_evidence_free() -> None:
    at = _app()
    at.run()
    at.file_uploader[0].set_value(("orders.csv", DEMO_CSV, "text/csv"))
    at.run()

    before = _controller(at).session
    assert before.preview_enabled is False
    assert len(at.dataframe) == 0
    toggle = _widget(at, "button", ui_state.UPLOAD_PREVIEW_WIDGET)
    assert "10-row preview" in toggle.label

    toggle.click()
    at.run()

    after = _controller(at).session
    assert after.preview_enabled is True
    assert after.command_history == before.command_history
    assert after.workflow_runtime is before.workflow_runtime is None
    assert after.source.identity == before.source.identity
    assert len(at.dataframe) == 1
    assert at.dataframe[0].value.shape[0] <= 10
    _assert_no_traceback(at)


def test_objective_exposes_keep_only_without_duplicate_required_column_concept() -> None:
    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")

    labels = [str(item.label) for item in at.multiselect]
    keep_only_label = "Keep only these columns"
    key_label = "Which column tells you two rows are the same record?"
    assert "Columns that must survive" not in labels
    assert keep_only_label in labels
    assert key_label in labels
    assert labels.index(keep_only_label) < labels.index(key_label)
    assert "Key columns" not in labels
    _assert_no_traceback(at)


def test_dashboard_questions_sit_in_their_own_section_below_the_cleaning_fields() -> None:
    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")

    text = _all_text(at)
    assert "Business questions for the dashboard" in text
    assert (
        "Optional. These shape the dashboard you get at the end — they do not "
        "change how your data is cleaned."
    ) in text

    headings = [item.value for item in at.markdown]
    questions_heading = headings.index("### Business questions for the dashboard")
    requests_heading = headings.index("### Typed transformation requests")
    assert requests_heading < questions_heading

    multiselects = [str(item.label) for item in at.multiselect]
    assert "Columns that must survive" not in multiselects
    keep_only_at = multiselects.index("Keep only these columns")
    key_at = multiselects.index("Which column tells you two rows are the same record?")
    cast_at = multiselects.index("Cast these columns to numeric")
    suggested_at = multiselects.index(
        "Deterministic suggested questions to carry into the dashboard"
    )
    # Cleaning fields keep their order; both question inputs come last. The
    # typed dedup request is gone: the key-column question above asks for it.
    assert "Deduplicate rows by these keys" not in multiselects
    assert keep_only_at < key_at < cast_at < suggested_at
    text_areas = [str(item.label) for item in at.text_area]
    assert text_areas[-1] == "Your analytical questions (one per line)"
    _assert_no_traceback(at)


def test_reset_clears_the_dashboard_question_widgets() -> None:
    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")
    _widget(at, "text_area", ui_state.QUESTIONS_WIDGET).set_value("Which region wins?")
    at.run()
    assert at.session_state[ui_state.QUESTIONS_WIDGET] == "Which region wins?"

    _widget(at, "button", ui_state.RESET_WIDGET).click()
    at.run()

    assert ui_state.QUESTIONS_WIDGET not in at.session_state
    assert ui_state.SUGGESTED_QUESTIONS_WIDGET not in at.session_state
    assert _controller(at).session.source is None
    _assert_locked_down(at)


def test_empty_plan_names_the_findings_it_cannot_act_on_and_still_passes() -> None:
    at = _app()
    _upload_and_diagnose(at, CONSTANT_KEY_CSV, "catalogue.csv")
    _submit_intent(at, row_loss=0.0)
    _prepare(at)

    session = _controller(at).session
    assert session.screen.value == "APPROVAL"
    plan = session.workflow_runtime.state.transformation_plan
    assert plan.operations == ()
    kinds = {issue.kind.value for issue in session.display_diagnostic_report.issues}
    assert "DUPLICATE_ROWS" in kinds and "NULL_VALUES" in kinds

    text = _all_text(at)
    assert "The reviewed plan is empty" in text
    assert "The diagnosis reported" in text
    assert "DUPLICATE_ROWS" in text
    assert "NULL_VALUES" in text
    assert "no executable operation for those" in text

    _approve_and_execute(at)

    final = _controller(at).session
    assert final.workflow_runtime.state.stage is WorkflowStage.QA_PASSED
    assert final.workflow_runtime.state.qa_report.status is QAStatus.PASS
    assert len(at.download_button) == 7
    _assert_no_traceback(at)


def test_a_non_empty_plan_does_not_show_the_empty_plan_explanation() -> None:
    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")
    _submit_intent(at, key_columns=["order_id"], row_loss=10.0)
    _prepare(at)

    session = _controller(at).session
    assert session.workflow_runtime.state.transformation_plan.operations
    text = _all_text(at)
    assert "The reviewed plan is empty" not in text
    assert "The diagnosis reported" not in text
    assert "no executable operation for those" not in text
    _assert_no_traceback(at)


def test_stage_indicator_navigates_back_without_losing_workflow_state() -> None:
    at = _pass_run()
    before = _controller(at).session
    assert before.screen.value == "RESULTS"

    _widget(at, "button", f"{ui_state.STAGE_NAV_WIDGET}_PLAN").click()
    at.run()

    after = _controller(at).session
    assert after.screen.value == "PLAN"
    assert after.command_history == before.command_history
    assert after.pending_approval == before.pending_approval
    assert after.workflow_runtime.state == before.workflow_runtime.state
    assert after.source.identity == before.source.identity
    assert len(at.download_button) == 0
    _assert_no_traceback(at)

    _widget(at, "button", f"{ui_state.STAGE_NAV_WIDGET}_RESULTS").click()
    at.run()
    returned = _controller(at).session
    assert returned.screen.value == "RESULTS"
    assert returned.command_history == before.command_history
    assert len(at.download_button) == 7


def test_navigating_forward_cannot_skip_approval_or_execution() -> None:
    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")
    _submit_intent(at, key_columns=["order_id"], row_loss=10.0)
    _prepare(at)
    controller = _controller(at)
    assert controller.session.workflow_runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert controller.session.pending_approval is None

    controller.navigate(ScreenId.RESULTS)
    at.run()

    session = _controller(at).session
    assert session.screen.value == "RESULTS"
    assert session.pending_approval is None
    assert session.workflow_runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert session.workflow_runtime.gold_dataframe is None
    assert len(at.download_button) == 0
    assert "DataChef Dashboard" not in [header.value for header in at.header]
    _assert_no_traceback(at)


def test_ui_never_imports_a_provider_or_the_transformation_agent() -> None:
    forbidden_roots = {"crewai", "langchain_google_genai"}
    forbidden_modules = {"google.genai", "crew.transformation_agent"}
    scanned = 0

    for path in sorted((REPO_ROOT / "ui").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not {name.split(".")[0] for name in imported} & forbidden_roots, path
        for name in imported:
            assert not any(
                name == blocked or name.startswith(f"{blocked}.")
                for blocked in forbidden_modules
            ), path

    assert scanned >= 9


def test_approval_screen_shows_row_loss_beside_the_users_threshold() -> None:
    """Informed exact approval: the numbers must be on the approval screen."""

    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")
    _submit_intent(at, key_columns=["order_id"], row_loss=10.0)
    _prepare(at)

    assert _controller(at).session.screen.value == "APPROVAL"
    labels = {item.label: item.value for item in at.metric}
    assert labels["Estimated cumulative row loss"] == "8.33%"
    assert labels["Your acceptable row loss"] == "10.00%"
    text = _all_text(at)
    assert "Estimated row removal per operation" in text
    assert "op-deduplicate-keys-order_id" in text
    assert "1 row(s), 8.33%" in text
    _assert_no_traceback(at)


def test_the_sidebar_always_shows_which_planner_is_active() -> None:
    at = _app()
    at.run()

    sidebar = " ".join(item.value for item in at.sidebar.markdown)
    assert "**Planner:**" in sidebar
    assert "deterministic" in sidebar
    captions = " ".join(item.value for item in at.sidebar.caption)
    assert "No provider is contacted." in captions
    _assert_no_traceback(at)


def test_the_approval_screen_names_the_planner_that_produced_the_plan() -> None:
    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")
    _submit_intent(at, key_columns=["order_id"], row_loss=10.0)
    _prepare(at)

    captions = " ".join(item.value for item in at.caption)
    assert "Planner:" in captions
    assert "deterministic" in captions
    _assert_no_traceback(at)


def test_the_approval_screen_renders_the_agent_tool_sequence() -> None:
    """With an agent-backed controller the trace panel shows the real sequence."""

    from datachef.agents import AgentPlanner, AgentReviewer
    from datachef.agents.plan_crew import CrewPlanResult
    from datachef.agents.tools import (
        DeduplicateByKeysArgs,
        PlanDraft,
        apply_operation_args,
        estimate_current_plan,
        finalize_plan,
        inspect_profile,
    )

    def scripted(ctx):
        draft = PlanDraft(context=ctx)
        issue = next(
            item.issue_id
            for item in ctx.diagnostic_report.issues
            if item.kind.value == "DUPLICATE_KEYS"
        )
        inspect_profile(draft)
        apply_operation_args(
            draft,
            "propose_deduplicate_by_keys",
            DeduplicateByKeysArgs(
                keys=["order_id"],
                diagnostic_issue_ids=[issue],
                rationale="Deterministic duplicate evidence.",
                expected_effect="Keep the first row per key.",
            ),
        )
        estimate_current_plan(draft)
        finalize_plan(draft, "Deduplicate orders by order_id.")
        return CrewPlanResult(plan=draft.build_plan(), draft=draft)

    class Registry(ui_state.AgentRegistry):
        def planner_factory(self):
            planner = AgentPlanner(environment={
                "DATACHEF_OFFLINE": "false",
                "GOOGLE_API_KEY": "k",
                "GEMINI_MODEL": "gemini-3.1-flash-lite",
            })
            planner._runner = scripted
            self.planner = planner
            return planner

    registry = Registry(live=True)
    controller = DataChefController(
        clock=lambda: NOW,
        planner_factory=registry.planner_factory,
        reviewer_factory=AgentReviewer,
    )
    at = _app(controller)
    at.session_state[ui_state.AGENT_REGISTRY] = registry
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")
    _submit_intent(at, key_columns=["order_id"], row_loss=10.0)
    _prepare(at)

    text = _all_text(at)
    assert "AI planner" in text
    assert (
        "Tool sequence: inspect_profile → propose_deduplicate_by_keys → "
        "estimate_current_plan → finalize_plan"
    ) in text
    assert "critic replied 8.33% row loss, no findings" in text
    assert "AGENT_PLAN_ACCEPTED" in text
    for leak in ("AIza", "C:\\", "Traceback", "gemini-3.1-flash-lite"):
        assert leak not in text
    _assert_no_traceback(at)


# A .env that tries as hard as it can to turn live mode on: live mode needs
# offline off, a credential, and a model name, so a hostile file supplies all
# three. This is the shape of a real developer .env, which is what made the
# original version of this test depend on an untracked local file.
_HOSTILE_DOTENV = "\n".join(
    (
        "DATACHEF_OFFLINE=false",
        "GOOGLE_API_KEY=not-a-real-key",
        "GEMINI_MODEL=gemini-3.1-flash-lite",
        "",
    )
)
_OFFLINE_DOTENV = "DATACHEF_OFFLINE=true\n"
_LIVE_MODE_VARIABLES = ("DATACHEF_OFFLINE", "GOOGLE_API_KEY", "GEMINI_MODEL")


@pytest.fixture
def local_dotenv(monkeypatch, tmp_path):
    """Point the shell's .env load at a file this test owns.

    The shell resolves .env from its own PROJECT_ROOT, and AppTest executes
    ``ui/app.py`` in a fresh namespace, so patching the imported module's
    PROJECT_ROOT would not reach it. Redirecting ``dotenv.load_dotenv`` does,
    and it forwards to the real loader with the caller's keywords intact, so
    ``override=False`` semantics and real file parsing are still what is under
    test -- only the path changes.

    Returns a callable that installs a given .env body, or no file at all.
    """

    import dotenv

    real_load_dotenv = dotenv.load_dotenv
    env_path = tmp_path / ".env"
    # load_dotenv writes into os.environ, and it may define variables this test
    # never registered with monkeypatch, so the whole mapping is restored.
    saved_environment = dict(os.environ)

    def install(body: str | None) -> None:
        if body is not None:
            env_path.write_text(body, encoding="utf-8")

        def load_from_this_file(dotenv_path=None, *args, **kwargs):
            del dotenv_path, args
            return real_load_dotenv(str(env_path), **kwargs)

        monkeypatch.setattr(dotenv, "load_dotenv", load_from_this_file)

    yield install

    os.environ.clear()
    os.environ.update(saved_environment)


@pytest.mark.parametrize(
    ("name", "dotenv_body"),
    (
        ("says true", _OFFLINE_DOTENV),
        ("says false and supplies a credential", _HOSTILE_DOTENV),
        ("is absent", None),
    ),
)
def test_env_loading_does_not_flip_the_offline_default(
    monkeypatch,
    local_dotenv,
    name: str,
    dotenv_body: str | None,
) -> None:
    """.env cannot override an explicit offline default, whatever it contains.

    The shell loads .env with ``override=False`` so a real environment variable
    stays authoritative. The result below is identical in all three conditions,
    which is what makes it independent of any developer's untracked .env.
    """

    del name
    monkeypatch.setenv("DATACHEF_OFFLINE", "true")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    local_dotenv(dotenv_body)

    at = _app()
    at.run()

    assert ui_state.live_mode_permitted() is False
    registry = at.session_state[ui_state.AGENT_REGISTRY]
    assert registry.live is False
    sidebar = " ".join(item.value for item in at.sidebar.markdown)
    assert "deterministic" in sidebar
    _assert_no_traceback(at)


@pytest.mark.parametrize("dotenv_body", (None, "", _OFFLINE_DOTENV))
def test_the_offline_default_holds_when_nothing_configures_live_mode(
    monkeypatch,
    local_dotenv,
    dotenv_body: str | None,
) -> None:
    """With nothing set and nothing in .env, the default is offline."""

    for variable in _LIVE_MODE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    local_dotenv(dotenv_body)

    at = _app()
    at.run()

    assert ui_state.live_mode_permitted() is False
    assert at.session_state[ui_state.AGENT_REGISTRY].live is False
    _assert_no_traceback(at)


def test_a_local_dotenv_is_what_activates_live_mode(
    monkeypatch,
    local_dotenv,
) -> None:
    """The counterpart that gives the guard above its meaning.

    Loading .env exists so a credential can reach os.environ at all; without
    this the shell could never run the crew. So .env *may* turn live mode on
    when nothing else has spoken -- it may not overrule something that has.
    Asserting both halves keeps a future change from closing the coupling by
    breaking the demo path instead.
    """

    for variable in _LIVE_MODE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    local_dotenv(_HOSTILE_DOTENV)

    at = _app()
    at.run()

    assert ui_state.live_mode_permitted() is True
    assert at.session_state[ui_state.AGENT_REGISTRY].live is True
    sidebar = " ".join(item.value for item in at.sidebar.markdown)
    assert "AI planner" in sidebar
    # Live mode is permitted, but no crew ran and no credential is displayed.
    for leak in ("not-a-real-key", "AIza"):
        assert leak not in " ".join(item.value for item in at.sidebar.markdown)
    _assert_no_traceback(at)


class _StubRegistry:
    """Stands in for the AgentRegistry so the shell can be driven offline.

    The live crew never runs in these tests; only the presentation path that
    reads what the crew reported is under test here.
    """

    def __init__(self, unsupported_requests: tuple[str, ...]) -> None:
        self.unsupported_requests = unsupported_requests
        self.planner = None
        self.reviewer = None
        self.live = False
        self.trace = None


def _at_approval(at: AppTest | None = None) -> AppTest:
    at = at or _app()
    _upload_and_diagnose(at)
    _submit_intent(at, key_columns=["order_id"])
    _prepare(at)
    return at


def _markdown_text(at: AppTest) -> str:
    return " ".join(element.value for element in at.markdown)


def test_results_screen_offers_the_pipeline_script_download() -> None:
    at = _pass_run()

    button = _widget(at, "download_button", "datachef_w_download_PIPELINE_SCRIPT_PY")

    assert button.label == "Download reusable pipeline script"
    assert len(at.download_button) == 7
    captions = " ".join(element.value for element in at.caption)
    assert "text/x-python" in captions
    assert "_pipeline.py" in captions
    _assert_no_traceback(at)


def test_approval_screen_states_what_the_agent_could_not_do() -> None:
    at = _at_approval()
    at.session_state[ui_state.AGENT_REGISTRY] = _StubRegistry(
        ("mean imputation for the amount column",)
    )

    at.run()

    text = _markdown_text(at)
    assert "Not in this plan" in text
    assert (
        "The objective also asked for mean imputation for the amount column. "
        "The offline allow-list has no executable operation for it, so it is "
        "not in this plan." in text
    )
    # A neutral statement of scope, not a warning.
    assert not any(
        "mean imputation" in element.value for element in at.warning
    )
    assert not any("mean imputation" in element.value for element in at.error)
    _assert_no_traceback(at)


def test_unsupported_request_statement_never_renders_agent_markup() -> None:
    at = _at_approval()
    at.session_state[ui_state.AGENT_REGISTRY] = _StubRegistry(
        ("<b>drop</b> the **imgUrl** [column](http://example.com)",)
    )

    at.run()

    text = _markdown_text(at)
    # Escaped, so the characters render as themselves rather than as markup.
    assert r"\<b\>drop\</b\>" in text
    assert r"\*\*imgUrl\*\*" in text
    _assert_no_traceback(at)


def test_no_screen_ever_enables_unsafe_html() -> None:
    """An AST check: prose about the flag must not read as the flag itself.

    Scoped to the product shell screens, which are the surfaces that render
    plan evidence and agent-supplied text. ``ui/ingestion_view.py`` belongs to
    the separate ingestion-agent view and does pass the flag today; that is a
    pre-existing condition outside this slice, not something this test blesses.
    """

    offenders: list[str] = []
    for path in (REPO_ROOT / "ui" / "screens").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg == "unsafe_allow_html":
                    offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_an_empty_unsupported_request_list_states_nothing() -> None:
    at = _at_approval()
    at.session_state[ui_state.AGENT_REGISTRY] = _StubRegistry(())

    at.run()

    assert "Not in this plan" not in _markdown_text(at)
    _assert_no_traceback(at)


def test_unsupported_requests_do_not_change_the_plan_or_the_approval_gate() -> None:
    baseline = _at_approval()
    baseline_plan = _controller(baseline).session.workflow_runtime.state
    assert baseline_plan.transformation_plan is not None

    at = _at_approval()
    at.session_state[ui_state.AGENT_REGISTRY] = _StubRegistry(("a column drop",))
    at.run()
    _approve_and_execute(at)

    session = _controller(at).session
    # Same plan, still approvable, still reaching PASS with the full bundle.
    assert (
        session.workflow_runtime.state.transformation_plan.plan_id
        == baseline_plan.transformation_plan.plan_id
    )
    assert session.workflow_runtime.state.stage is WorkflowStage.QA_PASSED
    assert len(at.download_button) == 7
    _assert_no_traceback(at)


def test_the_intent_screen_no_longer_offers_a_separate_dedup_request() -> None:
    at = _app()
    _upload_and_diagnose(at)

    # The key-column question remains; the request that restated it is gone.
    assert _has(at, "multiselect", ui_state.KEY_COLUMNS_WIDGET)
    assert _has(at, "multiselect", ui_state.CAST_REQUEST_WIDGET)
    assert not _has(at, "multiselect", ui_state.DEDUP_REQUEST_WIDGET)
    labels = [element.label for element in at.multiselect]
    assert "Deduplicate rows by these keys" not in labels
    assert "Which column tells you two rows are the same record?" in labels
    _assert_no_traceback(at)


def test_cast_requests_still_reconcile_after_the_dedup_field_is_gone() -> None:
    at = _app()
    _upload_and_diagnose(at, content=CAST_FAILURE_JSON, filename="orders.json", mime="application/json")
    _submit_intent(at, key_columns=["order_id"], cast_columns=["amount_text"])

    session = _controller(at).session
    requests = session.requested_transformations
    assert [item.request_id for item in requests] == ["request-cast-amount_text"]
    assert all(
        item.operation_type.value == "CAST_COLUMN" for item in requests
    )
    _prepare(at)
    plan = _controller(at).session.workflow_runtime.state.transformation_plan
    # The planner still accounts for the typed cast request.
    assert any(
        operation.operation_type.value == "CAST_COLUMN"
        for operation in plan.operations
    )
    _assert_no_traceback(at)


def test_reset_still_clears_cleanly_without_the_dedup_widget() -> None:
    at = _app()
    _upload_and_diagnose(at)
    _widget(at, "multiselect", ui_state.CAST_REQUEST_WIDGET).set_value(["amount"])
    at.run()
    assert at.session_state[ui_state.CAST_REQUEST_WIDGET] == ["amount"]

    _widget(at, "button", ui_state.RESET_WIDGET).click()
    at.run()

    # Reset lands back on Upload, so the intent widgets are not instantiated
    # and their cleared keys are simply gone from session state.
    live_keys = set(at.session_state.filtered_state)
    assert ui_state.CAST_REQUEST_WIDGET not in live_keys
    assert ui_state.DEDUP_REQUEST_WIDGET not in live_keys
    assert _controller(at).session.source is None
    assert _controller(at).session.requested_transformations == ()
    _assert_no_traceback(at)
