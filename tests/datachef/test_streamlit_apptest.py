from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from datachef.application import DataChefController
from datachef.contracts import QAStatus, WorkflowStage
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
    return at


def _submit_intent(
    at: AppTest,
    *,
    key_columns: list[str] | None = None,
    cast_columns: list[str] | None = None,
    dedup_keys: list[str] | None = None,
    row_loss: float = 50.0,
    goal: str | None = None,
) -> AppTest:
    if goal is not None:
        _widget(at, "text_area", ui_state.GOAL_WIDGET).set_value(goal)
    if key_columns:
        _widget(at, "multiselect", ui_state.KEY_COLUMNS_WIDGET).set_value(key_columns)
    if cast_columns:
        _widget(at, "multiselect", ui_state.CAST_REQUEST_WIDGET).set_value(cast_columns)
    if dedup_keys:
        _widget(at, "multiselect", ui_state.DEDUP_REQUEST_WIDGET).set_value(dedup_keys)
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


def test_happy_path_reaches_pass_with_full_bundle_and_dashboard() -> None:
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
        "datachef_w_download_MANIFEST_JSON",
    ]
    assert "DataChef Dashboard" in [header.value for header in at.header]
    assert tuple(item.kind.value for item in session.command_history) == (
        "PLAN_PREPARATION",
        "HUMAN_DECISION",
        "EXECUTION",
    )
    _assert_no_traceback(at)


def test_page_refresh_after_pass_performs_no_further_work() -> None:
    at = _pass_run()
    before = _controller(at).session

    at.run()
    at.run()

    after = _controller(at).session
    assert after.revision == before.revision
    assert after.command_history == before.command_history
    assert len(at.download_button) == 6
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
    assert session.screen.value == "QA"
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
    assert len(at.download_button) == 6
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


def test_full_reset_clears_state_and_rotates_the_uploader_key() -> None:
    at = _app()
    at.run()
    generation = _controller(at).session.uploader_generation
    assert _has(at, "file_uploader", ui_state.uploader_key(generation))
    _pass_run(at)
    assert _controller(at).session.uploader_generation == generation
    assert len(at.download_button) == 6

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
    assert len(at.download_button) == 6
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
    """Drive the constant-key dataset to a validation-refused plan."""

    at = _app()
    _upload_and_diagnose(at, CONSTANT_KEY_CSV, "catalogue.csv")
    _submit_intent(at, cast_columns=cast_columns, row_loss=0.0)
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
        + [item.value for item in at.markdown]
        + [item.value for item in at.caption]
        + [f"{item.label} {item.value}" for item in at.metric]
    )


def test_row_loss_refusal_shows_the_finding_operation_and_both_thresholds() -> None:
    at = _refused_plan(cast_columns=["price"])

    text = _all_text(at)
    assert "ROW_LOSS_THRESHOLD" in text
    assert "CUMULATIVE_ROW_LOSS_THRESHOLD" in text
    assert "Estimated row loss exceeds the user's approved threshold." in text
    assert "op-deduplicate-keys-category_id" in text
    assert "88.89%" in text
    assert "8 row(s)" in text
    labels = {item.label: item.value for item in at.metric}
    assert labels["Estimated cumulative row loss"] == "88.89%"
    assert labels["Your acceptable row loss"] == "0.00%"
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
    at = _refused_plan(cast_columns=["price"])
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
    at = _refused_plan(cast_columns=["price"])

    _assert_locked_down(at)
    assert _controller(at).session.workflow_runtime.gold_dataframe is None


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
