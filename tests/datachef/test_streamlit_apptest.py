from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from datachef.application import DataChefController, ScreenId
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


def test_stage_indicator_renders_all_six_stages_and_marks_the_current_one() -> None:
    at = _app()
    at.run()

    sidebar_text = " ".join(item.value for item in at.sidebar.markdown)
    for label in ("1 · Upload", "2 · Objective", "3 · Plan", "4 · Approve",
                  "5 · Quality", "6 · Results"):
        assert label in sidebar_text
    assert "**➡️ 1 · Upload**" in sidebar_text
    assert "◻️ 6 · Results" in sidebar_text
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


def test_key_question_is_plain_language_and_follows_required_columns() -> None:
    at = _app()
    _upload_and_diagnose(at, DEMO_CSV, "orders.csv")

    labels = [str(item.label) for item in at.multiselect]
    required_label = "Columns that must survive"
    key_label = "Which column tells you two rows are the same record?"
    assert required_label in labels
    assert key_label in labels
    assert labels.index(required_label) < labels.index(key_label)
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
    required_at = multiselects.index("Columns that must survive")
    key_at = multiselects.index("Which column tells you two rows are the same record?")
    cast_at = multiselects.index("Cast these columns to numeric")
    dedup_at = multiselects.index("Deduplicate rows by these keys")
    suggested_at = multiselects.index(
        "Deterministic suggested questions to carry into the dashboard"
    )
    # Cleaning fields keep their order; both question inputs come last.
    assert required_at < key_at < cast_at < dedup_at < suggested_at
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
    assert len(at.download_button) == 6
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
    assert len(at.download_button) == 6


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


def test_env_loading_does_not_flip_the_offline_default(monkeypatch) -> None:
    monkeypatch.delenv("DATACHEF_OFFLINE", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    at = _app()
    at.run()

    from ui.state import live_mode_permitted

    assert live_mode_permitted() is False
    registry = at.session_state[ui_state.AGENT_REGISTRY]
    assert registry.live is False
    sidebar = " ".join(item.value for item in at.sidebar.markdown)
    assert "deterministic" in sidebar
    _assert_no_traceback(at)
