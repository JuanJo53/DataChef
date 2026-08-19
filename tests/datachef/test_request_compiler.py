"""The free-form objective must become typed requests, and then real operations.

The gap this covers: the user typed an ML-preparation objective naming specific
imputations and column drops, and the run produced an empty plan because nothing
translated prose into typed requests. Compilation happens locally -- the
objective text and the raw frame never leave this process -- and the deterministic
planner then has to account for every compiled request.
"""

from __future__ import annotations

import pandas as pd
import pytest

from datachef.application import (
    ArtifactSet,
    CsvParserOptions,
    DataChefController,
    RequestedTransformation,
    UploadFormat,
    UploadRequest,
)
from datachef.application.request_compiler import compile_requests, measure_columns
from datachef.contracts import (
    DownstreamUse,
    DropColumnParameters,
    HumanDecision,
    ImputeMissingParameters,
    ImputeStrategy,
    OperationType,
    QAStatus,
    UserIntent,
    WorkflowStage,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.privacy import build_provider_planning_payload

# The objective from the failing session, verbatim.
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

# title repeats "t1", so a mode exists and the conditional imputes.
MODE_CSV = (
    b"asin,title,stars,price,category_id,boughtInLastMonth\n"
    b"a1,t1,4.5,10.0,104,0\n"
    b"a2,,,20.0,104,0\n"
    b"a3,,4.0,,104,\n"
    b"a4,,3.5,40.0,104,\n"
    b"a1,t1,4.5,10.0,104,\n"
)


def _frame(*, unique_titles: bool) -> pd.DataFrame:
    titles = ["t1", None, None, None, "t5"] if unique_titles else ["t1", None, None, None, "t1"]
    return pd.DataFrame(
        {
            "asin": ["a1", "a2", "a3", "a4", "a1"],
            "title": titles,
            "stars": [4.5, None, 4.0, 3.5, 4.5],
            "price": [10.0, 20.0, None, 40.0, 10.0],
            "category_id": [104, 104, 104, 104, 104],
            "boughtInLastMonth": [0, 0, None, None, None],
        }
    )


def _compile(objective: str, frame: pd.DataFrame):
    report = diagnose_raw_dataframe(frame, selected_key_columns=("asin",))
    return compile_requests(objective, frame, report)


def _by_column(requests) -> dict[str, RequestedTransformation]:
    return {request.target_columns[0]: request for request in requests}


# ---------------------------------------------------------------------------
# B. the whole requested scenario compiles to the expected typed requests
# ---------------------------------------------------------------------------


def test_the_ml_objective_compiles_to_the_expected_typed_requests() -> None:
    requests = _compile(ML_OBJECTIVE, _frame(unique_titles=False))

    assert [
        (request.operation_type, request.target_columns) for request in requests
    ] == [
        (OperationType.IMPUTE_MISSING, ("title",)),
        (OperationType.IMPUTE_MISSING, ("stars",)),
        (OperationType.IMPUTE_MISSING, ("price",)),
        (OperationType.DROP_COLUMN, ("category_id",)),
        (OperationType.DROP_COLUMN, ("boughtInLastMonth",)),
        (OperationType.DEDUPLICATE_BY_KEYS, ("asin",)),
    ]


def test_compilation_is_deterministic() -> None:
    frame = _frame(unique_titles=False)
    first = _compile(ML_OBJECTIVE, frame)
    second = _compile(ML_OBJECTIVE, frame)

    assert [item.model_dump(mode="json") for item in first] == [
        item.model_dump(mode="json") for item in second
    ]


# ---------------------------------------------------------------------------
# C-H. each clause of the objective
# ---------------------------------------------------------------------------


def test_title_is_dropped_when_over_threshold_and_no_mode_exists() -> None:
    """60% null and every remaining value unique -> drop, per the request."""

    frame = _frame(unique_titles=True)
    facts = measure_columns(frame)["title"]
    assert facts.null_pct > 40.0
    assert facts.mode_exists is False

    requests = _by_column(_compile(ML_OBJECTIVE, frame))

    assert requests["title"].operation_type is OperationType.DROP_COLUMN


def test_title_is_imputed_with_mode_when_a_mode_exists() -> None:
    frame = _frame(unique_titles=False)
    facts = measure_columns(frame)["title"]
    assert facts.null_pct > 40.0
    assert facts.mode_exists is True

    requests = _by_column(_compile(ML_OBJECTIVE, frame))

    request = requests["title"]
    assert request.operation_type is OperationType.IMPUTE_MISSING
    assert request.parameters.strategy is ImputeStrategy.MODE


def test_mode_existence_means_a_value_actually_repeats() -> None:
    """The explicit, deterministic reading of "there's no mode".

    ``Series.mode()`` is empty only for an all-null column and otherwise returns
    every value when each occurs once, which would let a column of unique titles
    claim a mode. This is the stricter reading the request intends.
    """

    unique = pd.DataFrame({"a": ["x", "y", "z", None]})
    repeated = pd.DataFrame({"a": ["x", "x", "z", None]})
    empty = pd.DataFrame({"a": [None, None]})

    assert measure_columns(unique)["a"].mode_exists is False
    assert measure_columns(repeated)["a"].mode_exists is True
    assert measure_columns(empty)["a"].mode_exists is False
    # pandas itself would disagree on the first case; that is the point.
    assert not unique["a"].mode(dropna=True).empty


@pytest.mark.parametrize(
    ("column", "strategy"),
    (("stars", ImputeStrategy.MEAN), ("price", ImputeStrategy.MEDIAN)),
)
def test_named_strategies_compile_for_their_named_columns(column, strategy) -> None:
    requests = _by_column(_compile(ML_OBJECTIVE, _frame(unique_titles=False)))

    request = requests[column]
    assert request.operation_type is OperationType.IMPUTE_MISSING
    assert request.parameters.strategy is strategy


def test_an_unconditional_drop_compiles() -> None:
    requests = _by_column(_compile(ML_OBJECTIVE, _frame(unique_titles=False)))

    assert requests["category_id"].operation_type is OperationType.DROP_COLUMN


def test_the_zero_and_null_condition_drops_only_when_both_hold() -> None:
    frame = _frame(unique_titles=False)
    facts = measure_columns(frame)["boughtInLastMonth"]
    assert facts.null_pct > 40.0 and facts.zero_count > 0

    dropped = _by_column(_compile(ML_OBJECTIVE, frame))
    assert dropped["boughtInLastMonth"].operation_type is OperationType.DROP_COLUMN

    # No zeros: the condition is not met, so the column is left alone.
    without_zeros = frame.copy(deep=True)
    without_zeros["boughtInLastMonth"] = [7, 9, None, None, None]
    assert measure_columns(without_zeros)["boughtInLastMonth"].zero_count == 0
    kept = _by_column(_compile(ML_OBJECTIVE, without_zeros))
    assert "boughtInLastMonth" not in kept


def test_explicit_deduplication_nominates_the_named_key() -> None:
    requests = _compile(ML_OBJECTIVE, _frame(unique_titles=False))

    dedup = [
        request
        for request in requests
        if request.operation_type is OperationType.DEDUPLICATE_BY_KEYS
    ]
    assert len(dedup) == 1
    assert dedup[0].parameters.keys == ("asin",)


def test_modelling_prose_is_not_read_as_the_mode_strategy() -> None:
    """"ML modelling" and "train a model" must not match the mode strategy."""

    frame = _frame(unique_titles=False)
    requests = _by_column(
        _compile(
            "Prepare this table for ML modelling to train a model to predict the "
            "price column, impute the column price using the median",
            frame,
        )
    )

    assert requests["price"].parameters.strategy is ImputeStrategy.MEDIAN


def test_an_objective_naming_nothing_supported_compiles_to_nothing() -> None:
    frame = _frame(unique_titles=False)

    assert _compile("Please make the data beautiful and insightful", frame) == ()
    assert _compile("", frame) == ()


def test_compilation_never_targets_a_column_that_is_being_dropped() -> None:
    """A drop wins: imputing a column that is about to disappear is wasted work."""

    frame = _frame(unique_titles=True)
    requests = _by_column(_compile(ML_OBJECTIVE, frame))

    assert requests["title"].operation_type is OperationType.DROP_COLUMN
    assert not any(
        request.operation_type is OperationType.IMPUTE_MISSING
        and request.target_columns == ("title",)
        for request in _compile(ML_OBJECTIVE, frame)
    )


# ---------------------------------------------------------------------------
# A, K, L, M. the whole trusted flow, from prose to QA-passing gold
# ---------------------------------------------------------------------------


def _controller_for(objective: str, csv: bytes = MODE_CSV) -> DataChefController:
    controller = DataChefController()
    assert controller.load_upload(
        UploadRequest(
            content=csv,
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(encoding="utf-8-sig"),
        )
    ).changed
    assert controller.diagnose().changed
    controller.submit_intent(
        UserIntent(
            intent_id="intent-ml",
            user_goal=objective,
            downstream_use=DownstreamUse.ANALYSIS,
            selected_key_columns=("asin",),
            acceptable_row_loss_pct=50,
        ),
        (),
    )
    return controller


def test_submit_intent_compiles_the_objective_into_session_requests() -> None:
    controller = _controller_for(ML_OBJECTIVE)

    kinds = [
        (request.operation_type, request.target_columns)
        for request in controller.session.requested_transformations
    ]

    assert (OperationType.IMPUTE_MISSING, ("stars",)) in kinds
    assert (OperationType.DROP_COLUMN, ("category_id",)) in kinds
    assert (OperationType.DEDUPLICATE_BY_KEYS, ("asin",)) in kinds


def test_the_plan_contains_every_requested_operation_and_no_blocking_finding() -> None:
    """A. the supported operations are planned, not reported as unsupported."""

    controller = _controller_for(ML_OBJECTIVE)

    assert controller.prepare_plan(command_id="plan").code == "PLAN_AWAITING_APPROVAL"
    plan = controller.session.workflow_runtime.state.transformation_plan
    planned = {
        (operation.operation_type, operation.target_columns)
        for operation in plan.operations
    }

    for expected in (
        (OperationType.IMPUTE_MISSING, ("title",)),
        (OperationType.IMPUTE_MISSING, ("stars",)),
        (OperationType.IMPUTE_MISSING, ("price",)),
        (OperationType.DROP_COLUMN, ("category_id",)),
        (OperationType.DROP_COLUMN, ("boughtInLastMonth",)),
        (OperationType.DEDUPLICATE_BY_KEYS, ("asin",)),
    ):
        assert expected in planned, expected
    # Reconciliation is satisfied: nothing requested is missing from the plan.
    assert [
        finding.code for finding in controller.session.findings if finding.blocking
    ] == []


def test_requested_operations_cite_the_user_request_not_a_diagnostic_issue() -> None:
    controller = _controller_for(ML_OBJECTIVE)
    controller.prepare_plan(command_id="plan")
    plan = controller.session.workflow_runtime.state.transformation_plan

    imputations = [
        operation
        for operation in plan.operations
        if operation.operation_type is OperationType.IMPUTE_MISSING
    ]
    assert imputations
    for operation in imputations:
        assert operation.user_requirement_ids
        assert operation.diagnostic_issue_ids == ()


def test_no_operation_targets_a_column_an_earlier_operation_dropped() -> None:
    """The property the old "drops last" rule existed to protect.

    Ordering now follows the objective, so drops are no longer unconditionally
    final. What must still hold is that nothing operates on a column that has
    already been removed.
    """

    controller = _controller_for(ML_OBJECTIVE)
    controller.prepare_plan(command_id="plan")
    plan = controller.session.workflow_runtime.state.transformation_plan

    removed: set[str] = set()
    for operation in plan.operations:
        assert removed.isdisjoint(operation.target_columns), operation.operation_id
        if operation.operation_type is OperationType.DROP_COLUMN:
            removed.update(operation.target_columns)


def test_execution_is_still_blocked_until_a_human_approves() -> None:
    """K. the approval gate is untouched by request compilation."""

    controller = _controller_for(ML_OBJECTIVE)
    controller.prepare_plan(command_id="plan")

    refused = controller.execute_current_plan(command_id="execute")

    assert not refused.changed
    runtime = controller.session.workflow_runtime
    assert runtime.state.stage is not WorkflowStage.QA_PASSED
    assert runtime.gold_dataframe is None


def test_the_objective_runs_end_to_end_to_qa_passing_gold() -> None:
    """L, M. the executor really performs the requested operations."""

    controller = _controller_for(ML_OBJECTIVE)
    controller.prepare_plan(command_id="plan")
    assert controller.record_human_decision(
        HumanDecision.APPROVE, command_id="approve"
    ).changed
    assert controller.execute_current_plan(command_id="execute").changed

    state = controller.session.workflow_runtime.state
    assert state.stage is WorkflowStage.QA_PASSED
    assert state.qa_report.status is QAStatus.PASS
    assert [
        result.invariant_id
        for result in state.qa_report.invariant_results
        if result.status.value != "PASS"
    ] == []

    gold = controller.session.workflow_runtime.gold_dataframe
    # Dropped.
    assert "category_id" not in gold.columns
    assert "boughtInLastMonth" not in gold.columns
    # Deduplicated on asin.
    assert len(gold) == 4
    assert gold["asin"].is_unique
    # Imputed: no nulls remain in the three named columns.
    for column in ("title", "stars", "price"):
        assert gold[column].isna().sum() == 0
    # Deduplication is last, so the mean and median still see the duplicate row:
    # stars 16.5 / 4 = 4.125 and price median of 10, 10, 20, 40 = 15.0.
    assert gold["stars"].tolist() == [4.5, 4.125, 4.0, 3.5]
    assert gold["price"].tolist() == [10.0, 20.0, 15.0, 40.0]

    bundle = controller.build_artifacts()
    assert isinstance(bundle, ArtifactSet)
    assert len(bundle.artifacts()) == 7


def test_a_multi_drop_plan_passes_the_structure_invariant() -> None:
    """Regression: two drops in one plan must not fail each other's invariant."""

    controller = _controller_for(ML_OBJECTIVE)
    controller.prepare_plan(command_id="plan")
    controller.record_human_decision(HumanDecision.APPROVE, command_id="approve")
    controller.execute_current_plan(command_id="execute")

    report = controller.session.workflow_runtime.state.qa_report
    structure = [
        result
        for result in report.invariant_results
        if result.kind.value == "DROPPED_COLUMN_STRUCTURE"
    ]
    assert len(structure) == 2
    for result in structure:
        assert result.status.value == "PASS"
        # The whole-plan arithmetic: six columns in, two dropped.
        assert result.observed_value == result.expected_value == 4


# ---------------------------------------------------------------------------
# I, J. the privacy boundary is unchanged by any of this
# ---------------------------------------------------------------------------


def test_the_provider_payload_never_carries_the_objective_text() -> None:
    controller = _controller_for(ML_OBJECTIVE)
    controller.prepare_plan(command_id="plan")
    context = controller.session.workflow_runtime.state.planning_context

    payload = build_provider_planning_payload(context)
    serialized = payload.model_dump_json()

    for fragment in (
        "ML modelling",
        "predict the price",
        "there's no mode",
        "boughtInLastMonth if it has",
        ML_OBJECTIVE[:40],
    ):
        assert fragment not in serialized, fragment
    assert payload.privacy_manifest.free_form_text_included is False
    assert payload.privacy_manifest.raw_rows_included is False
    assert payload.privacy_manifest.row_samples_included is False


def test_the_provider_payload_carries_aggregate_statistics_only() -> None:
    """J. counts, never cell values."""

    controller = _controller_for(ML_OBJECTIVE)
    controller.prepare_plan(command_id="plan")
    payload = build_provider_planning_payload(
        controller.session.workflow_runtime.state.planning_context
    )
    serialized = payload.model_dump_json()

    statistics = {item.column: item for item in payload.column_statistics}
    assert statistics["boughtInLastMonth"].zero_count == 2
    assert statistics["stars"].null_count == 1
    # No cell value from the frame appears anywhere in the payload.
    for value in ("t1", "t5", "a1", "a2", "104"):
        assert f'"{value}"' not in serialized, value


def test_compiled_requests_carry_no_free_form_text() -> None:
    requests = _compile(ML_OBJECTIVE, _frame(unique_titles=False))

    for request in requests:
        serialized = request.model_dump_json()
        assert "modelling" not in serialized
        assert "impute the" not in serialized


# ---------------------------------------------------------------------------
# An explicit caller still wins over compilation
# ---------------------------------------------------------------------------


def test_explicitly_supplied_requests_merge_with_the_compiled_objective() -> None:
    controller = DataChefController()
    controller.load_upload(
        UploadRequest(
            content=MODE_CSV,
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(encoding="utf-8-sig"),
        )
    )
    controller.diagnose()
    explicit = (
        RequestedTransformation(
            request_id="request-explicit",
            operation_type=OperationType.DROP_COLUMN,
            target_columns=("category_id",),
            parameters=DropColumnParameters(),
        ),
    )

    controller.submit_intent(
        UserIntent(
            intent_id="intent-explicit",
            user_goal=ML_OBJECTIVE,
            selected_key_columns=("asin",),
            acceptable_row_loss_pct=50,
        ),
        explicit,
    )

    requests = controller.session.requested_transformations
    ids = [request.request_id for request in requests]
    # The caller's request survives verbatim...
    assert "request-explicit" in ids
    # ...it is the only one for category_id, so compilation did not duplicate it...
    assert sum(
        1 for request in requests if request.target_columns == ("category_id",)
    ) == 1
    # ...and the rest of the written objective is still honoured rather than
    # being discarded because one typed request was supplied.
    kinds = {(request.operation_type, request.target_columns) for request in requests}
    assert (OperationType.IMPUTE_MISSING, ("stars",)) in kinds
    assert (OperationType.DEDUPLICATE_BY_KEYS, ("asin",)) in kinds


def test_a_widened_request_still_refuses_an_unsupported_operation() -> None:
    with pytest.raises(ValueError):
        RequestedTransformation(
            request_id="request-bad",
            operation_type=OperationType.TRIM_WHITESPACE,
            target_columns=("title",),
            parameters=DropColumnParameters(),
        )
    with pytest.raises(ValueError):
        # Imputation targets exactly one column.
        RequestedTransformation(
            request_id="request-bad-2",
            operation_type=OperationType.IMPUTE_MISSING,
            target_columns=("title", "stars"),
            parameters=ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
        )


# ---------------------------------------------------------------------------
# The real Streamlit path, not just the controller API
# ---------------------------------------------------------------------------


def _drive_ui(*, cast_columns: list[str] | None = None):
    """Drive ui/app.py exactly as a user does on the Objective screen."""

    from streamlit.testing.v1 import AppTest

    from ui import state as ui_state

    app = str(REPO_ROOT / "ui" / "app.py")

    def widget(at, kind, key):
        for element in getattr(at, kind):
            if getattr(element, "key", None) == key:
                return element
        raise KeyError(f"{kind}:{key}")

    at = AppTest.from_file(app, default_timeout=180)
    at.run()
    at.file_uploader[0].set_value(("amazon.csv", MODE_CSV, "text/csv"))
    at.run()
    widget(at, "button", ui_state.DIAGNOSE_WIDGET).click()
    at.run()
    widget(at, "button", ui_state.CONTINUE_TO_INTENT_WIDGET).click()
    at.run()
    widget(at, "text_area", ui_state.GOAL_WIDGET).set_value(ML_OBJECTIVE)
    widget(at, "multiselect", ui_state.KEY_COLUMNS_WIDGET).set_value(["asin"])
    widget(at, "slider", ui_state.ROW_LOSS_WIDGET).set_value(50.0)
    if cast_columns:
        widget(at, "multiselect", ui_state.CAST_REQUEST_WIDGET).set_value(cast_columns)
    widget(at, "button", ui_state.SUBMIT_INTENT_WIDGET).click()
    at.run()
    widget(at, "button", ui_state.PREPARE_PLAN_WIDGET).click()
    at.run()
    return at, ui_state, widget


REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def test_the_streamlit_objective_screen_reaches_the_request_compiler() -> None:
    """The typed objective must become planned operations through the real UI."""

    at, ui_state, widget = _drive_ui()
    controller = at.session_state[ui_state.CONTROLLER]

    planned = {
        (operation.operation_type, operation.target_columns)
        for operation in controller.session.workflow_runtime.state.transformation_plan.operations
    }
    for expected in (
        (OperationType.IMPUTE_MISSING, ("title",)),
        (OperationType.IMPUTE_MISSING, ("stars",)),
        (OperationType.IMPUTE_MISSING, ("price",)),
        (OperationType.DROP_COLUMN, ("category_id",)),
        (OperationType.DROP_COLUMN, ("boughtInLastMonth",)),
        (OperationType.DEDUPLICATE_BY_KEYS, ("asin",)),
    ):
        assert expected in planned, expected

    # The approval screen shows the plan, not a refusal.
    rendered = " ".join(element.value for element in at.markdown)
    assert "Not in this plan" not in rendered
    assert [f.code for f in controller.session.findings if f.blocking] == []
    assert len(at.exception) == 0, [item.value for item in at.exception]


def test_the_streamlit_path_still_requires_approval_then_produces_gold() -> None:
    at, ui_state, widget = _drive_ui()
    controller = at.session_state[ui_state.CONTROLLER]

    # Nothing executed yet.
    assert controller.session.workflow_runtime.gold_dataframe is None

    widget(at, "button", ui_state.APPROVE_WIDGET).click()
    at.run()
    widget(at, "button", ui_state.EXECUTE_WIDGET).click()
    at.run()

    runtime = at.session_state[ui_state.CONTROLLER].session.workflow_runtime
    assert runtime.state.stage is WorkflowStage.QA_PASSED
    assert runtime.state.qa_report.status is QAStatus.PASS
    gold = runtime.gold_dataframe
    assert list(gold.columns) == ["asin", "title", "stars", "price"]
    assert len(gold) == 4
    assert int(gold[["title", "stars", "price"]].isna().sum().sum()) == 0
    assert len(at.download_button) == 7
    assert len(at.exception) == 0


def test_a_typed_cast_selection_does_not_discard_the_written_objective() -> None:
    """Regression: ticking one checkbox used to silently drop the whole goal."""

    at, ui_state, _ = _drive_ui(cast_columns=["stars"])
    controller = at.session_state[ui_state.CONTROLLER]

    kinds = {
        (request.operation_type, request.target_columns)
        for request in controller.session.requested_transformations
    }
    # The explicit request survives...
    assert (OperationType.CAST_COLUMN, ("stars",)) in kinds
    # ...and so does everything the objective asked for.
    assert (OperationType.IMPUTE_MISSING, ("price",)) in kinds
    assert (OperationType.DROP_COLUMN, ("category_id",)) in kinds
    assert (OperationType.DEDUPLICATE_BY_KEYS, ("asin",)) in kinds

    planned = {
        (operation.operation_type, operation.target_columns)
        for operation in controller.session.workflow_runtime.state.transformation_plan.operations
    }
    assert (OperationType.DROP_COLUMN, ("category_id",)) in planned


def test_an_explicit_request_wins_over_a_compiled_one_for_the_same_column() -> None:
    controller = DataChefController()
    controller.load_upload(
        UploadRequest(
            content=MODE_CSV,
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(encoding="utf-8-sig"),
        )
    )
    controller.diagnose()
    explicit = (
        RequestedTransformation(
            request_id="request-explicit-price",
            operation_type=OperationType.IMPUTE_MISSING,
            target_columns=("price",),
            parameters=ImputeMissingParameters(strategy=ImputeStrategy.MEAN),
        ),
    )

    controller.submit_intent(
        UserIntent(
            intent_id="intent-merge",
            user_goal=ML_OBJECTIVE,
            selected_key_columns=("asin",),
            acceptable_row_loss_pct=50,
        ),
        explicit,
    )

    price = [
        request
        for request in controller.session.requested_transformations
        if request.target_columns == ("price",)
    ]
    # One request for price, and it is the caller's MEAN, not the objective's MEDIAN.
    assert len(price) == 1
    assert price[0].parameters.strategy is ImputeStrategy.MEAN


# ---------------------------------------------------------------------------
# The live-path bug the first manual smoke test exposed.
#
# The deterministic planner honoured compiled requests, but the live CrewAI
# planner builds its own plan and, when that plan validated, AgentPlanner
# returned it verbatim. The compiled requests were silently discarded and every
# one of them came back as a blocking REQUEST_NOT_PLANNED. Priming the
# deterministic fallback did not help, because the fallback only runs when the
# crew fails. These tests pin the enforcement layer that fixes it.
# ---------------------------------------------------------------------------

# Exactly what the real app produced: CONSTANT everywhere, including columns the
# user asked to drop.
CREW_CONSTANTS = {
    "title": "Unknown",
    "stars": 0.0,
    "price": 0.0,
    "category_id": 0,
    "boughtInLastMonth": 0,
}


class _CrewLikePlanner:
    """Stands in for the live crew: a valid plan that ignores the objective."""

    def __init__(self) -> None:
        self.calls = 0
        self.trace = None

    def propose(self, context, *, attempt):
        from datachef.contracts import RiskLevel, TransformationOperation
        from datachef.planning.plan import create_transformation_plan

        self.calls += 1
        operations = []
        for index, (column, value) in enumerate(CREW_CONSTANTS.items(), start=1):
            issue = next(
                (
                    item.issue_id
                    for item in context.diagnostic_report.issues
                    if column in item.affected_columns
                ),
                None,
            )
            if issue is None:
                continue
            operations.append(
                TransformationOperation(
                    operation_id=f"op-{index:03d}-impute_missing",
                    operation_type=OperationType.IMPUTE_MISSING,
                    target_columns=(column,),
                    parameters=ImputeMissingParameters(
                        strategy=ImputeStrategy.CONSTANT, constant_value=value
                    ),
                    diagnostic_issue_ids=(issue,),
                    rationale="crew choice",
                    expected_effect="crew effect",
                    risk=RiskLevel.MEDIUM,
                    requires_human_approval=True,
                )
            )
        return create_transformation_plan(
            dataset_id=context.dataset_identity.dataset_id,
            dataset_fingerprint=context.dataset_identity.fingerprint,
            version=1,
            operations=tuple(operations),
            summary="crew plan",
        )


def _crew_controller(objective: str = ML_OBJECTIVE) -> DataChefController:
    controller = DataChefController(planner_factory=_CrewLikePlanner)
    controller.load_upload(
        UploadRequest(
            content=MODE_CSV,
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(encoding="utf-8-sig"),
        )
    )
    controller.diagnose()
    controller.submit_intent(
        UserIntent(
            intent_id="intent-crew",
            user_goal=objective,
            selected_key_columns=("asin",),
            acceptable_row_loss_pct=50,
        ),
        (),
    )
    controller.prepare_plan(command_id="plan")
    return controller


def _plan_of(controller):
    return controller.session.workflow_runtime.state.transformation_plan


def test_a_crew_plan_that_ignores_the_objective_is_reconciled_not_accepted() -> None:
    """The regression: arbitrary CONSTANT imputations must not survive."""

    controller = _crew_controller()
    plan = _plan_of(controller)

    signatures = {
        (operation.operation_type, operation.target_columns)
        for operation in plan.operations
    }
    for expected in (
        (OperationType.IMPUTE_MISSING, ("title",)),
        (OperationType.IMPUTE_MISSING, ("stars",)),
        (OperationType.IMPUTE_MISSING, ("price",)),
        (OperationType.DROP_COLUMN, ("category_id",)),
        (OperationType.DROP_COLUMN, ("boughtInLastMonth",)),
        (OperationType.DEDUPLICATE_BY_KEYS, ("asin",)),
    ):
        assert expected in signatures, expected

    strategies = {
        operation.target_columns[0]: operation.parameters.strategy
        for operation in plan.operations
        if operation.operation_type is OperationType.IMPUTE_MISSING
    }
    assert strategies == {
        "title": ImputeStrategy.MODE,
        "stars": ImputeStrategy.MEAN,
        "price": ImputeStrategy.MEDIAN,
    }
    assert all(
        operation.parameters.strategy is not ImputeStrategy.CONSTANT
        for operation in plan.operations
        if operation.operation_type is OperationType.IMPUTE_MISSING
    )


def test_every_enforced_operation_cites_the_user_request() -> None:
    controller = _crew_controller()

    enforced = [
        operation
        for operation in _plan_of(controller).operations
        if operation.operation_type
        in {
            OperationType.IMPUTE_MISSING,
            OperationType.DROP_COLUMN,
            OperationType.DEDUPLICATE_BY_KEYS,
        }
    ]
    assert enforced
    for operation in enforced:
        assert operation.user_requirement_ids, operation.operation_id


def test_no_request_not_planned_finding_survives_the_live_style_plan() -> None:
    controller = _crew_controller()

    assert [
        finding.code for finding in controller.session.findings if finding.blocking
    ] == []


def test_a_column_the_user_asked_to_drop_is_not_imputed_first() -> None:
    """The crew imputed category_id; the user asked for it to be dropped."""

    controller = _crew_controller()

    for operation in _plan_of(controller).operations:
        if operation.operation_type is OperationType.DROP_COLUMN:
            continue
        assert "category_id" not in operation.target_columns
        assert "boughtInLastMonth" not in operation.target_columns


def test_enforcement_never_leaves_an_operation_on_a_dropped_column() -> None:
    removed: set[str] = set()
    for operation in _plan_of(_crew_controller()).operations:
        assert removed.isdisjoint(operation.target_columns), operation.operation_id
        if operation.operation_type is OperationType.DROP_COLUMN:
            removed.update(operation.target_columns)


def test_the_reconciled_live_plan_executes_to_qa_passing_gold() -> None:
    controller = _crew_controller()
    assert controller.record_human_decision(
        HumanDecision.APPROVE, command_id="approve"
    ).changed
    assert controller.execute_current_plan(command_id="execute").changed

    state = controller.session.workflow_runtime.state
    assert state.stage is WorkflowStage.QA_PASSED
    assert state.qa_report.status is QAStatus.PASS
    gold = controller.session.workflow_runtime.gold_dataframe
    assert list(gold.columns) == ["asin", "title", "stars", "price"]
    assert len(gold) == 4
    assert int(gold.isna().sum().sum()) == 0
    bundle = controller.build_artifacts()
    assert isinstance(bundle, ArtifactSet)
    assert len(bundle.artifacts()) == 7


def test_gold_does_not_exist_before_approval_on_the_live_path() -> None:
    controller = _crew_controller()

    assert controller.session.workflow_runtime.gold_dataframe is None
    refused = controller.execute_current_plan(command_id="execute")
    assert not refused.changed


def test_enforcement_is_idempotent() -> None:
    """Applying it to a plan that already satisfies the requests changes nothing."""

    from datachef.planning.requests import enforce_requested_operations

    controller = _crew_controller()
    plan = _plan_of(controller)
    context = controller.session.workflow_runtime.state.planning_context
    requested = tuple(
        request.as_requested_operation()
        for request in controller.session.requested_transformations
    )

    again = enforce_requested_operations(plan, requested, context)

    assert again.plan_id == plan.plan_id
    assert again.operations == plan.operations


def test_an_unpriced_deduplication_request_is_still_refused() -> None:
    """The conservative pricing rule survives enforcement."""

    from datachef.planning.requests import enforceable_requests

    controller = _crew_controller("drop the duplicate values based on the title column")
    context = controller.session.workflow_runtime.state.planning_context
    requested = tuple(
        request.as_requested_operation()
        for request in controller.session.requested_transformations
    )
    assert any(
        item.operation_type is OperationType.DEDUPLICATE_BY_KEYS for item in requested
    ), "the objective did request a deduplication"

    allowed = enforceable_requests(requested, context)

    # title is not a nominated key set, so the estimator cannot price it.
    assert all(
        item.operation_type is not OperationType.DEDUPLICATE_BY_KEYS for item in allowed
    )
    assert all(
        operation.operation_type is not OperationType.DEDUPLICATE_BY_KEYS
        or operation.target_columns != ("title",)
        for operation in _plan_of(controller).operations
    )
    # And it stays visibly unplanned rather than being silently discarded.
    assert any(
        finding.code == "REQUEST_NOT_PLANNED"
        for finding in controller.session.findings
    )


def test_a_cast_request_is_still_left_to_the_diagnosis() -> None:
    """Preserved behaviour: enforcement never invents a cast."""

    from datachef.contracts import CastColumnParameters, CastTarget
    from datachef.planning.requests import enforceable_requests

    controller = _crew_controller()
    context = controller.session.workflow_runtime.state.planning_context
    cast = RequestedTransformation(
        request_id="request-cast-stars",
        operation_type=OperationType.CAST_COLUMN,
        target_columns=("stars",),
        parameters=CastColumnParameters(target_type=CastTarget.NUMERIC),
    ).as_requested_operation()

    assert enforceable_requests((cast,), context) == ()


# ---------------------------------------------------------------------------
# Operation order follows the objective, on both planners.
#
# The objective ends "finally drop the duplicate values based on the asin
# column". The deterministic planner used to emit its diagnosis-driven
# deduplication first, so the same objective produced a different gold table
# depending on which planner ran: an imputation mean measured after the
# duplicate row had already gone. Enforcement now positions every requested
# operation in the compiled request order, and repositions a planner operation
# that merely satisfies a request, so both paths agree.
# ---------------------------------------------------------------------------

# The order the objective states, and therefore the order the plan must use.
EXPECTED_SEQUENCE = (
    (OperationType.IMPUTE_MISSING, ("title",), ImputeStrategy.MODE),
    (OperationType.IMPUTE_MISSING, ("stars",), ImputeStrategy.MEAN),
    (OperationType.IMPUTE_MISSING, ("price",), ImputeStrategy.MEDIAN),
    (OperationType.DROP_COLUMN, ("category_id",), None),
    (OperationType.DROP_COLUMN, ("boughtInLastMonth",), None),
    (OperationType.DEDUPLICATE_BY_KEYS, ("asin",), None),
)


def _sequence(plan):
    return tuple(
        (
            operation.operation_type,
            operation.target_columns,
            getattr(operation.parameters, "strategy", None),
        )
        for operation in plan.operations
    )


def _rule_based_controller(objective: str = ML_OBJECTIVE) -> DataChefController:
    controller = DataChefController()
    controller.load_upload(
        UploadRequest(
            content=MODE_CSV,
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(encoding="utf-8-sig"),
        )
    )
    controller.diagnose()
    controller.submit_intent(
        UserIntent(
            intent_id="intent-order",
            user_goal=objective,
            selected_key_columns=("asin",),
            acceptable_row_loss_pct=50,
        ),
        (),
    )
    controller.prepare_plan(command_id="plan")
    return controller


def test_the_ml_objective_produces_the_stated_operation_sequence() -> None:
    """Requirement 1: the exact six operations, in the objective's order."""

    controller = _rule_based_controller()

    assert _sequence(_plan_of(controller)) == EXPECTED_SEQUENCE


def test_deduplication_is_the_final_operation() -> None:
    """Requirement 2: "finally drop the duplicate values" means finally."""

    for controller in (_rule_based_controller(), _crew_controller()):
        operations = _plan_of(controller).operations
        assert operations[-1].operation_type is OperationType.DEDUPLICATE_BY_KEYS
        assert operations[-1].target_columns == ("asin",)
        assert all(
            operation.operation_type is not OperationType.DEDUPLICATE_BY_KEYS
            for operation in operations[:-1]
        )


def test_imputations_are_measured_on_the_pre_deduplication_data() -> None:
    """Requirement 3: the duplicate row still counts toward mean and median.

    stars holds 4.5, 4.0, 3.5 and a duplicated 4.5 before deduplication, so the
    mean is 16.5 / 4 = 4.125. Had the deduplication run first the mean would be
    4.0, which is what the two planners used to disagree about.
    """

    controller = _rule_based_controller()
    controller.record_human_decision(HumanDecision.APPROVE, command_id="approve")
    controller.execute_current_plan(command_id="execute")
    gold = controller.session.workflow_runtime.gold_dataframe

    assert gold["stars"].tolist() == [4.5, 4.125, 4.0, 3.5]
    # price is 10, 20, 40 and a duplicated 10 before deduplication: median 15.0.
    assert gold["price"].tolist() == [10.0, 20.0, 15.0, 40.0]
    # And the deduplication still happened.
    assert len(gold) == 4
    assert gold["asin"].is_unique


def test_both_planners_produce_the_same_canonical_ordering() -> None:
    """Requirement 4: the deterministic and live paths cannot disagree."""

    rule_based = _sequence(_plan_of(_rule_based_controller()))
    crew_like = _sequence(_plan_of(_crew_controller()))

    assert rule_based == crew_like == EXPECTED_SEQUENCE


def test_both_planners_produce_the_same_gold_table() -> None:
    frames = []
    for controller in (_rule_based_controller(), _crew_controller()):
        controller.record_human_decision(HumanDecision.APPROVE, command_id="approve")
        controller.execute_current_plan(command_id="execute")
        state = controller.session.workflow_runtime.state
        assert state.stage is WorkflowStage.QA_PASSED
        assert state.qa_report.status is QAStatus.PASS
        frames.append(controller.session.workflow_runtime.gold_dataframe)

    assert list(frames[0].columns) == list(frames[1].columns)
    assert frames[0].reset_index(drop=True).equals(frames[1].reset_index(drop=True))


def test_a_planner_only_operation_still_runs_before_the_requested_ones() -> None:
    """Enforcement reorders requests, not unrelated diagnosis-driven work."""

    from datachef.planning.requests import enforce_requested_operations

    controller = _rule_based_controller()
    context = controller.session.workflow_runtime.state.planning_context
    plan = _plan_of(controller)
    requested = tuple(
        request.as_requested_operation()
        for request in controller.session.requested_transformations
    )

    # Re-enforcing is stable, so ordering is a fixed point rather than a shuffle.
    again = enforce_requested_operations(plan, requested, context)
    assert _sequence(again) == EXPECTED_SEQUENCE
    assert again.operations == plan.operations


# ---------------------------------------------------------------------------
# The second real smoke test: two blockers the earlier fixtures hid.
#
# 1. The user never touched the key-columns control, so selected_key_columns was
#    empty, so diagnosis nominated no key set, so the estimator could not price
#    the requested asin deduplication and it stayed blocked as
#    REQUEST_NOT_PLANNED. Asking to deduplicate on a column *is* the statement
#    that the column identifies a record, so the request now nominates it.
#
# 2. The objective says "boughtLastMonth" while the column is
#    "boughtInLastMonth". The reference never resolved, no request was compiled,
#    and the live crew's arbitrary CONSTANT imputation filled the gap because
#    there was nothing to override it.
#
# The earlier fixtures masked both: they passed selected_key_columns=("asin",)
# and silently corrected the column name.
# ---------------------------------------------------------------------------

# The objective exactly as typed, including the near-miss column reference.
VERBATIM_OBJECTIVE = (
    "Prepare this table for ML modelling, the objective is to use this table to "
    "train a model to predict the price column based on the other columns. For "
    "the missing values, check if the missing values in the column title is over "
    "40% and there's no mode drop it, otherwise if the mode exists for the column "
    "title use it to impute all null values, impute the missing values of the "
    "column stars using the mean, and impute the column price using the median, "
    "drop the category_id column, check the distribution in boughtLastMonth if "
    "it has over 40% of null and 0s as values drop the column. finally drop the "
    "duplicate values based on the asin column"
)

VERBATIM_EXPECTED = (
    (OperationType.IMPUTE_MISSING, ("title",), ImputeStrategy.MODE),
    (OperationType.IMPUTE_MISSING, ("stars",), ImputeStrategy.MEAN),
    (OperationType.IMPUTE_MISSING, ("price",), ImputeStrategy.MEDIAN),
    (OperationType.DROP_COLUMN, ("category_id",), None),
    (OperationType.DROP_COLUMN, ("boughtInLastMonth",), None),
    (OperationType.DEDUPLICATE_BY_KEYS, ("asin",), None),
)


def _no_key_controller(
    objective: str = VERBATIM_OBJECTIVE,
    *,
    csv: bytes = MODE_CSV,
    planner_factory=None,
    row_loss: float = 50.0,
) -> DataChefController:
    """A session where the user typed only the objective, as in the real app."""

    kwargs = {"planner_factory": planner_factory} if planner_factory else {}
    controller = DataChefController(**kwargs)
    controller.load_upload(
        UploadRequest(
            content=csv,
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(encoding="utf-8-sig"),
        )
    )
    controller.diagnose()
    controller.submit_intent(
        UserIntent(
            intent_id="intent-verbatim",
            user_goal=objective,
            selected_key_columns=(),  # untouched, exactly as in the smoke test
            acceptable_row_loss_pct=row_loss,
        ),
        (),
    )
    controller.prepare_plan(command_id="plan")
    return controller


# ---------------------------------------------------------------------------
# Explicit deduplication on a key the name heuristic never nominates
# ---------------------------------------------------------------------------


def test_an_explicit_dedup_request_nominates_its_key() -> None:
    """asin is not id-shaped, so only the explicit request can nominate it."""

    controller = _no_key_controller()

    assert controller.session.intent.selected_key_columns == ("asin",)


def test_the_nominated_key_is_measured_by_diagnosis() -> None:
    controller = _no_key_controller()
    report = controller.session.workflow_runtime.state.diagnostic_report

    metrics = {
        tuple(metric.key_columns): metric for metric in report.key_duplicate_metrics
    }
    assert ("asin",) in metrics
    assert metrics[("asin",)].duplicate_row_count == 1
    assert metrics[("asin",)].null_key_row_count == 0


def test_the_requested_dedup_is_planned_and_priced() -> None:
    """The reported blocker: REQUEST_NOT_PLANNED for asin must be gone."""

    controller = _no_key_controller()
    state = controller.session.workflow_runtime.state

    dedup = [
        operation
        for operation in state.transformation_plan.operations
        if operation.operation_type is OperationType.DEDUPLICATE_BY_KEYS
    ]
    assert [operation.target_columns for operation in dedup] == [("asin",)]
    # Grounded in real evidence: once the key is nominated the diagnosis raises a
    # genuine DUPLICATE_KEYS issue, so the deterministic planner's own operation
    # satisfies the request and cites that issue. Where no such operation exists
    # the request itself is the grounding, via user_requirement_ids.
    assert dedup[0].diagnostic_issue_ids or dedup[0].user_requirement_ids

    estimates = {
        estimate.operation_id: estimate
        for estimate in state.plan_validation.row_loss_estimates
    }
    priced = estimates[dedup[0].operation_id]
    assert priced.estimated_rows == 1
    assert priced.estimated_pct == pytest.approx(20.0)

    assert [
        finding.code for finding in controller.session.findings if finding.blocking
    ] == []


def test_an_unselected_key_without_the_request_is_still_not_nominated() -> None:
    """Nomination follows the request; it is not a blanket relaxation."""

    controller = _no_key_controller("drop the category_id column")

    assert controller.session.intent.selected_key_columns == ()
    report = controller.session.workflow_runtime.state.diagnostic_report
    assert all(
        tuple(metric.key_columns) != ("asin",)
        for metric in report.key_duplicate_metrics
    )


# ---------------------------------------------------------------------------
# The safety invariant: unsafe deduplication is still refused
# ---------------------------------------------------------------------------


CONSTANT_KEY_CSV = (
    b"grouping,value\n"
    b"same,1\n"
    b"same,2\n"
    b"same,3\n"
    b"same,4\n"
    b"same,5\n"
)

NULL_KEY_CSV = (
    b"code,value\n"
    b"k1,1\n"
    b",2\n"
    b"k1,3\n"
    b"k2,4\n"
)


def test_a_destructive_dedup_request_is_priced_then_refused() -> None:
    """Nominating is not approving: the row-loss threshold still blocks it.

    Deduplicating on a column holding one repeated value would collapse the
    table. It is now measured rather than ignored, and the measurement is what
    refuses it.
    """

    controller = _no_key_controller(
        "drop the duplicate values based on the grouping column",
        csv=CONSTANT_KEY_CSV,
        row_loss=10.0,
    )
    state = controller.session.workflow_runtime.state

    assert controller.session.intent.selected_key_columns == ("grouping",)
    codes = {finding.code for finding in controller.session.findings}
    validation_codes = {finding.code for finding in state.plan_validation.findings}
    assert "ROW_LOSS_THRESHOLD" in validation_codes or "REQUEST_NOT_PLANNED" in codes
    assert not state.plan_validation.valid
    # And nothing can be approved while a blocking finding stands.
    refused = controller.record_human_decision(
        HumanDecision.APPROVE, command_id="approve"
    )
    assert refused.code == "PLAN_NOT_APPROVABLE"
    assert controller.session.pending_approval is None


def test_a_null_key_dedup_request_is_still_refused() -> None:
    """The committed null-key guard is untouched by nomination."""

    controller = _no_key_controller(
        "drop the duplicate values based on the code column",
        csv=NULL_KEY_CSV,
    )
    state = controller.session.workflow_runtime.state

    assert controller.session.intent.selected_key_columns == ("code",)
    assert not state.plan_validation.valid
    assert "NULL_KEYS_UNSAFE" in {
        finding.code for finding in state.plan_validation.findings
    }
    assert controller.session.workflow_runtime.gold_dataframe is None


# ---------------------------------------------------------------------------
# The near-miss column reference
# ---------------------------------------------------------------------------


def test_a_near_miss_column_reference_resolves_when_unambiguous() -> None:
    """"boughtLastMonth" resolves to boughtInLastMonth, deterministically."""

    frame = _frame(unique_titles=False)
    requests = _by_column(_compile(VERBATIM_OBJECTIVE, frame))

    assert requests["boughtInLastMonth"].operation_type is OperationType.DROP_COLUMN


def test_an_ambiguous_near_miss_reference_resolves_to_nothing() -> None:
    """Two plausible columns means the reference is left alone, not guessed."""

    frame = pd.DataFrame(
        {
            "boughtInLastMonth": [0, None],
            "boughtInLastMonthTotal": [0, None],
            "keep": [1, 2],
        }
    )
    report = diagnose_raw_dataframe(frame, selected_key_columns=())

    assert compile_requests("drop the boughtLastMonth column", frame, report) == ()


def test_a_single_loose_word_never_renames_a_column() -> None:
    """The relaxed rule needs at least two words, so "month" resolves nothing."""

    frame = pd.DataFrame({"boughtInLastMonth": [0, None], "keep": [1, 2]})
    report = diagnose_raw_dataframe(frame, selected_key_columns=())

    assert compile_requests("drop the month column", frame, report) == ()


# ---------------------------------------------------------------------------
# The whole objective, on both planners
# ---------------------------------------------------------------------------


def test_the_verbatim_objective_produces_the_requested_types_and_parameters() -> None:
    controller = _no_key_controller()

    assert _sequence(_plan_of(controller)) == VERBATIM_EXPECTED


def test_the_verbatim_objective_matches_on_the_live_style_planner() -> None:
    controller = _no_key_controller(planner_factory=_CrewLikePlanner)

    assert _sequence(_plan_of(controller)) == VERBATIM_EXPECTED
    # No arbitrary CONSTANT survived, including on boughtInLastMonth.
    assert all(
        operation.parameters.strategy is not ImputeStrategy.CONSTANT
        for operation in _plan_of(controller).operations
        if operation.operation_type is OperationType.IMPUTE_MISSING
    )
    assert [
        finding.code for finding in controller.session.findings if finding.blocking
    ] == []


def test_the_crew_no_longer_imputes_a_column_the_user_asked_to_drop() -> None:
    """The reported plan imputed boughtInLastMonth; it must be dropped instead."""

    controller = _no_key_controller(planner_factory=_CrewLikePlanner)
    plan = _plan_of(controller)

    imputed = {
        operation.target_columns[0]
        for operation in plan.operations
        if operation.operation_type is OperationType.IMPUTE_MISSING
    }
    assert "boughtInLastMonth" not in imputed
    assert "category_id" not in imputed
    dropped = {
        operation.target_columns[0]
        for operation in plan.operations
        if operation.operation_type is OperationType.DROP_COLUMN
    }
    assert {"boughtInLastMonth", "category_id"} <= dropped


def test_the_verbatim_objective_runs_to_qa_passing_gold_on_both_planners() -> None:
    for factory in (None, _CrewLikePlanner):
        controller = _no_key_controller(planner_factory=factory)
        assert controller.record_human_decision(
            HumanDecision.APPROVE, command_id="approve"
        ).changed
        assert controller.execute_current_plan(command_id="execute").changed

        state = controller.session.workflow_runtime.state
        assert state.stage is WorkflowStage.QA_PASSED
        assert state.qa_report.status is QAStatus.PASS
        gold = controller.session.workflow_runtime.gold_dataframe
        assert list(gold.columns) == ["asin", "title", "stars", "price"]
        assert len(gold) == 4
        assert gold["asin"].is_unique
        assert int(gold.isna().sum().sum()) == 0
        assert len(controller.build_artifacts().artifacts()) == 7


def test_the_streamlit_path_with_the_verbatim_objective_and_no_key_selection() -> None:
    """The real smoke test: type the objective, touch nothing else, approve."""

    from streamlit.testing.v1 import AppTest

    from ui import state as ui_state

    def widget(at, kind, key):
        for element in getattr(at, kind):
            if getattr(element, "key", None) == key:
                return element
        raise KeyError(f"{kind}:{key}")

    at = AppTest.from_file(str(REPO_ROOT / "ui" / "app.py"), default_timeout=180)
    at.run()
    at.file_uploader[0].set_value(("amazon.csv", MODE_CSV, "text/csv"))
    at.run()
    widget(at, "button", ui_state.DIAGNOSE_WIDGET).click()
    at.run()
    widget(at, "button", ui_state.CONTINUE_TO_INTENT_WIDGET).click()
    at.run()
    # Only the objective is typed. No key columns, no cast selection.
    widget(at, "text_area", ui_state.GOAL_WIDGET).set_value(VERBATIM_OBJECTIVE)
    widget(at, "slider", ui_state.ROW_LOSS_WIDGET).set_value(50.0)
    widget(at, "button", ui_state.SUBMIT_INTENT_WIDGET).click()
    at.run()
    widget(at, "button", ui_state.PREPARE_PLAN_WIDGET).click()
    at.run()

    controller = at.session_state[ui_state.CONTROLLER]
    assert _sequence(_plan_of(controller)) == VERBATIM_EXPECTED
    assert [
        finding.code for finding in controller.session.findings if finding.blocking
    ] == []
    rendered = " ".join(element.value for element in at.markdown)
    assert "Not in this plan" not in rendered
    assert controller.session.workflow_runtime.gold_dataframe is None

    widget(at, "button", ui_state.APPROVE_WIDGET).click()
    at.run()
    widget(at, "button", ui_state.EXECUTE_WIDGET).click()
    at.run()

    runtime = at.session_state[ui_state.CONTROLLER].session.workflow_runtime
    assert runtime.state.stage is WorkflowStage.QA_PASSED
    assert runtime.state.qa_report.status is QAStatus.PASS
    gold = runtime.gold_dataframe
    assert list(gold.columns) == ["asin", "title", "stars", "price"]
    assert gold["asin"].is_unique and len(gold) == 4
    assert len(at.download_button) == 7
    assert len(at.exception) == 0, [item.value for item in at.exception]
