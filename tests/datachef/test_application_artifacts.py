from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import pickle

import pandas as pd
import pytest

import datachef.application.artifacts as artifacts_module

from datachef.application import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactFailure,
    ArtifactKind,
    ArtifactSet,
    CsvParserOptions,
    DataChefController,
    DownloadArtifact,
    JsonRecordsParserOptions,
    RequestedTransformation,
    UploadFormat,
    UploadRequest,
    build_artifact_set,
)
from datachef.contracts import (
    CastColumnParameters,
    CastTarget,
    DownstreamUse,
    HumanDecision,
    OperationType,
    PIIHandling,
    QAStatus,
    UserIntent,
    WorkflowStage,
)
from datachef.application.pipeline_render import render_pipeline_bytes
from datachef.diagnostics import identify_dataset
from datachef.workflow import WorkflowRuntime


NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
SECRET_GOAL = "Prepare the table for the quarterly board review of margins."
SECRET_QUESTION = "Which region carries the confidential surcharge?"
CSV = (
    b"order_id,region,amount,ordered_on\n"
    b"1,North,10,2026-01-01\n"
    b"2,South,20,2026-01-02\n"
    b"2,South,20,2026-01-02\n"
    b"3,North,30,2026-01-03\n"
)
REPO_ROOT = Path(__file__).parents[2]


def _intent(**overrides) -> UserIntent:
    payload = {
        "intent_id": "intent-artifacts",
        "user_goal": SECRET_GOAL,
        "downstream_use": DownstreamUse.ANALYSIS,
        "selected_key_columns": ("order_id",),
        "required_columns": ("order_id",),
        "acceptable_row_loss_pct": 50,
        "pii_handling": PIIHandling.NONE,
        "questions": (SECRET_QUESTION,),
    }
    payload.update(overrides)
    return UserIntent(**payload)


def _csv_request(content: bytes = CSV) -> UploadRequest:
    return UploadRequest(
        content=content,
        declared_suffix=".csv",
        format=UploadFormat.CSV,
        parser_options=CsvParserOptions(encoding="utf-8-sig"),
    )


def _json_request(content: bytes) -> UploadRequest:
    return UploadRequest(
        content=content,
        declared_suffix=".json",
        format=UploadFormat.JSON_RECORDS,
        parser_options=JsonRecordsParserOptions(),
    )


def _controller(request: UploadRequest | None = None) -> DataChefController:
    controller = DataChefController(clock=lambda: NOW)
    assert controller.load_upload(request or _csv_request()).changed
    assert controller.diagnose().changed
    return controller


def _gold_controller() -> DataChefController:
    controller = _controller()
    controller.submit_intent(_intent(), ())
    assert controller.prepare_plan(command_id="plan").code == "PLAN_AWAITING_APPROVAL"
    assert controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="approve",
    ).changed
    assert controller.execute_current_plan(command_id="execute").changed
    runtime = controller.session.workflow_runtime
    assert runtime is not None and runtime.state.stage is WorkflowStage.QA_PASSED
    return controller


def _gold_runtime() -> WorkflowRuntime:
    runtime = _gold_controller().session.workflow_runtime
    assert runtime is not None
    return runtime


def _source_metadata(controller: DataChefController):
    source = controller.session.source
    assert source is not None
    return source.metadata


def _bundle() -> ArtifactSet:
    controller = _gold_controller()
    bundle = controller.build_artifacts()
    assert isinstance(bundle, ArtifactSet)
    return bundle


def _repository_files() -> set[Path]:
    seen: set[Path] = set()
    for relative in (".", "datachef", "tests"):
        base = REPO_ROOT / relative
        for path in base.rglob("*") if relative != "." else base.iterdir():
            if "__pycache__" in path.parts or ".git" in path.parts:
                continue
            if ".venv" in path.parts or ".pytest_cache" in path.parts:
                continue
            seen.add(path)
    return seen


def test_pass_run_builds_the_complete_approved_bundle() -> None:
    controller = _gold_controller()

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    assert tuple(item.kind for item in bundle.artifacts()) == (
        ArtifactKind.CLEANED_CSV,
        ArtifactKind.CLEANED_PARQUET,
        ArtifactKind.TRANSFORMATION_PLAN_JSON,
        ArtifactKind.QA_REPORT_JSON,
        ArtifactKind.EXECUTION_CHANGE_LOG_JSON,
        ArtifactKind.PIPELINE_SCRIPT_PY,
        ArtifactKind.MANIFEST_JSON,
    )
    assert len(bundle.downloads()) == 6
    assert bundle.manifest not in bundle.downloads()
    media_types = {item.kind: item.media_type for item in bundle.artifacts()}
    assert media_types[ArtifactKind.CLEANED_CSV] == "text/csv"
    assert media_types[ArtifactKind.CLEANED_PARQUET] == "application/vnd.apache.parquet"
    assert media_types[ArtifactKind.QA_REPORT_JSON] == "application/json"
    for artifact in bundle.artifacts():
        assert isinstance(artifact, DownloadArtifact)
        assert artifact.content
        assert artifact.byte_size == len(artifact.content)


def test_cleaned_csv_bytes_read_back_as_the_gold_table() -> None:
    controller = _gold_controller()
    gold = controller.session.workflow_runtime.gold_dataframe.reset_index(drop=True)

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    content = bundle.cleaned_csv.content
    assert content == gold.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="",
    ).encode("utf-8")
    reloaded = pd.read_csv(
        StringIO(content.decode("utf-8")),
        index_col=False,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    assert tuple(reloaded.columns) == tuple(gold.columns)
    assert reloaded.shape == gold.shape
    for position, column in enumerate(gold.columns):
        assert tuple(reloaded.iloc[:, position]) == tuple(
            artifacts_module._csv_text(value) for value in gold[column]
        )
    assert artifacts_module._csv_round_trips(content, gold)


def test_cleaned_parquet_bytes_read_back_with_the_gold_fingerprint() -> None:
    controller = _gold_controller()
    gold = controller.session.workflow_runtime.gold_dataframe.reset_index(drop=True)

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    reloaded = pd.read_parquet(
        BytesIO(bundle.cleaned_parquet.content),
        engine="pyarrow",
    )
    assert identify_dataset(reloaded).fingerprint == identify_dataset(gold).fingerprint


@pytest.mark.parametrize(
    ("attribute", "evidence"),
    (
        ("transformation_plan_json", "transformation_plan"),
        ("qa_report_json", "qa_report"),
        ("execution_change_log_json", "execution_result"),
    ),
)
def test_json_evidence_is_the_exact_canonical_phase1a_contract(
    attribute: str,
    evidence: str,
) -> None:
    controller = _gold_controller()
    state = controller.session.workflow_runtime.state
    expected = getattr(state, evidence).model_dump(mode="json")

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    content = getattr(bundle, attribute).content
    assert content == json.dumps(
        expected,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8") + b"\n"
    assert json.loads(content.decode("utf-8")) == expected
    assert content.endswith(b"\n")


def test_execution_change_log_keeps_ordered_operation_records() -> None:
    controller = _gold_controller()
    result = controller.session.workflow_runtime.state.execution_result
    plan = controller.session.workflow_runtime.state.transformation_plan

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    payload = json.loads(bundle.execution_change_log_json.content.decode("utf-8"))
    recorded = tuple(item["operation_id"] for item in payload["operation_records"])
    assert recorded == tuple(record.operation_id for record in result.operation_records)
    assert recorded == tuple(operation.operation_id for operation in plan.operations)


def test_manifest_describes_the_bundle_and_hashes_every_other_artifact() -> None:
    controller = _gold_controller()
    state = controller.session.workflow_runtime.state
    metadata = _source_metadata(controller)

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    manifest = json.loads(bundle.manifest.content.decode("utf-8"))
    assert manifest["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["dataset_id"] == state.dataset_identity.dataset_id
    assert manifest["dataset_fingerprint"] == state.dataset_identity.fingerprint
    assert manifest["source_format"] == metadata.format.value
    assert manifest["source_parser_options"] == metadata.parser_options.model_dump(
        mode="json"
    )
    assert manifest["plan_id"] == state.transformation_plan.plan_id
    assert manifest["plan_version"] == state.transformation_plan.version
    assert manifest["accepted_review_attempt"] == state.accepted_review.attempt
    assert manifest["qa_report_id"] == state.qa_report.qa_report_id
    assert manifest["execution_id"] == state.execution_result.execution_id
    assert manifest["result_fingerprint"] == state.execution_result.result_fingerprint
    assert manifest["approved_at"] == state.human_approval.decided_at.isoformat()

    described = {entry["kind"]: entry for entry in manifest["artifacts"]}
    assert ArtifactKind.MANIFEST_JSON.value not in described
    assert set(described) == {item.kind.value for item in bundle.downloads()}
    for artifact in bundle.downloads():
        entry = described[artifact.kind.value]
        assert entry["filename"] == artifact.filename
        assert entry["media_type"] == artifact.media_type
        assert entry["byte_size"] == len(artifact.content)
        assert entry["sha256"] == sha256(artifact.content).hexdigest()


def test_filenames_are_sanitized_generated_identifiers() -> None:
    controller = _gold_controller()
    state = controller.session.workflow_runtime.state

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    dataset_short = artifacts_module._short_id(state.dataset_identity.dataset_id)
    plan_short = artifacts_module._short_id(state.transformation_plan.plan_id)
    stem = f"datachef_{dataset_short}_{plan_short}"
    assert bundle.cleaned_csv.filename == f"{stem}_cleaned.csv"
    assert bundle.cleaned_parquet.filename == f"{stem}_cleaned.parquet"
    assert bundle.manifest.filename == f"{stem}_manifest.json"
    for artifact in bundle.artifacts():
        assert artifact.filename.startswith("datachef_")
        assert "/" not in artifact.filename and "\\" not in artifact.filename
        assert ".." not in artifact.filename
        assert artifact.filename == Path(artifact.filename).name


def test_bundle_leaks_no_request_id_path_free_form_intent_or_credentials(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-credential-value-do-not-leak")
    controller = _gold_controller()
    metadata = _source_metadata(controller)

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    forbidden = (
        metadata.request_id,
        SECRET_GOAL,
        SECRET_QUESTION,
        "fake-credential-value-do-not-leak",
        "GOOGLE_API_KEY",
        str(REPO_ROOT),
        REPO_ROOT.name + "\\",
    )
    manifest_text = bundle.manifest.content.decode("utf-8")
    for needle in forbidden:
        assert needle not in manifest_text
    assert ":\\" not in manifest_text and ":/" not in manifest_text
    for artifact in (
        bundle.transformation_plan_json,
        bundle.qa_report_json,
        bundle.execution_change_log_json,
        bundle.manifest,
    ):
        text = artifact.content.decode("utf-8")
        assert metadata.request_id not in text
        assert SECRET_GOAL not in text
        assert SECRET_QUESTION not in text
        assert "fake-credential-value-do-not-leak" not in text


def test_repeated_construction_is_byte_identical() -> None:
    controller = _gold_controller()

    first = controller.build_artifacts()
    second = controller.build_artifacts()

    assert isinstance(first, ArtifactSet) and isinstance(second, ArtifactSet)
    for left, right in zip(first.artifacts(), second.artifacts()):
        assert left.kind is right.kind
        assert left.filename == right.filename
        assert left.content == right.content
        assert left.sha256 == right.sha256


def test_building_the_bundle_creates_no_repository_file() -> None:
    controller = _gold_controller()
    before = _repository_files()

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    assert _repository_files() == before


def test_building_the_bundle_does_not_mutate_the_session() -> None:
    controller = _gold_controller()
    before = controller.session

    controller.build_artifacts()
    after = controller.session

    assert after.revision == before.revision
    assert after.command_history == before.command_history
    assert after.screen is before.screen
    assert after.workflow_runtime.state == before.workflow_runtime.state


def test_bundle_containers_are_immutable_and_unserializable() -> None:
    bundle = _bundle()

    with pytest.raises(Exception):
        bundle.cleaned_csv = None
    with pytest.raises(Exception):
        bundle.cleaned_csv.filename = "other.csv"
    with pytest.raises(TypeError):
        pickle.dumps(bundle)
    with pytest.raises(TypeError):
        pickle.dumps(bundle.cleaned_csv)
    assert "content" not in repr(bundle.cleaned_csv)


def test_raw_only_session_produces_no_bundle() -> None:
    controller = _controller()

    failure = controller.build_artifacts()

    assert isinstance(failure, ArtifactFailure)
    assert failure.code.value == "GOLD_UNAVAILABLE"


def test_rejected_plan_produces_no_bundle() -> None:
    controller = _controller()
    controller.submit_intent(_intent(), ())
    controller.prepare_plan(command_id="plan")
    controller.record_human_decision(HumanDecision.REJECT, command_id="reject")
    controller.execute_current_plan(command_id="execute")

    failure = controller.build_artifacts()

    assert isinstance(failure, ArtifactFailure)
    assert failure.code.value == "GOLD_UNAVAILABLE"
    assert controller.session.workflow_runtime.state.stage is WorkflowStage.PLAN_REJECTED


def test_genuine_qa_failure_produces_no_bundle() -> None:
    controller = _controller(
        _json_request(
            b'[{"order_id":1,"amount_text":"1"},{"order_id":2,"amount_text":"2"},'
            b'{"order_id":3,"amount_text":"3"},{"order_id":4,"amount_text":"4"},'
            b'{"order_id":5,"amount_text":"bad"}]'
        )
    )
    controller.submit_intent(
        _intent(),
        (
            RequestedTransformation(
                request_id="request-cast-amount_text",
                operation_type=OperationType.CAST_COLUMN,
                target_columns=("amount_text",),
                parameters=CastColumnParameters(target_type=CastTarget.NUMERIC),
            ),
        ),
    )
    controller.prepare_plan(command_id="plan")
    controller.record_human_decision(HumanDecision.APPROVE, command_id="approve")
    controller.execute_current_plan(command_id="execute")
    runtime = controller.session.workflow_runtime
    assert runtime.state.stage is WorkflowStage.QA_FAILED
    assert runtime.gold_dataframe is None

    failure = controller.build_artifacts()

    assert isinstance(failure, ArtifactFailure)
    assert failure.code.value == "GOLD_UNAVAILABLE"


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    (
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={"stage": WorkflowStage.QA_WARNING}
                ),
            ),
            "GOLD_UNAVAILABLE",
            id="warn-stage",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={"stage": WorkflowStage.QA_FAILED}
                ),
            ),
            "GOLD_UNAVAILABLE",
            id="fail-stage",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={"stage": WorkflowStage.EXECUTION_FAILED}
                ),
            ),
            "GOLD_UNAVAILABLE",
            id="execution-failed-stage",
        ),
        pytest.param(
            lambda runtime: replace(runtime, gold_dataframe=None),
            "GOLD_UNAVAILABLE",
            id="missing-gold",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(update={"execution_result": None}),
            ),
            "EVIDENCE_INCOMPLETE",
            id="missing-execution-evidence",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(update={"human_approval": None}),
            ),
            "EVIDENCE_INCOMPLETE",
            id="missing-approval",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={
                        "qa_report": runtime.state.qa_report.model_copy(
                            update={"status": QAStatus.WARN}
                        )
                    }
                ),
            ),
            "GOLD_EVIDENCE_MISMATCH",
            id="forged-warn-status",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={
                        "execution_result": runtime.state.execution_result.model_copy(
                            update={"success": False}
                        )
                    }
                ),
            ),
            "GOLD_EVIDENCE_MISMATCH",
            id="failed-execution-claiming-gold",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={
                        "execution_result": runtime.state.execution_result.model_copy(
                            update={"plan_id": "foreign-plan"}
                        )
                    }
                ),
            ),
            "GOLD_EVIDENCE_MISMATCH",
            id="foreign-plan-evidence",
        ),
        pytest.param(
            lambda runtime: replace(
                runtime,
                state=runtime.state.model_copy(
                    update={
                        "qa_report": runtime.state.qa_report.model_copy(
                            update={"dataset_id": "foreign-dataset"}
                        )
                    }
                ),
            ),
            "GOLD_EVIDENCE_MISMATCH",
            id="foreign-qa-evidence",
        ),
    ),
)
def test_untrustworthy_runtimes_never_produce_bundle_bytes(
    mutate,
    expected_code: str,
) -> None:
    controller = _gold_controller()
    tampered = mutate(controller.session.workflow_runtime)

    failure = build_artifact_set(tampered, _source_metadata(controller))

    assert isinstance(failure, ArtifactFailure)
    assert failure.code.value == expected_code
    assert failure.safe_message and failure.suggested_action


def test_stale_fingerprint_after_modified_gold_produces_no_bundle() -> None:
    controller = _gold_controller()
    runtime = controller.session.workflow_runtime
    forged = runtime.gold_dataframe.copy(deep=True)
    forged.loc[forged.index[0], "amount"] = 999_999

    failure = build_artifact_set(
        replace(runtime, gold_dataframe=forged),
        _source_metadata(controller),
    )

    assert isinstance(failure, ArtifactFailure)
    assert failure.code.value == "GOLD_EVIDENCE_MISMATCH"


@pytest.mark.parametrize("candidate", (object(), None, "runtime"))
def test_non_runtime_candidates_are_refused(candidate: object) -> None:
    controller = _gold_controller()

    failure = build_artifact_set(candidate, _source_metadata(controller))

    assert isinstance(failure, ArtifactFailure)
    assert failure.code.value == "GOLD_UNAVAILABLE"


@pytest.mark.parametrize("metadata", (object(), None, "CSV"))
def test_foreign_source_metadata_is_refused(metadata: object) -> None:
    failure = build_artifact_set(_gold_runtime(), metadata)

    assert isinstance(failure, ArtifactFailure)
    assert failure.code.value == "SOURCE_METADATA_MISMATCH"


@pytest.mark.parametrize(
    "target",
    ("_csv_bytes", "_parquet_bytes", "canonical_json", "render_pipeline_bytes"),
)
def test_serialization_failure_yields_no_partial_bundle(monkeypatch, target: str) -> None:
    def failing(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("private serializer detail")

    monkeypatch.setattr(artifacts_module, target, failing)
    controller = _gold_controller()
    before = _repository_files()

    failure = controller.build_artifacts()

    assert isinstance(failure, ArtifactFailure)
    assert failure.code.value == "SERIALIZATION_FAILURE"
    assert "private serializer detail" not in repr(failure)
    assert _repository_files() == before


def test_bytes_that_do_not_read_back_as_gold_yield_no_bundle(monkeypatch) -> None:
    monkeypatch.setattr(
        artifacts_module,
        "_csv_round_trips",
        lambda content, gold: False,
    )

    failure = _gold_controller().build_artifacts()

    assert isinstance(failure, ArtifactFailure)
    assert failure.code.value == "ROUND_TRIP_MISMATCH"


# The gold tables below stand in for what a real run produces. Every one of
# them packages perfectly well; each used to be refused because the *reader*
# used for verification disagreed with the bytes, not because the bytes were
# wrong. They are the regression set for that defect.
_SOUND_GOLD_TABLES = {
    # A mean or median lands on a full-precision double. Pandas' default CSV
    # float parser is not round-trip exact, so re-reading it changed the last
    # bits and the whole bundle was refused. This is the case a user hit.
    "imputed-mean": pd.DataFrame(
        {"amount": [6.1, 15.0, 48.3, 22.983333333333334, 35.3]}
    ),
    "arbitrary-floats": pd.DataFrame(
        {"x": [0.1 + 0.2, 1 / 3, 1e-20, -1.7976931348623157e308]}
    ),
    # Text that merely looks like something else. Re-reading with inference
    # turned "1.50" into 1.5, "007" into 7, and "NA" into a missing value.
    "numeric-looking-text": pd.DataFrame({"sku": ["1.50", "2.00", "3", "007"]}),
    "missing-token-text": pd.DataFrame({"code": ["NA", "null", "None", "ok"]}),
    "empty-string-text": pd.DataFrame({"title": ["a", "", "c"]}),
    "mixed-text-and-missing": pd.DataFrame({"note": ["0", None, "2"]}),
    "nullable-integers": pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64")}),
    "dates": pd.DataFrame({"day": pd.to_datetime(["2026-01-01", "2026-01-02"])}),
    "awkward-text": pd.DataFrame(
        {"text": ['he said "hi"', "a,b", "x\ny", "  padded  ", "caf\u00e9"]}
    ),
    "everything-missing": pd.DataFrame({"a": [None, None], "b": [1.0, 2.0]}),
}


@pytest.mark.parametrize("name", sorted(_SOUND_GOLD_TABLES))
def test_a_soundly_packaged_table_is_never_called_a_round_trip_mismatch(
    name: str,
) -> None:
    gold = _SOUND_GOLD_TABLES[name].reset_index(drop=True)

    assert artifacts_module._csv_round_trips(artifacts_module._csv_bytes(gold), gold)
    assert artifacts_module._parquet_round_trips(
        artifacts_module._parquet_bytes(gold),
        gold,
    )


def test_a_mean_imputed_run_still_produces_the_whole_bundle() -> None:
    """End to end over the controller, on the shape that exposed the defect.

    ``amount`` is imputed with its own mean, so gold carries a value no
    fixed-width decimal renders exactly. The bundle must still be built.
    """

    controller = _controller(
        _csv_request(
            b"order_id,region,amount\n"
            b"1,North,6.1\n"
            b"2,South,15.0\n"
            b"3,North,48.3\n"
            b"4,West,\n"
            b"5,East,35.3\n"
            b"6,North,\n"
            b"7,South,6.6\n"
            b"8,West,26.6\n"
        )
    )
    controller.submit_intent(
        _intent(user_goal="Impute amount with mean.", selected_key_columns=()),
        (),
    )
    assert controller.prepare_plan(command_id="plan").code == "PLAN_AWAITING_APPROVAL"
    assert controller.record_human_decision(
        HumanDecision.APPROVE,
        command_id="approve",
    ).changed
    assert controller.execute_current_plan(command_id="execute").changed
    runtime = controller.session.workflow_runtime
    assert runtime is not None and runtime.state.stage is WorkflowStage.QA_PASSED
    # The imputed value is genuinely one the default CSV reader rewrites.
    imputed = runtime.gold_dataframe["amount"].iloc[3]
    assert imputed == 22.983333333333334
    assert pd.read_csv(StringIO(f"a\n{imputed}\n"))["a"].iloc[0] != imputed

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    assert len(bundle.artifacts()) == 7


@pytest.mark.parametrize(
    ("name", "corrupt"),
    (
        ("changed-value", lambda text: text.replace("North", "Nrth", 1)),
        ("dropped-row", lambda text: "\n".join(text.split("\n")[:-2]) + "\n"),
        ("renamed-column", lambda text: text.replace("region", "area", 1)),
        ("blanked-cell", lambda text: text.replace(",North,", ",,", 1)),
        ("extra-row", lambda text: text + "9,North,99,2026-01-09\n"),
    ),
)
def test_a_genuinely_corrupted_csv_still_refuses_the_bundle(
    monkeypatch,
    name: str,
    corrupt,
) -> None:
    """The verification is looser about formatting, not about content."""

    del name
    honest = artifacts_module._csv_bytes

    def corrupted(gold: pd.DataFrame) -> bytes:
        text = honest(gold).decode("utf-8")
        altered = corrupt(text)
        assert altered != text
        return altered.encode("utf-8")

    monkeypatch.setattr(artifacts_module, "_csv_bytes", corrupted)

    failure = _gold_controller().build_artifacts()

    assert isinstance(failure, ArtifactFailure)
    assert failure.code.value == "ROUND_TRIP_MISMATCH"


def test_a_corrupted_parquet_still_refuses_the_bundle(monkeypatch) -> None:
    honest = artifacts_module._parquet_bytes

    def corrupted(gold: pd.DataFrame) -> bytes:
        altered = gold.copy()
        altered.iloc[0, 0] = altered.iloc[1, 0]
        return honest(altered)

    monkeypatch.setattr(artifacts_module, "_parquet_bytes", corrupted)

    failure = _gold_controller().build_artifacts()

    assert isinstance(failure, ArtifactFailure)
    assert failure.code.value == "ROUND_TRIP_MISMATCH"


def test_verification_reads_the_packaged_bytes_and_not_the_gold_frame() -> None:
    """A guard against the check quietly becoming a comparison with itself."""

    gold = pd.DataFrame({"a": [1.5, 2.5], "b": ["x", "y"]})
    sound = artifacts_module._csv_bytes(gold)

    assert artifacts_module._csv_round_trips(sound, gold)
    assert not artifacts_module._csv_round_trips(sound.replace(b"x", b"z"), gold)
    assert not artifacts_module._csv_round_trips(sound.replace(b"1.5", b"1.6"), gold)
    assert not artifacts_module._csv_round_trips(sound.replace(b"a,b", b"a,c"), gold)


def test_pipeline_script_ships_as_the_seventh_artifact() -> None:
    controller = _gold_controller()
    state = controller.session.workflow_runtime.state

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    assert len(bundle.artifacts()) == 7
    assert len(bundle.downloads()) == 6
    script = bundle.pipeline_script
    assert script.kind is ArtifactKind.PIPELINE_SCRIPT_PY
    assert script.media_type == "text/x-python"
    assert script in bundle.downloads()
    # Byte-identical to the renderer called directly on the approved plan.
    assert script.content == render_pipeline_bytes(state.transformation_plan)
    assert b"import datachef" not in script.content
    assert script.content.decode("utf-8").startswith("#!/usr/bin/env python3")


def test_pipeline_script_filename_follows_the_existing_scheme() -> None:
    controller = _gold_controller()
    state = controller.session.workflow_runtime.state

    bundle = controller.build_artifacts()

    assert isinstance(bundle, ArtifactSet)
    dataset_short = artifacts_module._short_id(state.dataset_identity.dataset_id)
    plan_short = artifacts_module._short_id(state.transformation_plan.plan_id)
    expected = f"datachef_{dataset_short}_{plan_short}_pipeline.py"
    assert bundle.pipeline_script.filename == expected


def test_manifest_schema_version_is_two_and_describes_six_artifacts() -> None:
    bundle = _bundle()

    manifest = json.loads(bundle.manifest.content.decode("utf-8"))

    # The artifacts list is its own schema: six described entries, not five.
    assert manifest["artifact_schema_version"] == 2
    assert ARTIFACT_SCHEMA_VERSION == 2
    assert len(manifest["artifacts"]) == 6
    described = {entry["kind"]: entry for entry in manifest["artifacts"]}
    assert ArtifactKind.PIPELINE_SCRIPT_PY.value in described
    assert ArtifactKind.MANIFEST_JSON.value not in described
    entry = described[ArtifactKind.PIPELINE_SCRIPT_PY.value]
    # Recomputed from the bytes actually returned in the ArtifactSet.
    assert entry["sha256"] == sha256(bundle.pipeline_script.content).hexdigest()
    assert entry["byte_size"] == len(bundle.pipeline_script.content)
    assert entry["media_type"] == "text/x-python"
    assert entry["filename"] == bundle.pipeline_script.filename


def test_pipeline_script_bytes_are_identical_across_repeated_builds() -> None:
    first = _bundle().pipeline_script.content
    second = _bundle().pipeline_script.content

    assert first == second
