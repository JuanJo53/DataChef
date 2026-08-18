"""One complete trusted download bundle built from verified Phase 1A evidence.

The bundle is produced as a whole or not at all. Every claim printed into the
manifest is re-derived from the final artifact bytes rather than copied from the
runtime, and nothing is written to disk: artifact bytes stay runtime-only,
exactly like uploaded bytes and parsed snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from io import BytesIO, StringIO
import json
import re
from typing import Any

import pandas as pd

from datachef.application.models import (
    SourceMetadata,
    StrictApplicationModel,
)
from datachef.application.pipeline_render import (
    PIPELINE_MEDIA_TYPE,
    render_pipeline_bytes,
)
from datachef.contracts import (
    HumanDecision,
    QAStatus,
    WorkflowStage,
)
from datachef.diagnostics import identify_dataset
from datachef.workflow import WorkflowRuntime
from pydantic import Field


# Version 2 adds the rendered pipeline script to the manifest's artifact list.
# That list is its own schema: a consumer written against version 1 expects five
# described artifacts and would silently mis-read six, so the bump is the signal.
ARTIFACT_SCHEMA_VERSION = 2

CSV_MEDIA_TYPE = "text/csv"
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
JSON_MEDIA_TYPE = "application/json"


class ArtifactKind(StrEnum):
    CLEANED_CSV = "CLEANED_CSV"
    CLEANED_PARQUET = "CLEANED_PARQUET"
    TRANSFORMATION_PLAN_JSON = "TRANSFORMATION_PLAN_JSON"
    QA_REPORT_JSON = "QA_REPORT_JSON"
    EXECUTION_CHANGE_LOG_JSON = "EXECUTION_CHANGE_LOG_JSON"
    PIPELINE_SCRIPT_PY = "PIPELINE_SCRIPT_PY"
    MANIFEST_JSON = "MANIFEST_JSON"


class ArtifactFailureCode(StrEnum):
    GOLD_UNAVAILABLE = "GOLD_UNAVAILABLE"
    EVIDENCE_INCOMPLETE = "EVIDENCE_INCOMPLETE"
    GOLD_EVIDENCE_MISMATCH = "GOLD_EVIDENCE_MISMATCH"
    SOURCE_METADATA_MISMATCH = "SOURCE_METADATA_MISMATCH"
    SERIALIZATION_FAILURE = "SERIALIZATION_FAILURE"
    ROUND_TRIP_MISMATCH = "ROUND_TRIP_MISMATCH"


class ArtifactFailure(StrictApplicationModel):
    code: ArtifactFailureCode
    safe_message: str = Field(min_length=1)
    suggested_action: str = Field(min_length=1)


_SAFE_MESSAGES = {
    ArtifactFailureCode.GOLD_UNAVAILABLE: (
        "No verified gold table is available to download.",
        "Complete an approved run whose quality assurance passed, then download.",
    ),
    ArtifactFailureCode.EVIDENCE_INCOMPLETE: (
        "The completed run does not carry the evidence a download must cite.",
        "Re-run the approved plan so plan, execution, and quality evidence exist.",
    ),
    ArtifactFailureCode.GOLD_EVIDENCE_MISMATCH: (
        "The gold table does not match its recorded execution evidence.",
        "Re-run the approved plan; do not distribute this result.",
    ),
    ArtifactFailureCode.SOURCE_METADATA_MISMATCH: (
        "The supplied source description does not belong to this run.",
        "Rebuild the download from the current session source.",
    ),
    ArtifactFailureCode.SERIALIZATION_FAILURE: (
        "The verified gold table could not be packaged for download.",
        "Retry the download; if it persists, re-run the approved plan.",
    ),
    ArtifactFailureCode.ROUND_TRIP_MISMATCH: (
        "A packaged file did not read back as the verified gold table.",
        "Retry the download; if it persists, re-run the approved plan.",
    ),
}


def _failure(code: ArtifactFailureCode) -> ArtifactFailure:
    message, action = _SAFE_MESSAGES[code]
    return ArtifactFailure(code=code, safe_message=message, suggested_action=action)


@dataclass(frozen=True, slots=True)
class DownloadArtifact:
    """Runtime-only immutable owner of one artifact's bytes."""

    kind: ArtifactKind
    filename: str
    media_type: str
    content: bytes = field(repr=False)
    dataset_id: str
    plan_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise TypeError("artifact content must be bytes")
        if not self.filename.strip() or not self.media_type.strip():
            raise ValueError("artifact filename and media type must be nonempty")
        if not self.dataset_id.strip() or not self.plan_id.strip():
            raise ValueError("artifact provenance identifiers must be nonempty")

    @property
    def byte_size(self) -> int:
        return len(self.content)

    @property
    def sha256(self) -> str:
        return sha256(self.content).hexdigest()

    def __getstate__(self) -> object:
        raise TypeError("download artifact is runtime-only and cannot be serialized")


@dataclass(frozen=True, slots=True)
class ArtifactSet:
    """Runtime-only owner of the complete approved download bundle."""

    cleaned_csv: DownloadArtifact
    cleaned_parquet: DownloadArtifact
    transformation_plan_json: DownloadArtifact
    qa_report_json: DownloadArtifact
    execution_change_log_json: DownloadArtifact
    pipeline_script: DownloadArtifact
    manifest: DownloadArtifact

    def downloads(self) -> tuple[DownloadArtifact, ...]:
        """Return every non-manifest artifact the manifest must describe."""

        return (
            self.cleaned_csv,
            self.cleaned_parquet,
            self.transformation_plan_json,
            self.qa_report_json,
            self.execution_change_log_json,
            self.pipeline_script,
        )

    def artifacts(self) -> tuple[DownloadArtifact, ...]:
        return (*self.downloads(), self.manifest)

    def __getstate__(self) -> object:
        raise TypeError("artifact set is runtime-only and cannot be serialized")


def _short_id(value: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", value) if part]
    tail = parts[-1] if parts else ""
    cleaned = re.sub(r"[^a-z0-9]", "", tail.lower())
    return cleaned[:12] or "unknown"


def canonical_json(payload: Any) -> bytes:
    """Encode Phase 1A evidence deterministically for download."""

    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def _cell(value: object) -> str:
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return "<NA>"
    return str(value)


def _rendered(frame: pd.DataFrame) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(_cell(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    )


def _csv_bytes(gold: pd.DataFrame) -> bytes:
    return gold.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")


def _parquet_bytes(gold: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    gold.to_parquet(buffer, engine="pyarrow", index=False, compression="snappy")
    return buffer.getvalue()


def _csv_round_trips(content: bytes, gold: pd.DataFrame) -> bool:
    """CSV preserves rendered tabular values, not Pandas dtype identity."""

    reloaded = pd.read_csv(
        StringIO(content.decode("utf-8")),
        index_col=False,
        skip_blank_lines=False,
    )
    return bool(
        tuple(reloaded.columns) == tuple(gold.columns)
        and reloaded.shape == gold.shape
        and _rendered(reloaded) == _rendered(gold)
    )


def _parquet_round_trips(content: bytes, gold: pd.DataFrame) -> bool:
    """Parquet preserves the gold values and schema exactly."""

    reloaded = pd.read_parquet(BytesIO(content), engine="pyarrow")
    return bool(
        tuple(reloaded.columns) == tuple(gold.columns)
        and reloaded.shape == gold.shape
        and identify_dataset(reloaded).fingerprint == identify_dataset(gold).fingerprint
    )


def gold_evidence_failure(runtime: object) -> ArtifactFailure | None:
    """Re-derive the PASS-only gate; return the blocking failure, or None."""

    if not isinstance(runtime, WorkflowRuntime):
        return _failure(ArtifactFailureCode.GOLD_UNAVAILABLE)
    state = runtime.state
    if state.stage is not WorkflowStage.QA_PASSED or runtime.gold_dataframe is None:
        return _failure(ArtifactFailureCode.GOLD_UNAVAILABLE)
    result = state.execution_result
    report = state.qa_report
    plan = state.transformation_plan
    identity = state.dataset_identity
    approval = state.human_approval
    accepted = state.accepted_review
    if (
        result is None
        or report is None
        or plan is None
        or identity is None
        or approval is None
        or accepted is None
        or result.result_fingerprint is None
    ):
        return _failure(ArtifactFailureCode.EVIDENCE_INCOMPLETE)
    if not result.success or report.status is not QAStatus.PASS:
        return _failure(ArtifactFailureCode.GOLD_EVIDENCE_MISMATCH)
    try:
        gold_identity = identify_dataset(runtime.gold_dataframe)
    except Exception:
        return _failure(ArtifactFailureCode.GOLD_EVIDENCE_MISMATCH)
    if (
        gold_identity.fingerprint != result.result_fingerprint
        or gold_identity.row_count != result.after_row_count
        or gold_identity.column_count != result.after_column_count
        or gold_identity.row_count != report.after_row_count
        or gold_identity.column_count != report.after_column_count
    ):
        return _failure(ArtifactFailureCode.GOLD_EVIDENCE_MISMATCH)
    if (
        result.dataset_id != identity.dataset_id
        or result.source_fingerprint != identity.fingerprint
        or result.plan_id != plan.plan_id
        or result.plan_version != plan.version
        or result.accepted_review_attempt != accepted.attempt
        or report.dataset_id != identity.dataset_id
        or report.plan_id != plan.plan_id
        or plan.dataset_id != identity.dataset_id
        or plan.dataset_fingerprint != identity.fingerprint
        or approval.decision is not HumanDecision.APPROVE
        or approval.plan_id != plan.plan_id
        or approval.plan_version != plan.version
        or approval.dataset_id != identity.dataset_id
        or approval.dataset_fingerprint != identity.fingerprint
        or accepted.plan_id != plan.plan_id
        or accepted.plan_version != plan.version
    ):
        return _failure(ArtifactFailureCode.GOLD_EVIDENCE_MISMATCH)
    return None


def _manifest_payload(
    runtime: WorkflowRuntime,
    source_metadata: SourceMetadata,
    downloads: tuple[DownloadArtifact, ...],
) -> dict[str, Any]:
    state = runtime.state
    result = state.execution_result
    report = state.qa_report
    plan = state.transformation_plan
    identity = state.dataset_identity
    approval = state.human_approval
    accepted = state.accepted_review
    assert result is not None and report is not None and plan is not None
    assert identity is not None and approval is not None and accepted is not None
    assert result.result_fingerprint is not None
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset_id": identity.dataset_id,
        "dataset_fingerprint": identity.fingerprint,
        "source_format": source_metadata.format.value,
        "source_parser_options": source_metadata.parser_options.model_dump(mode="json"),
        "plan_id": plan.plan_id,
        "plan_version": plan.version,
        "accepted_review_attempt": accepted.attempt,
        "qa_report_id": report.qa_report_id,
        "execution_id": result.execution_id,
        "result_fingerprint": result.result_fingerprint,
        "approved_at": approval.decided_at.isoformat(),
        "artifacts": [
            {
                "kind": artifact.kind.value,
                "filename": artifact.filename,
                "media_type": artifact.media_type,
                "byte_size": artifact.byte_size,
                "sha256": artifact.sha256,
            }
            for artifact in sorted(downloads, key=lambda item: item.kind.value)
        ],
    }


def build_artifact_set(
    runtime: object,
    source_metadata: object,
) -> ArtifactSet | ArtifactFailure:
    """Build the complete approved bundle, or refuse without emitting bytes."""

    blocked = gold_evidence_failure(runtime)
    if blocked is not None:
        return blocked
    assert isinstance(runtime, WorkflowRuntime)
    if not isinstance(source_metadata, SourceMetadata):
        return _failure(ArtifactFailureCode.SOURCE_METADATA_MISMATCH)
    state = runtime.state
    plan = state.transformation_plan
    identity = state.dataset_identity
    result = state.execution_result
    report = state.qa_report
    assert plan is not None and identity is not None
    assert result is not None and report is not None

    gold = runtime.gold_dataframe
    assert gold is not None
    gold = gold.reset_index(drop=True)
    dataset_short = _short_id(identity.dataset_id)
    plan_short = _short_id(plan.plan_id)
    stem = f"datachef_{dataset_short}_{plan_short}"

    def _artifact(
        kind: ArtifactKind,
        suffix: str,
        media_type: str,
        content: bytes,
    ) -> DownloadArtifact:
        return DownloadArtifact(
            kind=kind,
            filename=f"{stem}_{suffix}",
            media_type=media_type,
            content=content,
            dataset_id=identity.dataset_id,
            plan_id=plan.plan_id,
        )

    try:
        csv_content = _csv_bytes(gold)
        parquet_content = _parquet_bytes(gold)
        plan_content = canonical_json(plan.model_dump(mode="json"))
        qa_content = canonical_json(report.model_dump(mode="json"))
        change_log_content = canonical_json(result.model_dump(mode="json"))
        # Rendered inside the same guard as every other serializer: a render
        # failure refuses the whole bundle rather than shipping six of seven.
        pipeline_content = render_pipeline_bytes(plan)
    except Exception:
        return _failure(ArtifactFailureCode.SERIALIZATION_FAILURE)

    try:
        round_trips = _csv_round_trips(csv_content, gold) and _parquet_round_trips(
            parquet_content,
            gold,
        )
    except Exception:
        return _failure(ArtifactFailureCode.SERIALIZATION_FAILURE)
    if not round_trips:
        return _failure(ArtifactFailureCode.ROUND_TRIP_MISMATCH)

    downloads = (
        _artifact(ArtifactKind.CLEANED_CSV, "cleaned.csv", CSV_MEDIA_TYPE, csv_content),
        _artifact(
            ArtifactKind.CLEANED_PARQUET,
            "cleaned.parquet",
            PARQUET_MEDIA_TYPE,
            parquet_content,
        ),
        _artifact(
            ArtifactKind.TRANSFORMATION_PLAN_JSON,
            "transformation_plan.json",
            JSON_MEDIA_TYPE,
            plan_content,
        ),
        _artifact(
            ArtifactKind.QA_REPORT_JSON,
            "qa_report.json",
            JSON_MEDIA_TYPE,
            qa_content,
        ),
        _artifact(
            ArtifactKind.EXECUTION_CHANGE_LOG_JSON,
            "execution_change_log.json",
            JSON_MEDIA_TYPE,
            change_log_content,
        ),
        _artifact(
            ArtifactKind.PIPELINE_SCRIPT_PY,
            "pipeline.py",
            PIPELINE_MEDIA_TYPE,
            pipeline_content,
        ),
    )
    try:
        manifest_content = canonical_json(
            _manifest_payload(runtime, source_metadata, downloads)
        )
    except Exception:
        return _failure(ArtifactFailureCode.SERIALIZATION_FAILURE)
    manifest = _artifact(
        ArtifactKind.MANIFEST_JSON,
        "manifest.json",
        JSON_MEDIA_TYPE,
        manifest_content,
    )
    return ArtifactSet(
        cleaned_csv=downloads[0],
        cleaned_parquet=downloads[1],
        transformation_plan_json=downloads[2],
        qa_report_json=downloads[3],
        execution_change_log_json=downloads[4],
        pipeline_script=downloads[5],
        manifest=manifest,
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactFailure",
    "ArtifactFailureCode",
    "ArtifactKind",
    "ArtifactSet",
    "DownloadArtifact",
    "build_artifact_set",
    "canonical_json",
    "gold_evidence_failure",
]
