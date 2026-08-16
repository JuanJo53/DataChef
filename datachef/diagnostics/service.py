"""Typed adapter around the existing deterministic ingestion report."""

from __future__ import annotations

from hashlib import sha256
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
import json
import math
import re

import numpy as np
import pandas as pd

from crew.ingestion_agent.ingestion_agent import build_ingestion_report
from datachef.contracts import (
    ColumnProfile,
    ColumnSchema,
    DatasetIdentity,
    DiagnosticIssue,
    DiagnosticIssueKind,
    DiagnosticReport,
    IssueClassification,
    KeyDuplicateMetric,
    LegacyDiagnosticEvidence,
    MetricEvidence,
    OperationType,
    Severity,
)


class DatasetShapeFailure(StrEnum):
    DUPLICATE_COLUMN_LABEL = "DUPLICATE_COLUMN_LABEL"
    NON_STRING_COLUMN_LABEL = "NON_STRING_COLUMN_LABEL"
    UNSUPPORTED_OBJECT_VALUE = "UNSUPPORTED_OBJECT_VALUE"
    UNSUPPORTED_INDEX_VALUE = "UNSUPPORTED_INDEX_VALUE"


class DatasetShapeError(ValueError):
    """Sanitized rejection for data outside the Phase 1A fingerprint domain."""

    def __init__(self, failure: DatasetShapeFailure) -> None:
        super().__init__("Dataset shape is outside the supported fingerprint domain")
        self.failure = failure


def _is_supported_scalar(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, np.generic):
        value = value.item()
    return isinstance(
        value,
        (str, bytes, bool, int, float, Decimal, date, datetime, timedelta, pd.Timestamp, pd.Timedelta),
    )


def _validate_fingerprint_domain(dataframe: pd.DataFrame) -> None:
    columns = tuple(dataframe.columns)
    if any(not isinstance(column, str) for column in columns):
        raise DatasetShapeError(DatasetShapeFailure.NON_STRING_COLUMN_LABEL)
    if len(set(columns)) != len(columns):
        raise DatasetShapeError(DatasetShapeFailure.DUPLICATE_COLUMN_LABEL)
    for value in dataframe.to_numpy(dtype=object, copy=False).flat:
        if not _is_supported_scalar(value):
            raise DatasetShapeError(DatasetShapeFailure.UNSUPPORTED_OBJECT_VALUE)
    for column in columns:
        series = dataframe[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            for category in series.cat.categories:
                if not _is_supported_scalar(category):
                    raise DatasetShapeError(
                        DatasetShapeFailure.UNSUPPORTED_OBJECT_VALUE
                    )
    index_values = dataframe.index.to_numpy(dtype=object, copy=False)
    for value in index_values:
        if not _is_supported_scalar(value):
            raise DatasetShapeError(DatasetShapeFailure.UNSUPPORTED_INDEX_VALUE)


def _canonical_scalar_metadata(value: object) -> dict[str, object]:
    if value is None:
        return {"type": "none"}
    if value is pd.NA:
        return {"type": "pandas_na"}
    if value is pd.NaT:
        return {"type": "pandas_nat"}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if math.isnan(value):
            encoded: object = "nan"
        elif math.isinf(value):
            encoded = "positive_infinity" if value > 0 else "negative_infinity"
        else:
            encoded = value.hex()
        return {"type": "float", "value": encoded}
    if isinstance(value, Decimal):
        decimal = value.as_tuple()
        return {
            "type": "decimal",
            "sign": decimal.sign,
            "digits": list(decimal.digits),
            "exponent": decimal.exponent,
        }
    if isinstance(value, str):
        return {"type": "string", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, pd.Timestamp):
        return {"type": "pandas_timestamp", "value": value.isoformat()}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"type": "pandas_timedelta", "value": value.value}
    if isinstance(value, timedelta):
        return {
            "type": "timedelta",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    raise DatasetShapeError(DatasetShapeFailure.UNSUPPORTED_OBJECT_VALUE)


def _dtype_metadata(series: pd.Series) -> dict[str, object]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return {
            "kind": "categorical",
            "ordered": bool(series.cat.ordered),
            "categories": [
                _canonical_scalar_metadata(category)
                for category in series.cat.categories
            ],
        }
    return {"kind": "dtype", "value": str(series.dtype)}


def dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    """Hash supported scalar data; values, dtypes, ordering, and index are identity."""

    _validate_fingerprint_domain(dataframe)
    schema = [
        {"name": column, "dtype": _dtype_metadata(dataframe[column])}
        for column in dataframe
    ]
    schema_bytes = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = sha256(schema_bytes)
    value_hashes = pd.util.hash_pandas_object(
        dataframe,
        index=True,
        categorize=True,
    )
    digest.update(value_hashes.to_numpy().tobytes())
    return digest.hexdigest()


def identify_dataset(dataframe: pd.DataFrame) -> DatasetIdentity:
    fingerprint = dataframe_fingerprint(dataframe)
    schema = tuple(
        ColumnSchema(name=str(column), dtype=str(dataframe[column].dtype))
        for column in dataframe.columns
    )
    return DatasetIdentity(
        dataset_id=f"dataset-{fingerprint[:16]}",
        fingerprint=fingerprint,
        row_count=int(len(dataframe)),
        column_count=int(dataframe.shape[1]),
        column_schema=schema,
    )


def _severity(value: str) -> Severity:
    return Severity[value.upper()]


def _legacy_issue_to_contract(issue: dict[str, object]) -> DiagnosticIssue:
    issue_id = str(issue["id"])
    affected = tuple(
        match.group(1)
        for match in re.finditer(r"'([^']+)'", str(issue.get("title", "")))
    )
    count = int(issue.get("count", 0))
    if issue_id.startswith("nulls_"):
        kind = DiagnosticIssueKind.NULL_VALUES
        classification = IssueClassification.OBSERVED_DEFECT
        suggested = None
    elif issue_id == "dup_rows":
        kind = DiagnosticIssueKind.DUPLICATE_ROWS
        classification = IssueClassification.OBSERVED_DEFECT
        suggested = OperationType.DROP_DUPLICATE_ROWS
    elif issue_id.startswith("pii_"):
        kind = DiagnosticIssueKind.POSSIBLE_PII
        classification = IssueClassification.PRIVACY_RISK
        suggested = None
    else:
        kind = DiagnosticIssueKind.CANDIDATE_TYPE_CONVERSION
        classification = IssueClassification.CANDIDATE_CONVERSION
        suggested = OperationType.CAST_COLUMN
    return DiagnosticIssue(
        issue_id=issue_id,
        kind=kind,
        classification=classification,
        title=str(issue["title"]),
        severity=_severity(str(issue["severity"])),
        affected_columns=affected,
        evidence=(MetricEvidence(metric="affected_row_count", value=count),),
        suggested_operation=suggested,
        explanation=str(issue["detail"]),
    )


def _candidate_key_sets(
    dataframe: pd.DataFrame,
    selected_key_columns: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    candidates: list[tuple[str, ...]] = []
    if selected_key_columns and all(
        column in dataframe.columns for column in selected_key_columns
    ):
        candidates.append(selected_key_columns)
    for column in dataframe.columns:
        normalized = str(column).lower()
        if normalized == "id" or normalized.endswith("_id"):
            candidate = (str(column),)
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def _safe_issue_id(kind: DiagnosticIssueKind, columns: tuple[str, ...]) -> str:
    material = f"{kind.value}|{'|'.join(columns)}"
    return f"issue-{sha256(material.encode('utf-8')).hexdigest()[:16]}"


def diagnose_raw_dataframe(
    dataframe: pd.DataFrame,
    *,
    selected_key_columns: tuple[str, ...] = (),
) -> DiagnosticReport:
    """Diagnose an immutable raw snapshot without using legacy LLM fallbacks."""

    identity = identify_dataset(dataframe)
    legacy_report = build_ingestion_report(dataframe.copy(deep=True))
    profiles = tuple(
        ColumnProfile(
            name=str(item["name"]),
            dtype=str(item["dtype"]),
            sql_type=str(item["sql_type"]),
            null_count=int(item["nulls"]),
            null_pct=float(item["null_pct"]),
            unique_count=int(item["unique"]),
            is_primary_key_candidate=bool(item["is_pk_candidate"]),
            possible_pii=bool(item["is_pii"]),
        )
        for item in legacy_report["columns"]
    )
    issues = [_legacy_issue_to_contract(item) for item in legacy_report["issues"]]
    missing_selected_keys = tuple(
        column for column in selected_key_columns if column not in dataframe.columns
    )
    if missing_selected_keys:
        issues.append(
            DiagnosticIssue(
                issue_id=_safe_issue_id(
                    DiagnosticIssueKind.MISSING_KEY_COLUMN,
                    missing_selected_keys,
                ),
                kind=DiagnosticIssueKind.MISSING_KEY_COLUMN,
                classification=IssueClassification.OBSERVED_DEFECT,
                title="One or more selected key columns are unavailable.",
                severity=Severity.HIGH,
                affected_columns=missing_selected_keys,
                evidence=(
                    MetricEvidence(
                        metric="missing_selected_key_column_count",
                        value=len(missing_selected_keys),
                    ),
                ),
                explanation="The selected key cannot be evaluated against this schema.",
            )
        )
    key_metrics: list[KeyDuplicateMetric] = []
    for keys in _candidate_key_sets(dataframe, selected_key_columns):
        key_frame = dataframe.loc[:, list(keys)]
        null_key_mask = key_frame.isna().any(axis=1)
        null_key_count = int(null_key_mask.sum())
        non_null_keys = key_frame.loc[~null_key_mask]
        duplicate_count = int(non_null_keys.duplicated(keep="first").sum())
        key_metrics.append(
            KeyDuplicateMetric(
                key_columns=keys,
                duplicate_row_count=duplicate_count,
                null_key_row_count=null_key_count,
            )
        )
        if null_key_count:
            issues.append(
                DiagnosticIssue(
                    issue_id=_safe_issue_id(DiagnosticIssueKind.NULL_KEYS, keys),
                    kind=DiagnosticIssueKind.NULL_KEYS,
                    classification=IssueClassification.OBSERVED_DEFECT,
                    title="Null values detected in selected key columns.",
                    severity=Severity.HIGH,
                    affected_columns=keys,
                    evidence=(
                        MetricEvidence(
                            metric="null_key_row_count",
                            value=null_key_count,
                        ),
                    ),
                    explanation="Null-key rows remain unchanged and cannot be deduplicated safely.",
                )
            )
        if duplicate_count:
            joined = ", ".join(keys)
            issues.append(
                DiagnosticIssue(
                    issue_id=_safe_issue_id(DiagnosticIssueKind.DUPLICATE_KEYS, keys),
                    kind=DiagnosticIssueKind.DUPLICATE_KEYS,
                    classification=IssueClassification.OBSERVED_DEFECT,
                    title=f"Duplicate values detected for key columns: {joined}",
                    severity=Severity.HIGH,
                    affected_columns=keys,
                    evidence=(
                        MetricEvidence(
                            metric="duplicate_key_row_count",
                            value=duplicate_count,
                        ),
                    ),
                    suggested_operation=OperationType.DEDUPLICATE_BY_KEYS,
                    explanation=(
                        "Multiple rows share the selected key; no row has been removed."
                    ),
                )
            )
    health = legacy_report["health"]
    evidence = LegacyDiagnosticEvidence(
        health_score=int(health["score"]),
        health_grade=str(health["grade"]),
        completeness_pct=float(health["completeness_pct"]),
        uniqueness_pct=float(health["uniqueness_pct"]),
        suggested_primary_key=legacy_report["primary_key"],
    )
    issue_ids = "|".join(sorted(issue.issue_id for issue in issues))
    report_id = sha256(
        f"{identity.fingerprint}|{issue_ids}".encode("utf-8")
    ).hexdigest()
    return DiagnosticReport(
        report_id=f"diagnostic-{report_id[:16]}",
        dataset_identity=identity,
        column_profiles=profiles,
        issues=tuple(issues),
        duplicate_row_count=int(health["duplicate_rows"]),
        key_duplicate_metrics=tuple(key_metrics),
        legacy_evidence=evidence,
    )
