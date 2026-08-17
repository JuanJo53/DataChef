"""Strict Phase 1B application-boundary contracts.

Pydantic models in this module are safe, serializable control metadata. Uploaded
bytes and DataFrames deliberately remain runtime-only objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from datachef.contracts import (
    CastColumnParameters,
    CastTarget,
    DatasetIdentity,
    DeduplicateByKeysParameters,
    OperationType,
)


class StrictApplicationModel(BaseModel):
    """Base for immutable application metadata crossing component boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class UploadFormat(StrEnum):
    CSV = "CSV"
    JSON_RECORDS = "JSON_RECORDS"
    JSON_LINES = "JSON_LINES"
    PARQUET = "PARQUET"


class CsvParserOptions(StrictApplicationModel):
    kind: Literal["CSV"] = "CSV"
    encoding: Literal["utf-8", "utf-8-sig"] = "utf-8-sig"
    delimiter: Literal[","] = ","


class JsonRecordsParserOptions(StrictApplicationModel):
    kind: Literal["JSON_RECORDS"] = "JSON_RECORDS"
    encoding: Literal["utf-8", "utf-8-sig"] = "utf-8-sig"


class JsonLinesParserOptions(StrictApplicationModel):
    kind: Literal["JSON_LINES"] = "JSON_LINES"
    encoding: Literal["utf-8", "utf-8-sig"] = "utf-8-sig"


class ParquetParserOptions(StrictApplicationModel):
    kind: Literal["PARQUET"] = "PARQUET"
    engine: Literal["pyarrow"] = "pyarrow"


ParserOptions = Annotated[
    CsvParserOptions
    | JsonRecordsParserOptions
    | JsonLinesParserOptions
    | ParquetParserOptions,
    Field(discriminator="kind"),
]


def _options_format(options: ParserOptions) -> UploadFormat:
    return UploadFormat(options.kind)


class UploadPolicy(StrictApplicationModel):
    maximum_bytes: int = Field(default=25 * 1024 * 1024, gt=0)


@dataclass(frozen=True, slots=True)
class UploadRequest:
    """Runtime-only upload bytes plus explicit parsing choices."""

    content: bytes = field(repr=False)
    declared_suffix: str
    format: UploadFormat
    parser_options: ParserOptions

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise TypeError("upload content must be bytes")
        if not self.declared_suffix or not self.declared_suffix.startswith("."):
            raise ValueError("declared suffix must be a nonempty extension")
        if _options_format(self.parser_options) is not self.format:
            raise ValueError("format and parser options must agree")

    def __getstate__(self) -> object:
        raise TypeError("upload request is runtime-only and cannot be serialized")


class SourceMetadata(StrictApplicationModel):
    request_id: str = Field(min_length=1)
    format: UploadFormat
    byte_size: int = Field(ge=0)
    parser_options: ParserOptions

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request ID must contain non-whitespace text")
        return value

    @model_validator(mode="after")
    def validate_parser_options(self) -> "SourceMetadata":
        if _options_format(self.parser_options) is not self.format:
            raise ValueError("format and parser options must agree")
        return self


class ParsedDataset:
    """Runtime-only owner of one immutable-by-copy parsed source snapshot."""

    __slots__ = ("__metadata", "__identity", "__snapshot")

    def __init__(
        self,
        *,
        metadata: SourceMetadata,
        dataframe: pd.DataFrame,
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("parsed dataset requires a DataFrame")
        from datachef.diagnostics import identify_dataset

        snapshot = dataframe.copy(deep=True)
        identity = identify_dataset(snapshot)
        object.__setattr__(self, "_ParsedDataset__metadata", metadata)
        object.__setattr__(self, "_ParsedDataset__identity", identity)
        object.__setattr__(self, "_ParsedDataset__snapshot", snapshot)

    def __getattribute__(self, name: str) -> object:
        if name in {"_snapshot", "_ParsedDataset__snapshot"}:
            raise AttributeError("parsed dataset backing storage is not public")
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("parsed dataset is immutable")

    @property
    def metadata(self) -> SourceMetadata:
        return object.__getattribute__(self, "_ParsedDataset__metadata")

    @property
    def identity(self) -> DatasetIdentity:
        return object.__getattribute__(self, "_ParsedDataset__identity")

    def raw_copy(self) -> pd.DataFrame:
        """Return a new deep copy; callers never receive the stored snapshot."""

        snapshot = object.__getattribute__(self, "_ParsedDataset__snapshot")
        return snapshot.copy(deep=True)

    def __repr__(self) -> str:
        return (
            "ParsedDataset("
            f"metadata={self.metadata!r}, identity={self.identity!r}, "
            "dataframe=<private>)"
        )

    def __getstate__(self) -> object:
        raise TypeError("parsed dataset is runtime-only and cannot be serialized")


class UploadFailureCode(StrEnum):
    EMPTY_CONTENT = "EMPTY_CONTENT"
    SIZE_LIMIT_EXCEEDED = "SIZE_LIMIT_EXCEEDED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    FORMAT_MISMATCH = "FORMAT_MISMATCH"
    UNSUPPORTED_ENCODING = "UNSUPPORTED_ENCODING"
    MALFORMED_CSV = "MALFORMED_CSV"
    RAGGED_CSV_RECORD = "RAGGED_CSV_RECORD"
    MALFORMED_JSON = "MALFORMED_JSON"
    DUPLICATE_JSON_KEY = "DUPLICATE_JSON_KEY"
    NESTED_JSON_VALUE = "NESTED_JSON_VALUE"
    MALFORMED_PARQUET = "MALFORMED_PARQUET"
    EMPTY_DATASET = "EMPTY_DATASET"
    DUPLICATE_COLUMN_LABEL = "DUPLICATE_COLUMN_LABEL"
    INVALID_COLUMN_LABEL = "INVALID_COLUMN_LABEL"
    UNSUPPORTED_DATASET_VALUE = "UNSUPPORTED_DATASET_VALUE"
    PARSER_FAILURE = "PARSER_FAILURE"


class UploadFailure(StrictApplicationModel):
    code: UploadFailureCode
    safe_message: str = Field(min_length=1)
    suggested_action: str = Field(min_length=1)
    request_id: str | None = None


RequestedParameters = Annotated[
    CastColumnParameters | DeduplicateByKeysParameters,
    Field(discriminator="kind"),
]


class RequestedTransformation(StrictApplicationModel):
    """An explicit offline request the deterministic planner must account for."""

    request_id: str = Field(min_length=1)
    operation_type: OperationType
    target_columns: tuple[str, ...] = Field(min_length=1)
    parameters: RequestedParameters
    diagnostic_issue_id: str | None = Field(default=None, min_length=1)

    @field_validator("request_id", "diagnostic_issue_id")
    @classmethod
    def validate_optional_identifier(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("identifier must contain non-whitespace text")
        return value

    @field_validator("target_columns")
    @classmethod
    def validate_target_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not column.strip() for column in value):
            raise ValueError("target columns must contain non-whitespace text")
        return value

    @model_validator(mode="after")
    def validate_supported_request(self) -> "RequestedTransformation":
        if self.operation_type is OperationType.CAST_COLUMN:
            if not isinstance(self.parameters, CastColumnParameters):
                raise ValueError("cast requests require cast parameters")
            if self.parameters.target_type is not CastTarget.NUMERIC:
                raise ValueError("only numeric cast requests are available offline")
            if len(self.target_columns) != 1:
                raise ValueError("numeric cast requests target exactly one column")
        elif self.operation_type is OperationType.DEDUPLICATE_BY_KEYS:
            if not isinstance(self.parameters, DeduplicateByKeysParameters):
                raise ValueError("key deduplication requests require key parameters")
            if self.target_columns != self.parameters.keys:
                raise ValueError("deduplication targets and keys must match exactly")
            if any(not key.strip() for key in self.parameters.keys):
                raise ValueError("deduplication keys must contain non-whitespace text")
            if len(set(self.parameters.keys)) != len(self.parameters.keys):
                raise ValueError("deduplication keys must be unique")
        else:
            raise ValueError("requested operation is not available offline")
        return self


class ApplicationFinding(StrictApplicationModel):
    code: str = Field(min_length=1)
    blocking: bool
    safe_message: str = Field(min_length=1)
    request_id: str | None = Field(default=None, min_length=1)


class ScreenId(StrEnum):
    UPLOAD = "UPLOAD"
    DIAGNOSE = "DIAGNOSE"
    INTENT = "INTENT"
    PLAN = "PLAN"
    APPROVAL = "APPROVAL"
    QA = "QA"
    RESULTS = "RESULTS"


class CommandKind(StrEnum):
    PLAN_PREPARATION = "PLAN_PREPARATION"
    HUMAN_DECISION = "HUMAN_DECISION"
    EXECUTION = "EXECUTION"


class CommandOutcome(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class CommandAttempt(StrictApplicationModel):
    command_id: str = Field(min_length=1)
    kind: CommandKind
    binding_id: str = Field(min_length=1)
    outcome: CommandOutcome
    result_code: str = Field(min_length=1)

    @field_validator("command_id", "binding_id", "result_code")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command evidence text must not be whitespace")
        return value


class TransitionResult(StrictApplicationModel):
    changed: bool
    screen: ScreenId
    code: str = Field(min_length=1)
    findings: tuple[ApplicationFinding, ...] = Field(default_factory=tuple)
    revision: int = Field(default=0, ge=0)


__all__ = [
    "ApplicationFinding",
    "CommandAttempt",
    "CommandKind",
    "CommandOutcome",
    "CsvParserOptions",
    "JsonLinesParserOptions",
    "JsonRecordsParserOptions",
    "ParsedDataset",
    "ParquetParserOptions",
    "ParserOptions",
    "RequestedTransformation",
    "ScreenId",
    "SourceMetadata",
    "TransitionResult",
    "UploadFailure",
    "UploadFailureCode",
    "UploadFormat",
    "UploadPolicy",
    "UploadRequest",
]
