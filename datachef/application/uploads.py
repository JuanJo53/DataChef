"""In-memory, typed upload parsing for CSV, JSON, JSON Lines, and Parquet."""

from __future__ import annotations

import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from typing import Any

import pandas as pd

from datachef.application.models import (
    CsvParserOptions,
    JsonLinesParserOptions,
    JsonRecordsParserOptions,
    ParsedDataset,
    ParquetParserOptions,
    SourceMetadata,
    UploadFailure,
    UploadFailureCode,
    UploadFormat,
    UploadPolicy,
    UploadRequest,
)
from datachef.diagnostics import DatasetShapeError, DatasetShapeFailure


_SUPPORTED_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".ndjson", ".parquet"})
_FORMAT_SUFFIXES = {
    UploadFormat.CSV: frozenset({".csv"}),
    UploadFormat.JSON_RECORDS: frozenset({".json"}),
    UploadFormat.JSON_LINES: frozenset({".json", ".jsonl", ".ndjson"}),
    UploadFormat.PARQUET: frozenset({".parquet"}),
}

_SAFE_MESSAGES = {
    UploadFailureCode.EMPTY_CONTENT: (
        "The upload is empty.",
        "Choose a nonempty CSV, JSON, JSON Lines, or Parquet file.",
    ),
    UploadFailureCode.SIZE_LIMIT_EXCEEDED: (
        "The upload exceeds the configured size limit.",
        "Choose a smaller file or ask an administrator to review the local limit.",
    ),
    UploadFailureCode.UNSUPPORTED_FORMAT: (
        "The declared file extension is not supported.",
        "Choose a .csv, .json, .jsonl, .ndjson, or .parquet file.",
    ),
    UploadFailureCode.FORMAT_MISMATCH: (
        "The declared extension does not match the selected parser.",
        "Select the parser that matches the file format.",
    ),
    UploadFailureCode.UNSUPPORTED_ENCODING: (
        "The text upload is not valid UTF-8.",
        "Export the source as UTF-8 or UTF-8 with BOM and try again.",
    ),
    UploadFailureCode.MALFORMED_CSV: (
        "The CSV structure is malformed or contains binary content.",
        "Check quoting, delimiters, and the header row.",
    ),
    UploadFailureCode.RAGGED_CSV_RECORD: (
        "A CSV record does not contain the same number of fields as the header.",
        "Make every nonblank CSV record match the header width and try again.",
    ),
    UploadFailureCode.MALFORMED_JSON: (
        "The JSON structure does not match the selected JSON mode.",
        "Use a list of objects for JSON Records or one object per line for JSON Lines.",
    ),
    UploadFailureCode.DUPLICATE_JSON_KEY: (
        "A JSON object contains a duplicate key.",
        "Make every object key unique before uploading.",
    ),
    UploadFailureCode.NESTED_JSON_VALUE: (
        "Nested JSON cell values are not supported in this offline slice.",
        "Flatten nested arrays or objects into scalar columns before uploading.",
    ),
    UploadFailureCode.MALFORMED_PARQUET: (
        "The upload is not a readable Parquet file.",
        "Export a valid Parquet file with standard PAR1 framing.",
    ),
    UploadFailureCode.EMPTY_DATASET: (
        "The parsed dataset has no rows or no columns.",
        "Provide a dataset containing at least one row and one named column.",
    ),
    UploadFailureCode.DUPLICATE_COLUMN_LABEL: (
        "The dataset contains duplicate column labels.",
        "Rename duplicate columns at the source before uploading.",
    ),
    UploadFailureCode.INVALID_COLUMN_LABEL: (
        "Every column label must be a unique nonempty string.",
        "Provide nonempty text labels for every column.",
    ),
    UploadFailureCode.UNSUPPORTED_DATASET_VALUE: (
        "The dataset contains a value type outside the supported scalar domain.",
        "Flatten nested or custom objects into CSV/JSON/Parquet scalar values.",
    ),
    UploadFailureCode.PARSER_FAILURE: (
        "The selected parser could not read this upload.",
        "Verify the format and parser selection, then try again.",
    ),
}


class _DuplicateJsonKey(ValueError):
    pass


class _NestedJsonValue(ValueError):
    pass


class _NonstandardJsonConstant(ValueError):
    pass


def _failure(
    code: UploadFailureCode,
    *,
    request_id: str | None = None,
) -> UploadFailure:
    message, action = _SAFE_MESSAGES[code]
    return UploadFailure(
        code=code,
        safe_message=message,
        suggested_action=action,
        request_id=request_id,
    )


def _format_gate(request: UploadRequest) -> UploadFailure | None:
    suffix = request.declared_suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        return _failure(UploadFailureCode.UNSUPPORTED_FORMAT)
    if suffix not in _FORMAT_SUFFIXES[request.format]:
        return _failure(UploadFailureCode.FORMAT_MISMATCH)
    return None


def _request_digest(request: UploadRequest) -> str:
    options = request.parser_options.model_dump_json(exclude_none=False)
    digest = sha256()
    digest.update(request.content)
    digest.update(b"\x00")
    digest.update(request.format.value.encode("ascii"))
    digest.update(b"\x00")
    digest.update(options.encode("utf-8"))
    return f"upload-{digest.hexdigest()}"


def upload_request_id(
    request: UploadRequest,
    policy: UploadPolicy,
) -> str | UploadFailure:
    """Return a local idempotency key only after cheap safety gates pass."""

    if not request.content:
        return _failure(UploadFailureCode.EMPTY_CONTENT)
    if len(request.content) > policy.maximum_bytes:
        return _failure(UploadFailureCode.SIZE_LIMIT_EXCEEDED)
    mismatch = _format_gate(request)
    if mismatch is not None:
        return mismatch
    return _request_digest(request)


def source_metadata_for_upload(
    request: UploadRequest,
    policy: UploadPolicy,
) -> SourceMetadata | UploadFailure:
    """Build the authoritative, content-bound metadata for one upload request."""

    request_identity = upload_request_id(request, policy)
    if isinstance(request_identity, UploadFailure):
        return request_identity
    return SourceMetadata(
        request_id=request_identity,
        format=request.format,
        byte_size=len(request.content),
        parser_options=request.parser_options,
    )


def _decode(content: bytes, encoding: str) -> str | UploadFailure:
    try:
        return content.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        return _failure(UploadFailureCode.UNSUPPORTED_ENCODING)


def _parse_csv(request: UploadRequest) -> pd.DataFrame | UploadFailure:
    options = request.parser_options
    if not isinstance(options, CsvParserOptions):
        return _failure(UploadFailureCode.FORMAT_MISMATCH)
    if b"\x00" in request.content:
        return _failure(UploadFailureCode.MALFORMED_CSV)
    decoded = _decode(request.content, options.encoding)
    if isinstance(decoded, UploadFailure):
        return decoded
    try:
        rows = csv.reader(StringIO(decoded), delimiter=options.delimiter, strict=True)
        header = next(rows)
        header_width = len(header)
        data_rows: list[list[str]] = []
        for row in rows:
            if not row:
                continue
            if len(row) != header_width:
                return _failure(UploadFailureCode.RAGGED_CSV_RECORD)
            data_rows.append(row)
    except (csv.Error, StopIteration):
        return _failure(UploadFailureCode.MALFORMED_CSV)
    normalized_header = tuple(label.lstrip("\ufeff") for label in header)
    if any(not label.strip() for label in normalized_header):
        return _failure(UploadFailureCode.INVALID_COLUMN_LABEL)
    if len(set(normalized_header)) != len(normalized_header):
        return _failure(UploadFailureCode.DUPLICATE_COLUMN_LABEL)
    try:
        canonical = StringIO()
        writer = csv.writer(
            canonical,
            delimiter=options.delimiter,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writerow(normalized_header)
        writer.writerows(data_rows)
        canonical.seek(0)
        dataframe = pd.read_csv(
            canonical,
            sep=options.delimiter,
            index_col=False,
            on_bad_lines="error",
            skip_blank_lines=False,
        )
    except Exception:
        return _failure(UploadFailureCode.MALFORMED_CSV)
    if tuple(dataframe.columns) != normalized_header:
        return _failure(UploadFailureCode.INVALID_COLUMN_LABEL)
    return dataframe


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    del value
    raise _NonstandardJsonConstant


def _load_json(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_json_constant,
    )


def _validate_flat_records(records: list[dict[str, Any]]) -> None:
    for record in records:
        for value in record.values():
            if isinstance(value, (dict, list)):
                raise _NestedJsonValue


def _parse_json_records(request: UploadRequest) -> pd.DataFrame | UploadFailure:
    options = request.parser_options
    if not isinstance(options, JsonRecordsParserOptions):
        return _failure(UploadFailureCode.FORMAT_MISMATCH)
    decoded = _decode(request.content, options.encoding)
    if isinstance(decoded, UploadFailure):
        return decoded
    try:
        payload = _load_json(decoded)
        if not isinstance(payload, list) or any(
            not isinstance(item, dict) for item in payload
        ):
            return _failure(UploadFailureCode.MALFORMED_JSON)
        _validate_flat_records(payload)
        return pd.DataFrame.from_records(payload)
    except _DuplicateJsonKey:
        return _failure(UploadFailureCode.DUPLICATE_JSON_KEY)
    except _NestedJsonValue:
        return _failure(UploadFailureCode.NESTED_JSON_VALUE)
    except (
        _NonstandardJsonConstant,
        json.JSONDecodeError,
        UnicodeError,
        TypeError,
        ValueError,
    ):
        return _failure(UploadFailureCode.MALFORMED_JSON)


def _parse_json_lines(request: UploadRequest) -> pd.DataFrame | UploadFailure:
    options = request.parser_options
    if not isinstance(options, JsonLinesParserOptions):
        return _failure(UploadFailureCode.FORMAT_MISMATCH)
    decoded = _decode(request.content, options.encoding)
    if isinstance(decoded, UploadFailure):
        return decoded
    records: list[dict[str, Any]] = []
    try:
        for line in decoded.splitlines():
            if not line.strip():
                continue
            payload = _load_json(line)
            if not isinstance(payload, dict):
                return _failure(UploadFailureCode.MALFORMED_JSON)
            records.append(payload)
        _validate_flat_records(records)
        return pd.DataFrame.from_records(records)
    except _DuplicateJsonKey:
        return _failure(UploadFailureCode.DUPLICATE_JSON_KEY)
    except _NestedJsonValue:
        return _failure(UploadFailureCode.NESTED_JSON_VALUE)
    except (
        _NonstandardJsonConstant,
        json.JSONDecodeError,
        UnicodeError,
        TypeError,
        ValueError,
    ):
        return _failure(UploadFailureCode.MALFORMED_JSON)


def _parse_parquet(request: UploadRequest) -> pd.DataFrame | UploadFailure:
    options = request.parser_options
    if not isinstance(options, ParquetParserOptions):
        return _failure(UploadFailureCode.FORMAT_MISMATCH)
    if not (
        request.content.startswith(b"PAR1") and request.content.endswith(b"PAR1")
    ):
        return _failure(UploadFailureCode.MALFORMED_PARQUET)
    try:
        return pd.read_parquet(BytesIO(request.content), engine=options.engine)
    except Exception:
        return _failure(UploadFailureCode.MALFORMED_PARQUET)


def _translate_dataset_shape(error: DatasetShapeError) -> UploadFailure:
    mapping = {
        DatasetShapeFailure.DUPLICATE_COLUMN_LABEL: UploadFailureCode.DUPLICATE_COLUMN_LABEL,
        DatasetShapeFailure.NON_STRING_COLUMN_LABEL: UploadFailureCode.INVALID_COLUMN_LABEL,
        DatasetShapeFailure.UNSUPPORTED_OBJECT_VALUE: UploadFailureCode.UNSUPPORTED_DATASET_VALUE,
        DatasetShapeFailure.UNSUPPORTED_INDEX_VALUE: UploadFailureCode.UNSUPPORTED_DATASET_VALUE,
    }
    return _failure(mapping[error.failure])


def parse_upload(
    request: UploadRequest,
    policy: UploadPolicy,
) -> ParsedDataset | UploadFailure:
    """Parse untrusted bytes without cleanup and bind the raw snapshot immediately."""

    metadata = source_metadata_for_upload(request, policy)
    if isinstance(metadata, UploadFailure):
        return metadata
    parsers = {
        UploadFormat.CSV: _parse_csv,
        UploadFormat.JSON_RECORDS: _parse_json_records,
        UploadFormat.JSON_LINES: _parse_json_lines,
        UploadFormat.PARQUET: _parse_parquet,
    }
    try:
        parsed = parsers[request.format](request)
    except Exception:
        return _failure(UploadFailureCode.PARSER_FAILURE, request_id=metadata.request_id)
    if isinstance(parsed, UploadFailure):
        return parsed.model_copy(update={"request_id": metadata.request_id})
    if parsed.shape[0] == 0 or parsed.shape[1] == 0:
        return _failure(UploadFailureCode.EMPTY_DATASET, request_id=metadata.request_id)
    try:
        return ParsedDataset(metadata=metadata, dataframe=parsed)
    except DatasetShapeError as error:
        failure = _translate_dataset_shape(error)
        return failure.model_copy(update={"request_id": metadata.request_id})
    except Exception:
        return _failure(UploadFailureCode.PARSER_FAILURE, request_id=metadata.request_id)


__all__ = ["parse_upload", "source_metadata_for_upload", "upload_request_id"]
