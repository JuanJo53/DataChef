from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from datachef.application import (
    CsvParserOptions,
    JsonLinesParserOptions,
    JsonRecordsParserOptions,
    ParquetParserOptions,
    ParsedDataset,
    UploadFailure,
    UploadFailureCode,
    UploadFormat,
    UploadPolicy,
    UploadRequest,
    parse_upload,
)
from datachef.contracts import DiagnosticIssueKind
from datachef.diagnostics import diagnose_raw_dataframe


POLICY = UploadPolicy(maximum_bytes=1024 * 1024)


def _request(
    content: bytes,
    format_: UploadFormat,
    options,
    suffix: str,
) -> UploadRequest:
    return UploadRequest(
        content=content,
        declared_suffix=suffix,
        format=format_,
        parser_options=options,
    )


def _success(result: ParsedDataset | UploadFailure) -> ParsedDataset:
    assert isinstance(result, ParsedDataset), result
    return result


def _failure(
    result: ParsedDataset | UploadFailure,
    code: UploadFailureCode,
) -> UploadFailure:
    assert isinstance(result, UploadFailure), result
    assert result.code is code
    return result


def _parquet_bytes(dataframe: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    dataframe.to_parquet(buffer, engine="pyarrow", index=False)
    return buffer.getvalue()


def test_csv_json_records_json_lines_and_parquet_parse_equivalent_data() -> None:
    expected = pd.DataFrame({"category": ["A", "B"], "measure": [1, 2]})
    requests = (
        _request(
            b"category,measure\nA,1\nB,2\n",
            UploadFormat.CSV,
            CsvParserOptions(),
            ".csv",
        ),
        _request(
            b'[{"category":"A","measure":1},{"category":"B","measure":2}]',
            UploadFormat.JSON_RECORDS,
            JsonRecordsParserOptions(),
            ".json",
        ),
        _request(
            b'{"category":"A","measure":1}\n{"category":"B","measure":2}\n',
            UploadFormat.JSON_LINES,
            JsonLinesParserOptions(),
            ".jsonl",
        ),
        _request(
            _parquet_bytes(expected),
            UploadFormat.PARQUET,
            ParquetParserOptions(),
            ".parquet",
        ),
    )

    parsed = tuple(_success(parse_upload(request, POLICY)) for request in requests)

    for dataset in parsed:
        assert_frame_equal(dataset.raw_copy(), expected, check_dtype=False)
        assert dataset.identity.row_count == 2
        assert dataset.identity.column_count == 2


def test_csv_header_respects_quoting_and_utf8_bom() -> None:
    content = b'\xef\xbb\xbf"order,id",value\n1,ok\n'
    result = _success(
        parse_upload(
            _request(content, UploadFormat.CSV, CsvParserOptions(), ".csv"),
            POLICY,
        )
    )

    assert result.raw_copy().columns.tolist() == ["order,id", "value"]


@pytest.mark.parametrize(
    "content",
    (
        b"a,b\n1,2,3\n",
        b"a,b\n1\n",
        b"a,b\n   \n",
        b"a,b\n1,2\n3\n",
        b"a,b\n\n1,2\n\n3,4,5\n",
        b"a,b\r\n1,2\r\n3\r\n",
    ),
)
def test_csv_rejects_every_nonblank_ragged_record_before_fingerprinting(
    content: bytes,
    monkeypatch,
) -> None:
    calls = {"fingerprint": 0}

    def forbidden_identity(dataframe):
        del dataframe
        calls["fingerprint"] += 1
        raise AssertionError("ragged CSV reached dataset identity")

    monkeypatch.setattr("datachef.diagnostics.identify_dataset", forbidden_identity)

    failure = _failure(
        parse_upload(
            _request(content, UploadFormat.CSV, CsvParserOptions(), ".csv"),
            POLICY,
        ),
        UploadFailureCode.RAGGED_CSV_RECORD,
    )

    assert calls == {"fingerprint": 0}
    assert "1,2,3" not in failure.safe_message


@pytest.mark.parametrize(
    ("content", "expected"),
    (
        (b'a,b\n"x,y",z\n', pd.DataFrame({"a": ["x,y"], "b": ["z"]})),
        (b'a,b\n"line 1\nline 2",z\n', pd.DataFrame({"a": ["line 1\nline 2"], "b": ["z"]})),
        (b"a,b\r\n\r\n1,2\r\n", pd.DataFrame({"a": [1], "b": [2]})),
        (b"a,b,c\n1,2,\n", pd.DataFrame({"a": [1], "b": [2], "c": [float("nan")]})),
        (b"a\n   \n", pd.DataFrame({"a": ["   "]})),
    ),
)
def test_csv_structural_gate_preserves_valid_csv_records(
    content: bytes,
    expected: pd.DataFrame,
) -> None:
    parsed = _success(
        parse_upload(
            _request(content, UploadFormat.CSV, CsvParserOptions(), ".csv"),
            POLICY,
        )
    )

    assert_frame_equal(parsed.raw_copy(), expected, check_dtype=False)


@pytest.mark.parametrize(
    ("content", "code"),
    (
        (b"\xff\xfevalue\n1\n", UploadFailureCode.UNSUPPORTED_ENCODING),
        (b"value\x00other\n1,2\n", UploadFailureCode.MALFORMED_CSV),
        (b'"unterminated,value\n1,2\n', UploadFailureCode.MALFORMED_CSV),
        (b"a,a\n1,2\n", UploadFailureCode.DUPLICATE_COLUMN_LABEL),
        (b"a,\n1,2\n", UploadFailureCode.INVALID_COLUMN_LABEL),
    ),
)
def test_csv_rejects_unsafe_or_malformed_inputs(
    content: bytes,
    code: UploadFailureCode,
) -> None:
    failure = _failure(
        parse_upload(
            _request(content, UploadFormat.CSV, CsvParserOptions(), ".csv"),
            POLICY,
        ),
        code,
    )

    assert content.decode("utf-8", errors="ignore") not in failure.safe_message


def test_json_modes_reject_duplicate_keys_nested_values_and_wrong_shapes() -> None:
    duplicate = b'[{"category":"A","category":"B"}]'
    nested = b'[{"category":{"unsafe":"value"}}]'
    scalar = b'"not records"'
    lines_array = b'[{"category":"A"}]\n'

    _failure(
        parse_upload(
            _request(
                duplicate,
                UploadFormat.JSON_RECORDS,
                JsonRecordsParserOptions(),
                ".json",
            ),
            POLICY,
        ),
        UploadFailureCode.DUPLICATE_JSON_KEY,
    )
    _failure(
        parse_upload(
            _request(
                nested,
                UploadFormat.JSON_RECORDS,
                JsonRecordsParserOptions(),
                ".json",
            ),
            POLICY,
        ),
        UploadFailureCode.NESTED_JSON_VALUE,
    )
    _failure(
        parse_upload(
            _request(
                scalar,
                UploadFormat.JSON_RECORDS,
                JsonRecordsParserOptions(),
                ".json",
            ),
            POLICY,
        ),
        UploadFailureCode.MALFORMED_JSON,
    )
    _failure(
        parse_upload(
            _request(
                lines_array,
                UploadFormat.JSON_LINES,
                JsonLinesParserOptions(),
                ".jsonl",
            ),
            POLICY,
        ),
        UploadFailureCode.MALFORMED_JSON,
    )


def test_json_lines_does_not_retry_records_and_records_does_not_retry_lines() -> None:
    records = b'[{"a":1},{"a":2}]'
    lines = b'{"a":1}\n{"a":2}\n'

    _failure(
        parse_upload(
            _request(records, UploadFormat.JSON_LINES, JsonLinesParserOptions(), ".jsonl"),
            POLICY,
        ),
        UploadFailureCode.MALFORMED_JSON,
    )
    _failure(
        parse_upload(
            _request(lines, UploadFormat.JSON_RECORDS, JsonRecordsParserOptions(), ".json"),
            POLICY,
        ),
        UploadFailureCode.MALFORMED_JSON,
    )


@pytest.mark.parametrize("constant", (b"NaN", b"Infinity", b"-Infinity"))
@pytest.mark.parametrize(
    ("format_", "options", "suffix", "template"),
    (
        (
            UploadFormat.JSON_RECORDS,
            JsonRecordsParserOptions(),
            ".json",
            b'[{"value":%s}]',
        ),
        (
            UploadFormat.JSON_LINES,
            JsonLinesParserOptions(),
            ".jsonl",
            b'{"value":%s}\n',
        ),
    ),
)
def test_json_modes_reject_nonstandard_numeric_constants(
    constant: bytes,
    format_: UploadFormat,
    options,
    suffix: str,
    template: bytes,
) -> None:
    failure = _failure(
        parse_upload(
            _request(template % constant, format_, options, suffix),
            POLICY,
        ),
        UploadFailureCode.MALFORMED_JSON,
    )

    assert constant.decode("ascii") not in failure.safe_message


def test_parquet_requires_magic_and_rejects_unsupported_objects() -> None:
    malformed = _request(
        b"not parquet",
        UploadFormat.PARQUET,
        ParquetParserOptions(),
        ".parquet",
    )
    unsupported = pd.DataFrame({"nested": [[1, 2]]})

    _failure(parse_upload(malformed, POLICY), UploadFailureCode.MALFORMED_PARQUET)
    _failure(
        parse_upload(
            _request(
                _parquet_bytes(unsupported),
                UploadFormat.PARQUET,
                ParquetParserOptions(),
                ".parquet",
            ),
            POLICY,
        ),
        UploadFailureCode.UNSUPPORTED_DATASET_VALUE,
    )


def test_common_boundary_rejects_empty_oversized_mismatched_and_empty_dataset() -> None:
    _failure(
        parse_upload(
            _request(b"", UploadFormat.CSV, CsvParserOptions(), ".csv"), POLICY
        ),
        UploadFailureCode.EMPTY_CONTENT,
    )
    oversized = _failure(
        parse_upload(
            _request(b"a\n" + b"1" * 20, UploadFormat.CSV, CsvParserOptions(), ".csv"),
            UploadPolicy(maximum_bytes=4),
        ),
        UploadFailureCode.SIZE_LIMIT_EXCEEDED,
    )
    assert oversized.request_id is None
    _failure(
        parse_upload(
            _request(b"a\n1\n", UploadFormat.CSV, CsvParserOptions(), ".json"),
            POLICY,
        ),
        UploadFailureCode.FORMAT_MISMATCH,
    )
    _failure(
        parse_upload(
            _request(b"a\n1\n", UploadFormat.CSV, CsvParserOptions(), ".txt"),
            POLICY,
        ),
        UploadFailureCode.UNSUPPORTED_FORMAT,
    )
    _failure(
        parse_upload(
            _request(b"a\n", UploadFormat.CSV, CsvParserOptions(), ".csv"), POLICY
        ),
        UploadFailureCode.EMPTY_DATASET,
    )
    _failure(
        parse_upload(
            _request(b"[{}]", UploadFormat.JSON_RECORDS, JsonRecordsParserOptions(), ".json"),
            POLICY,
        ),
        UploadFailureCode.EMPTY_DATASET,
    )


def test_request_identity_is_stable_and_includes_parser_options() -> None:
    content = b"value\n1\n"
    default = _success(
        parse_upload(
            _request(content, UploadFormat.CSV, CsvParserOptions(), ".csv"), POLICY
        )
    )
    repeated = _success(
        parse_upload(
            _request(content, UploadFormat.CSV, CsvParserOptions(), ".csv"), POLICY
        )
    )
    changed = _success(
        parse_upload(
            _request(
                content,
                UploadFormat.CSV,
                CsvParserOptions(encoding="utf-8"),
                ".csv",
            ),
            POLICY,
        )
    )

    assert default.metadata.request_id == repeated.metadata.request_id
    assert default.metadata.request_id != changed.metadata.request_id


def test_parse_stores_defensive_copy_and_creates_no_product_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = _request(
        b"value\n1\n",
        UploadFormat.CSV,
        CsvParserOptions(),
        ".csv",
    )
    parsed = _success(parse_upload(request, POLICY))
    copy = parsed.raw_copy()
    copy.loc[0, "value"] = 999

    assert parsed.raw_copy().loc[0, "value"] == 1
    assert list(tmp_path.iterdir()) == []


def test_phase1b_fixture_is_fictional_and_supports_key_and_cast_diagnostics() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "phase1b_orders.csv"
    parsed = _success(
        parse_upload(
            _request(
                fixture.read_bytes(),
                UploadFormat.CSV,
                CsvParserOptions(),
                ".csv",
            ),
            POLICY,
        )
    )
    frame = parsed.raw_copy()
    report = diagnose_raw_dataframe(frame, selected_key_columns=("order_id",))

    assert frame["order_id"].duplicated().any()
    assert str(frame["amount_text"].dtype) in {"object", "str"}
    assert any(
        issue.kind is DiagnosticIssueKind.CANDIDATE_TYPE_CONVERSION
        and issue.affected_columns == ("amount_text",)
        for issue in report.issues
    )
    assert any(
        issue.kind is DiagnosticIssueKind.DUPLICATE_KEYS
        and issue.affected_columns == ("order_id",)
        for issue in report.issues
    )
