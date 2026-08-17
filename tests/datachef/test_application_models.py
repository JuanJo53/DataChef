from __future__ import annotations

from dataclasses import asdict
import pickle

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from datachef.application import (
    ApplicationFinding,
    CsvParserOptions,
    JsonLinesParserOptions,
    ParsedDataset,
    RequestedTransformation,
    ScreenId,
    SourceMetadata,
    TransitionResult,
    UploadFormat,
    UploadPolicy,
    UploadRequest,
)
from datachef.contracts import (
    CastColumnParameters,
    CastTarget,
    DeduplicateByKeysParameters,
    KeepPolicy,
    OperationType,
)


def test_pydantic_application_contracts_are_strict_frozen_and_forbid_extras() -> None:
    policy = UploadPolicy(maximum_bytes=1024)

    with pytest.raises(ValidationError):
        UploadPolicy(maximum_bytes="1024")
    with pytest.raises(ValidationError):
        UploadPolicy(maximum_bytes=1024, unexpected=True)
    with pytest.raises(ValidationError):
        policy.maximum_bytes = 2048


def test_parser_options_must_match_the_selected_format() -> None:
    with pytest.raises(ValueError, match="format and parser options"):
        UploadRequest(
            content=b"value\n1\n",
            declared_suffix=".csv",
            format=UploadFormat.CSV,
            parser_options=JsonLinesParserOptions(),
        )

    with pytest.raises(ValidationError):
        SourceMetadata(
            request_id="request-1",
            format=UploadFormat.CSV,
            byte_size=8,
            parser_options=JsonLinesParserOptions(),
        )


def test_identifiers_and_sizes_are_validated() -> None:
    with pytest.raises(ValidationError):
        UploadPolicy(maximum_bytes=0)
    with pytest.raises(ValidationError):
        SourceMetadata(
            request_id="",
            format=UploadFormat.CSV,
            byte_size=-1,
            parser_options=CsvParserOptions(),
        )
    with pytest.raises(ValidationError):
        SourceMetadata(
            request_id="   ",
            format=UploadFormat.CSV,
            byte_size=8,
            parser_options=CsvParserOptions(),
        )


def test_upload_request_retains_only_suffix_and_hides_bytes_from_repr() -> None:
    request = UploadRequest(
        content=b"value\n1\n",
        declared_suffix=".csv",
        format=UploadFormat.CSV,
        parser_options=CsvParserOptions(),
    )

    assert request.declared_suffix == ".csv"
    assert "content" not in repr(request)
    assert not hasattr(request, "filename")
    assert not hasattr(request, "path")


def test_parsed_dataset_keeps_a_private_snapshot_and_returns_fresh_copies() -> None:
    source = pd.DataFrame({"value": [1]})
    parsed = ParsedDataset(
        metadata=SourceMetadata(
            request_id="request-1",
            format=UploadFormat.CSV,
            byte_size=8,
            parser_options=CsvParserOptions(),
        ),
        dataframe=source,
    )
    source.loc[0, "value"] = 99
    first = parsed.raw_copy()
    first.loc[0, "value"] = 88

    assert_frame_equal(parsed.raw_copy(), pd.DataFrame({"value": [1]}))
    assert parsed.identity.row_count == 1
    assert not hasattr(parsed, "dataframe")
    assert not hasattr(parsed, "_snapshot")
    assert not hasattr(parsed, "_ParsedDataset__snapshot")
    assert not hasattr(parsed, "model_dump")
    assert "99" not in repr(parsed)
    with pytest.raises(TypeError):
        asdict(parsed)
    with pytest.raises(TypeError, match="runtime-only"):
        pickle.dumps(parsed)


def test_parsed_dataset_computes_identity_and_rejects_identity_injection() -> None:
    metadata = SourceMetadata(
        request_id="request-identity",
        format=UploadFormat.CSV,
        byte_size=8,
        parser_options=CsvParserOptions(),
    )
    frame = pd.DataFrame(
        {
            "category": pd.Categorical(
                ["a", "b"], categories=["a", "b"], ordered=True
            )
        },
        index=pd.Index([10, 20], name="row_id"),
    )
    parsed = ParsedDataset(metadata=metadata, dataframe=frame)
    original_fingerprint = parsed.identity.fingerprint

    frame.loc[10, "category"] = "b"
    frame.index = [1, 2]
    copy_one = parsed.raw_copy()
    copy_one["category"] = copy_one["category"].cat.reorder_categories(["b", "a"])
    copy_one.index = [3, 4]

    copy_two = parsed.raw_copy()
    assert list(copy_two.index) == [10, 20]
    assert list(copy_two["category"].cat.categories) == ["a", "b"]
    assert parsed.identity.fingerprint == original_fingerprint
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ParsedDataset(metadata=metadata, dataframe=copy_two, identity=parsed.identity)


def test_only_numeric_cast_and_key_dedup_requests_are_representable() -> None:
    cast_request = RequestedTransformation(
        request_id="cast-amount",
        operation_type=OperationType.CAST_COLUMN,
        target_columns=("amount_text",),
        parameters=CastColumnParameters(target_type=CastTarget.NUMERIC),
    )
    dedup_request = RequestedTransformation(
        request_id="dedup-order",
        operation_type=OperationType.DEDUPLICATE_BY_KEYS,
        target_columns=("order_id",),
        parameters=DeduplicateByKeysParameters(
            keys=("order_id",),
            keep=KeepPolicy.FIRST,
        ),
    )

    assert cast_request.parameters.target_type is CastTarget.NUMERIC
    assert dedup_request.parameters.keys == ("order_id",)
    with pytest.raises(ValidationError):
        RequestedTransformation(
            request_id="boolean-cast",
            operation_type=OperationType.CAST_COLUMN,
            target_columns=("flag",),
            parameters=CastColumnParameters(target_type=CastTarget.BOOLEAN),
        )
    with pytest.raises(ValidationError):
        RequestedTransformation(
            request_id="rename",
            operation_type=OperationType.RENAME_COLUMN,
            target_columns=("old",),
            parameters=CastColumnParameters(target_type=CastTarget.NUMERIC),
        )


@pytest.mark.parametrize(
    "payload",
    (
        {
            "request_id": " ",
            "operation_type": OperationType.CAST_COLUMN,
            "target_columns": ("amount",),
            "parameters": CastColumnParameters(target_type=CastTarget.NUMERIC),
        },
        {
            "request_id": "cast-blank",
            "operation_type": OperationType.CAST_COLUMN,
            "target_columns": (" ",),
            "parameters": CastColumnParameters(target_type=CastTarget.NUMERIC),
        },
        {
            "request_id": "cast-none",
            "operation_type": OperationType.CAST_COLUMN,
            "target_columns": (),
            "parameters": CastColumnParameters(target_type=CastTarget.NUMERIC),
        },
        {
            "request_id": "cast-many",
            "operation_type": OperationType.CAST_COLUMN,
            "target_columns": ("amount", "other"),
            "parameters": CastColumnParameters(target_type=CastTarget.NUMERIC),
        },
        {
            "request_id": "cast-issue",
            "operation_type": OperationType.CAST_COLUMN,
            "target_columns": ("amount",),
            "parameters": CastColumnParameters(target_type=CastTarget.NUMERIC),
            "diagnostic_issue_id": " ",
        },
        {
            "request_id": "dedup-blank",
            "operation_type": OperationType.DEDUPLICATE_BY_KEYS,
            "target_columns": ("",),
            "parameters": DeduplicateByKeysParameters(
                keys=("",), keep=KeepPolicy.FIRST
            ),
        },
        {
            "request_id": "dedup-duplicate",
            "operation_type": OperationType.DEDUPLICATE_BY_KEYS,
            "target_columns": ("id", "id"),
            "parameters": DeduplicateByKeysParameters(
                keys=("id", "id"), keep=KeepPolicy.FIRST
            ),
        },
        {
            "request_id": "dedup-mismatch",
            "operation_type": OperationType.DEDUPLICATE_BY_KEYS,
            "target_columns": ("other_id",),
            "parameters": DeduplicateByKeysParameters(
                keys=("id",), keep=KeepPolicy.FIRST
            ),
        },
        {
            "request_id": "dedup-keep",
            "operation_type": OperationType.DEDUPLICATE_BY_KEYS,
            "target_columns": ("id",),
            "parameters": {
                "kind": "DEDUPLICATE_BY_KEYS",
                "keys": ("id",),
                "keep": "MIDDLE",
            },
        },
    ),
)
def test_requested_transformation_rejects_ambiguous_identifiers_and_keys(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        RequestedTransformation(**payload)


def test_transition_collections_are_immutable_and_default_independently() -> None:
    first = TransitionResult(changed=False, screen=ScreenId.UPLOAD, code="NO_CHANGE")
    second = TransitionResult(changed=True, screen=ScreenId.DIAGNOSE, code="SOURCE_LOADED")
    finding = ApplicationFinding(
        code="SAFE_FINDING",
        blocking=False,
        safe_message="A safe application finding.",
    )

    assert first.findings == ()
    assert second.findings == ()
    assert finding.request_id is None
    with pytest.raises(ValidationError):
        first.findings = (finding,)
