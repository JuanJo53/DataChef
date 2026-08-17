"""Authoritative operation catalogue and deterministic Pandas handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from datachef.contracts import (
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    DeduplicateByKeysParameters,
    DropDuplicateRowsParameters,
    KeepPolicy,
    NormalizeMissingTokensParameters,
    OperationParameters,
    OperationType,
    RenameColumnParameters,
    TransformationOperation,
    TrimWhitespaceParameters,
)


@dataclass(frozen=True)
class OperationEffect:
    dataframe: pd.DataFrame
    affected_cell_count: int
    introduced_null_count: int | None = None


OperationHandler = Callable[[pd.DataFrame, TransformationOperation], OperationEffect]


@dataclass(frozen=True)
class OperationDefinition:
    operation_type: OperationType
    parameter_type: type
    handler: OperationHandler
    material: bool
    may_drop_rows: bool


def _changed_cells(before: pd.Series, after: pd.Series) -> int:
    equal = before.eq(after) | (before.isna() & after.isna())
    return int((~equal.fillna(False)).sum())


def _trim_whitespace(
    dataframe: pd.DataFrame,
    operation: TransformationOperation,
) -> OperationEffect:
    assert isinstance(operation.parameters, TrimWhitespaceParameters)
    changed = 0
    for column in operation.target_columns:
        before = dataframe[column].copy(deep=True)
        after = before.map(lambda value: value.strip() if isinstance(value, str) else value)
        dataframe[column] = after
        changed += _changed_cells(before, after)
    return OperationEffect(dataframe=dataframe, affected_cell_count=changed)


def _normalize_missing_tokens(
    dataframe: pd.DataFrame,
    operation: TransformationOperation,
) -> OperationEffect:
    parameters = operation.parameters
    assert isinstance(parameters, NormalizeMissingTokensParameters)
    tokens = set(parameters.tokens)
    lowered = {token.casefold() for token in tokens}
    changed = 0

    def normalize(value: object) -> object:
        if not isinstance(value, str):
            return value
        matches = value in tokens if parameters.case_sensitive else value.casefold() in lowered
        return pd.NA if matches else value

    for column in operation.target_columns:
        before = dataframe[column].copy(deep=True)
        after = before.map(normalize)
        dataframe[column] = after
        changed += _changed_cells(before, after)
    return OperationEffect(dataframe=dataframe, affected_cell_count=changed)


def _cast_boolean(
    series: pd.Series,
    parameters: CastColumnParameters,
) -> pd.Series:
    true_values = {value.casefold() for value in parameters.true_values}
    false_values = {value.casefold() for value in parameters.false_values}

    def convert(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().casefold()
        if normalized in true_values:
            return True
        if normalized in false_values:
            return False
        if parameters.errors is CastErrorPolicy.COERCE:
            return pd.NA
        raise ValueError("value is not in the approved Boolean token catalogue")

    return series.map(convert).astype("boolean")


def _cast_column(
    dataframe: pd.DataFrame,
    operation: TransformationOperation,
) -> OperationEffect:
    parameters = operation.parameters
    assert isinstance(parameters, CastColumnParameters)
    column = operation.target_columns[0]
    before = dataframe[column].copy(deep=True)
    errors = parameters.errors.value.lower()
    if parameters.target_type is CastTarget.STRING:
        after = before.astype("string")
    elif parameters.target_type is CastTarget.NUMERIC:
        after = pd.to_numeric(before, errors=errors)
    elif parameters.target_type is CastTarget.BOOLEAN:
        after = _cast_boolean(before, parameters)
    elif parameters.target_type is CastTarget.DATETIME:
        after = pd.to_datetime(
            before,
            format=parameters.datetime_format,
            errors=errors,
            utc=parameters.utc,
        )
    else:  # pragma: no cover - the enum makes this unreachable
        raise ValueError("unsupported cast target")
    dataframe[column] = after
    introduced_null_count = int((before.notna() & after.isna()).sum())
    return OperationEffect(
        dataframe=dataframe,
        affected_cell_count=max(_changed_cells(before, after), int(before.notna().sum())),
        introduced_null_count=introduced_null_count,
    )


def _rename_column(
    dataframe: pd.DataFrame,
    operation: TransformationOperation,
) -> OperationEffect:
    parameters = operation.parameters
    assert isinstance(parameters, RenameColumnParameters)
    source = operation.target_columns[0]
    dataframe.rename(columns={source: parameters.new_name}, inplace=True)
    return OperationEffect(dataframe=dataframe, affected_cell_count=int(len(dataframe)))


def _keep_value(policy: KeepPolicy) -> str:
    return "first" if policy is KeepPolicy.FIRST else "last"


def _drop_duplicate_rows(
    dataframe: pd.DataFrame,
    operation: TransformationOperation,
) -> OperationEffect:
    parameters = operation.parameters
    assert isinstance(parameters, DropDuplicateRowsParameters)
    before_rows = len(dataframe)
    result = dataframe.drop_duplicates(keep=_keep_value(parameters.keep))
    removed = before_rows - len(result)
    return OperationEffect(
        dataframe=result,
        affected_cell_count=int(removed * dataframe.shape[1]),
    )


def _deduplicate_by_keys(
    dataframe: pd.DataFrame,
    operation: TransformationOperation,
) -> OperationEffect:
    parameters = operation.parameters
    assert isinstance(parameters, DeduplicateByKeysParameters)
    before_rows = len(dataframe)
    result = dataframe.drop_duplicates(
        subset=list(parameters.keys),
        keep=_keep_value(parameters.keep),
    )
    removed = before_rows - len(result)
    return OperationEffect(
        dataframe=result,
        affected_cell_count=int(removed * dataframe.shape[1]),
    )


OPERATION_CATALOGUE: dict[OperationType, OperationDefinition] = {
    OperationType.TRIM_WHITESPACE: OperationDefinition(
        OperationType.TRIM_WHITESPACE,
        TrimWhitespaceParameters,
        _trim_whitespace,
        material=False,
        may_drop_rows=False,
    ),
    OperationType.NORMALIZE_MISSING_TOKENS: OperationDefinition(
        OperationType.NORMALIZE_MISSING_TOKENS,
        NormalizeMissingTokensParameters,
        _normalize_missing_tokens,
        material=True,
        may_drop_rows=False,
    ),
    OperationType.CAST_COLUMN: OperationDefinition(
        OperationType.CAST_COLUMN,
        CastColumnParameters,
        _cast_column,
        material=True,
        may_drop_rows=False,
    ),
    OperationType.RENAME_COLUMN: OperationDefinition(
        OperationType.RENAME_COLUMN,
        RenameColumnParameters,
        _rename_column,
        material=True,
        may_drop_rows=False,
    ),
    OperationType.DROP_DUPLICATE_ROWS: OperationDefinition(
        OperationType.DROP_DUPLICATE_ROWS,
        DropDuplicateRowsParameters,
        _drop_duplicate_rows,
        material=True,
        may_drop_rows=True,
    ),
    OperationType.DEDUPLICATE_BY_KEYS: OperationDefinition(
        OperationType.DEDUPLICATE_BY_KEYS,
        DeduplicateByKeysParameters,
        _deduplicate_by_keys,
        material=True,
        may_drop_rows=True,
    ),
}


assert set(OPERATION_CATALOGUE) == set(OperationType)
assert all(definition.handler for definition in OPERATION_CATALOGUE.values())
