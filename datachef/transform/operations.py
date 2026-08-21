"""Authoritative operation catalogue and deterministic Pandas handlers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

import pandas as pd

from datachef.contracts import (
    CastColumnParameters,
    ComputeColumnParameters,
    ComputeOperator,
    DropColumnParameters,
    ImputeMissingParameters,
    ImputeStrategy,
    NormalizeNumericTextParameters,
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
    # Measured by handlers that rewrite existing values, so QA can verify the
    # rewrite rather than trust it. See _impute_missing.
    changed_non_null_count: int | None = None
    filled_null_count: int | None = None


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



# A closed set, enumerated. Currency is a named noise class, not a pattern the
# caller supplies, so the operation stays inside the allow-list.
_CURRENCY_SYMBOLS = frozenset("$€£¥₹₩₽")
_THOUSANDS_SEPARATORS = frozenset(",_")
_ACCOUNTING_NUMBER = re.compile(r"\+?(?:\d+(?:\.\d*)?|\.\d+)")


def _strip_numeric_noise(
    value: object,
    parameters: NormalizeNumericTextParameters,
) -> object:
    """Remove named noise classes. Never returns null for a non-null input."""

    if not isinstance(value, str):
        return value
    text = value
    if parameters.strip_whitespace:
        text = text.strip()
    accounting_negative = text.startswith("(") and text.endswith(")")
    if accounting_negative:
        text = text[1:-1]
        if parameters.strip_whitespace:
            text = text.strip()
    if parameters.strip_currency_symbols:
        text = "".join(
            character for character in text if character not in _CURRENCY_SYMBOLS
        )
    if parameters.strip_thousands_separators:
        text = "".join(
            character for character in text if character not in _THOUSANDS_SEPARATORS
        )
    if parameters.strip_whitespace:
        text = text.strip()
    if accounting_negative:
        if _ACCOUNTING_NUMBER.fullmatch(text):
            text = "-" + text.removeprefix("+")
        else:
            # Preserve an invalid accounting token as invalid text. The next
            # cast therefore fails its value-preservation gate instead of
            # inventing a number (especially never zero).
            text = f"({text})"
    return text


def _normalize_numeric_text(
    dataframe: pd.DataFrame,
    operation: TransformationOperation,
) -> OperationEffect:
    parameters = operation.parameters
    assert isinstance(parameters, NormalizeNumericTextParameters)
    changed = 0
    introduced = 0
    for column in operation.target_columns:
        before = dataframe[column].copy(deep=True)
        after = before.map(lambda value: _strip_numeric_noise(value, parameters))
        dataframe[column] = after
        changed += _changed_cells(before, after)
        introduced += int((before.notna() & after.isna()).sum())
    return OperationEffect(
        dataframe=dataframe,
        affected_cell_count=changed,
        introduced_null_count=introduced,
    )


def _drop_column(
    dataframe: pd.DataFrame,
    operation: TransformationOperation,
) -> OperationEffect:
    parameters = operation.parameters
    assert isinstance(parameters, DropColumnParameters)
    columns = list(operation.target_columns)
    # Raises KeyError on an absent column, which the runner turns into a typed
    # failure. Dropping something that is not there is never treated as success.
    result = dataframe.drop(columns=columns)
    return OperationEffect(
        dataframe=result,
        affected_cell_count=int(len(dataframe) * len(columns)),
    )


def _impute_fill_value(
    series: pd.Series,
    parameters: ImputeMissingParameters,
) -> object:
    if parameters.strategy is ImputeStrategy.CONSTANT:
        if parameters.constant_value is None:
            raise ValueError("constant imputation requires a constant value")
        return parameters.constant_value
    if parameters.strategy is ImputeStrategy.MODE:
        modes = series.mode(dropna=True)
        if modes.empty:
            raise ValueError("no mode exists for this column")
        return modes.iloc[0]
    if parameters.strategy is ImputeStrategy.MEAN:
        return series.mean()
    if parameters.strategy is ImputeStrategy.MEDIAN:
        return series.median()
    raise ValueError("unsupported imputation strategy")


def _impute_missing(
    dataframe: pd.DataFrame,
    operation: TransformationOperation,
) -> OperationEffect:
    parameters = operation.parameters
    assert isinstance(parameters, ImputeMissingParameters)
    column = operation.target_columns[0]
    before = dataframe[column].copy(deep=True)
    missing_before = before.isna()
    if not bool(missing_before.any()):
        # Nothing to fill. Refused rather than silently applied so the operation
        # can never report an effect it did not have.
        raise ValueError("imputation requires at least one missing value")
    after = before.fillna(_impute_fill_value(before, parameters))
    dataframe[column] = after
    equal = before.eq(after) | (before.isna() & after.isna())
    rewritten = ~equal.fillna(False)
    changed_non_null = int((rewritten & before.notna()).sum())
    filled = int((missing_before & after.notna()).sum())
    return OperationEffect(
        dataframe=dataframe,
        affected_cell_count=filled,
        changed_non_null_count=changed_non_null,
        filled_null_count=filled,
    )


def _compute_column(
    dataframe: pd.DataFrame,
    operation: TransformationOperation,
) -> OperationEffect:
    parameters = operation.parameters
    assert isinstance(parameters, ComputeColumnParameters)
    left = dataframe[parameters.left_column]
    right = dataframe[parameters.right_column]
    if parameters.operator is ComputeOperator.DIVIDE:
        if bool(right.eq(0).fillna(False).any()):
            raise ValueError("division by zero is not allowed")
        result = left / right
    elif parameters.operator is ComputeOperator.ADD:
        result = left + right
    elif parameters.operator is ComputeOperator.SUBTRACT:
        result = left - right
    elif parameters.operator is ComputeOperator.MULTIPLY:
        result = left * right
    else:  # pragma: no cover - closed enum
        raise ValueError("unsupported compute operator")
    dataframe[parameters.output_column] = result
    return OperationEffect(
        dataframe=dataframe,
        affected_cell_count=int(len(dataframe)),
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
    OperationType.DROP_COLUMN: OperationDefinition(
        OperationType.DROP_COLUMN,
        DropColumnParameters,
        _drop_column,
        material=True,
        may_drop_rows=False,
    ),
    OperationType.IMPUTE_MISSING: OperationDefinition(
        OperationType.IMPUTE_MISSING,
        ImputeMissingParameters,
        _impute_missing,
        material=True,
        may_drop_rows=False,
    ),
    OperationType.NORMALIZE_NUMERIC_TEXT: OperationDefinition(
        OperationType.NORMALIZE_NUMERIC_TEXT,
        NormalizeNumericTextParameters,
        _normalize_numeric_text,
        material=True,
        may_drop_rows=False,
    ),
    OperationType.COMPUTE_COLUMN: OperationDefinition(
        OperationType.COMPUTE_COLUMN,
        ComputeColumnParameters,
        _compute_column,
        material=True,
        may_drop_rows=False,
    ),
}


assert set(OPERATION_CATALOGUE) == set(OperationType)
assert all(definition.handler for definition in OPERATION_CATALOGUE.values())
