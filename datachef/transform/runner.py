"""Pure allow-listed operation runner shared by execution and QA replay."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from datachef.contracts import (
    DeduplicateByKeysParameters,
    ComputeColumnParameters,
    ComputeOperator,
    OperationExecutionRecord,
    OperationExecutionStatus,
    TransformationPlan,
)
from datachef.transform.operations import OPERATION_CATALOGUE


@dataclass(frozen=True)
class OperationRun:
    success: bool
    dataframe: pd.DataFrame | None
    operation_records: tuple[OperationExecutionRecord, ...]
    error_code: str | None = None
    computation_evidence: tuple["ComputationReplayEvidence", ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class ComputationReplayEvidence:
    operation_id: str
    operator: ComputeOperator
    output_column: str
    before_columns: tuple[str, ...]
    after_columns: tuple[str, ...]
    rows_before: int
    rows_after: int
    existing_columns_preserved: bool
    left_values: pd.Series = field(repr=False, compare=False)
    right_values: pd.Series = field(repr=False, compare=False)
    output_values: pd.Series = field(repr=False, compare=False)


def run_allowlisted_plan(
    source: pd.DataFrame,
    plan: TransformationPlan,
) -> OperationRun:
    """Run a declarative plan on a fresh copy without trusting prior evidence."""

    working = source.copy(deep=True)
    records: list[OperationExecutionRecord] = []
    computation_evidence: list[ComputationReplayEvidence] = []
    for operation in plan.operations:
        rows_before = len(working)
        definition = OPERATION_CATALOGUE.get(operation.operation_type)
        if definition is None:
            return OperationRun(
                success=False,
                dataframe=None,
                operation_records=tuple(records),
                error_code="OPERATION_NOT_REGISTERED",
            )
        try:
            compute_before = None
            if isinstance(operation.parameters, ComputeColumnParameters):
                compute_before = working.copy(deep=True)
            if isinstance(operation.parameters, DeduplicateByKeysParameters):
                key_frame = working.loc[:, list(operation.parameters.keys)]
                if bool(key_frame.isna().any(axis=1).any()):
                    raise ValueError("null keys are unsafe for key deduplication")
            effect = definition.handler(working, operation)
            if not isinstance(effect.dataframe, pd.DataFrame):
                raise TypeError("operation handler did not return a DataFrame")
            working = effect.dataframe
            if compute_before is not None:
                parameters = operation.parameters
                assert isinstance(parameters, ComputeColumnParameters)
                before_columns = tuple(str(column) for column in compute_before.columns)
                after_columns = tuple(str(column) for column in working.columns)
                preserved = (
                    tuple(column for column in after_columns if column != parameters.output_column)
                    == before_columns
                    and all(
                        compute_before[column].equals(working[column])
                        for column in before_columns
                    )
                )
                computation_evidence.append(
                    ComputationReplayEvidence(
                        operation_id=operation.operation_id,
                        operator=parameters.operator,
                        output_column=parameters.output_column,
                        before_columns=before_columns,
                        after_columns=after_columns,
                        rows_before=len(compute_before),
                        rows_after=len(working),
                        existing_columns_preserved=preserved,
                        left_values=compute_before[parameters.left_column].copy(deep=True),
                        right_values=compute_before[parameters.right_column].copy(deep=True),
                        output_values=working[parameters.output_column].copy(deep=True),
                    )
                )
            records.append(
                OperationExecutionRecord(
                    operation_id=operation.operation_id,
                    status=OperationExecutionStatus.APPLIED,
                    rows_before=rows_before,
                    rows_after=len(working),
                    affected_cell_count=effect.affected_cell_count,
                    introduced_null_count=effect.introduced_null_count,
                    changed_non_null_count=effect.changed_non_null_count,
                    filled_null_count=effect.filled_null_count,
                )
            )
        except Exception:
            records.append(
                OperationExecutionRecord(
                    operation_id=operation.operation_id,
                    status=OperationExecutionStatus.FAILED,
                    rows_before=rows_before,
                    rows_after=len(working),
                    affected_cell_count=0,
                    error_code="OPERATION_RUNTIME_ERROR",
                )
            )
            return OperationRun(
                success=False,
                dataframe=None,
                operation_records=tuple(records),
                error_code="PLAN_EXECUTION_ABORTED",
            )
    return OperationRun(
        success=True,
        dataframe=working,
        operation_records=tuple(records),
        computation_evidence=tuple(computation_evidence),
    )


__all__ = ["ComputationReplayEvidence", "OperationRun", "run_allowlisted_plan"]
