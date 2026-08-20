"""Pure allow-listed operation runner shared by execution and QA replay."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datachef.contracts import (
    DeduplicateByKeysParameters,
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


def run_allowlisted_plan(
    source: pd.DataFrame,
    plan: TransformationPlan,
) -> OperationRun:
    """Run a declarative plan on a fresh copy without trusting prior evidence."""

    working = source.copy(deep=True)
    records: list[OperationExecutionRecord] = []
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
            if isinstance(operation.parameters, DeduplicateByKeysParameters):
                key_frame = working.loc[:, list(operation.parameters.keys)]
                if bool(key_frame.isna().any(axis=1).any()):
                    raise ValueError("null keys are unsafe for key deduplication")
            effect = definition.handler(working, operation)
            if not isinstance(effect.dataframe, pd.DataFrame):
                raise TypeError("operation handler did not return a DataFrame")
            working = effect.dataframe
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
    )


__all__ = ["OperationRun", "run_allowlisted_plan"]
