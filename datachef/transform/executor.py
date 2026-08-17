"""Approval-gated, copy-based execution of allow-listed operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from datachef.contracts import (
    AcceptedReviewEvidence,
    DiagnosticReport,
    ExecutionResult,
    HumanApproval,
    HumanDecision,
    PlanValidationResult,
    PlanningContext,
    TransformationPlan,
    UserIntent,
)
from datachef.diagnostics import dataframe_fingerprint, identify_dataset
from datachef.planning.plan import expected_plan_id
from datachef.planning.authoritative import (
    context_claim_matches,
    recompute_validation_facts,
)
from datachef.transform.runner import run_allowlisted_plan


class ApprovalFailure(StrEnum):
    MISSING_APPROVAL = "MISSING_APPROVAL"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    INVALID_PLAN = "INVALID_PLAN"
    PLAN_CHANGED = "PLAN_CHANGED"
    PLAN_VERSION_CHANGED = "PLAN_VERSION_CHANGED"
    DATASET_CHANGED = "DATASET_CHANGED"
    OPERATION_SET_CHANGED = "OPERATION_SET_CHANGED"
    MISSING_ACCEPTED_REVIEW = "MISSING_ACCEPTED_REVIEW"
    REVIEW_EVIDENCE_CHANGED = "REVIEW_EVIDENCE_CHANGED"
    NONCANONICAL_PLAN = "NONCANONICAL_PLAN"
    VALIDATION_CLAIM_MISMATCH = "VALIDATION_CLAIM_MISMATCH"
    DIAGNOSTIC_CONTEXT_CHANGED = "DIAGNOSTIC_CONTEXT_CHANGED"


class ApprovalGateError(RuntimeError):
    def __init__(self, failures: tuple[ApprovalFailure, ...]) -> None:
        super().__init__("Execution refused by the typed approval gate")
        self.failures = failures


@dataclass(frozen=True)
class ExecutionBundle:
    result: ExecutionResult
    dataframe: pd.DataFrame | None


def _validate_approval(
    source: pd.DataFrame,
    diagnostic_report: DiagnosticReport,
    context: PlanningContext,
    intent: UserIntent,
    plan: TransformationPlan,
    validation: PlanValidationResult,
    accepted_review: AcceptedReviewEvidence | None,
    approval: HumanApproval | None,
    expected_review_attempt: int,
) -> None:
    failures: list[ApprovalFailure] = []
    current_identity = identify_dataset(source)
    authoritative = recompute_validation_facts(source, intent, plan)
    recomputed_validation = authoritative.validation
    if approval is None:
        raise ApprovalGateError((ApprovalFailure.MISSING_APPROVAL,))
    if accepted_review is None:
        failures.append(ApprovalFailure.MISSING_ACCEPTED_REVIEW)
    elif (
        accepted_review.dataset_id != plan.dataset_id
        or accepted_review.dataset_fingerprint != plan.dataset_fingerprint
        or accepted_review.plan_id != plan.plan_id
        or accepted_review.plan_version != plan.version
        or accepted_review.attempt != expected_review_attempt
        or accepted_review.validation_plan_id != validation.plan_id
        or not accepted_review.validation_valid
    ):
        failures.append(ApprovalFailure.REVIEW_EVIDENCE_CHANGED)
    if approval.decision is not HumanDecision.APPROVE:
        failures.append(ApprovalFailure.APPROVAL_REJECTED)
    if plan.plan_id != expected_plan_id(plan):
        failures.append(ApprovalFailure.NONCANONICAL_PLAN)
    if not recomputed_validation.valid:
        failures.append(ApprovalFailure.INVALID_PLAN)
    if validation != recomputed_validation:
        failures.append(ApprovalFailure.VALIDATION_CLAIM_MISMATCH)
    if approval.plan_id != plan.plan_id:
        failures.append(ApprovalFailure.PLAN_CHANGED)
    if approval.plan_version != plan.version:
        failures.append(ApprovalFailure.PLAN_VERSION_CHANGED)
    if (
        approval.dataset_id != plan.dataset_id
        or approval.dataset_id != context.dataset_identity.dataset_id
        or current_identity.dataset_id != context.dataset_identity.dataset_id
        or approval.dataset_fingerprint != plan.dataset_fingerprint
        or current_identity.fingerprint != plan.dataset_fingerprint
    ):
        failures.append(ApprovalFailure.DATASET_CHANGED)
    if (
        diagnostic_report != authoritative.diagnostic_report
        or not context_claim_matches(context, authoritative.planning_context)
        or context.dataset_identity.dataset_id != current_identity.dataset_id
        or context.dataset_identity.fingerprint != current_identity.fingerprint
    ):
        failures.append(ApprovalFailure.DIAGNOSTIC_CONTEXT_CHANGED)
    planned_ids = tuple(operation.operation_id for operation in plan.operations)
    if approval.approved_operation_ids != planned_ids:
        failures.append(ApprovalFailure.OPERATION_SET_CHANGED)
    if failures:
        raise ApprovalGateError(tuple(dict.fromkeys(failures)))


def execute_approved_plan(
    source: pd.DataFrame,
    diagnostic_report: DiagnosticReport,
    context: PlanningContext,
    intent: UserIntent,
    plan: TransformationPlan,
    validation: PlanValidationResult,
    accepted_review: AcceptedReviewEvidence | None,
    approval: HumanApproval | None,
    *,
    expected_review_attempt: int,
) -> ExecutionBundle:
    """Execute a whole approved plan on a fresh deep copy or return no frame."""

    _validate_approval(
        source,
        diagnostic_report,
        context,
        intent,
        plan,
        validation,
        accepted_review,
        approval,
        expected_review_attempt,
    )
    source_fingerprint = dataframe_fingerprint(source)
    operation_run = run_allowlisted_plan(source, plan)
    records = operation_run.operation_records
    if not operation_run.success or operation_run.dataframe is None:
        after_rows = records[-1].rows_after if records else len(source)
        result = ExecutionResult(
            execution_id=f"execution-{plan.plan_id}",
            dataset_id=plan.dataset_id,
            plan_id=plan.plan_id,
            plan_version=plan.version,
            accepted_review_attempt=(accepted_review.attempt if accepted_review else 1),
            success=False,
            source_fingerprint=source_fingerprint,
            before_row_count=len(source),
            after_row_count=after_rows,
            before_column_count=source.shape[1],
            after_column_count=source.shape[1],
            operation_records=records,
            error_code=operation_run.error_code,
        )
        return ExecutionBundle(result=result, dataframe=None)

    working = operation_run.dataframe

    if dataframe_fingerprint(source) != source_fingerprint:
        raise RuntimeError("immutable source fingerprint changed during execution")
    result_fingerprint = dataframe_fingerprint(working)
    result = ExecutionResult(
        execution_id=f"execution-{plan.plan_id}",
        dataset_id=plan.dataset_id,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        accepted_review_attempt=(accepted_review.attempt if accepted_review else 1),
        success=True,
        source_fingerprint=source_fingerprint,
        result_fingerprint=result_fingerprint,
        before_row_count=len(source),
        after_row_count=len(working),
        before_column_count=source.shape[1],
        after_column_count=working.shape[1],
        operation_records=records,
    )
    return ExecutionBundle(result=result, dataframe=working)
