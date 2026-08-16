"""Framework-independent state transitions for the Phase 1A workflow."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datachef.contracts import (
    HumanApproval,
    HumanDecision,
    PlanningContext,
    QAStatus,
    QualityInvariant,
    ReviewerDecision,
    UserIntent,
    WorkflowStage,
    WorkflowState,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.intent import discover_questions
from datachef.planning import (
    Planner,
    ReviewEvidenceError,
    Reviewer,
    accept_review,
    validate_plan,
)
from datachef.privacy import ColumnAliasMap, build_column_alias_map, build_planning_context
from datachef.qa import run_quality_assurance
from datachef.transform.executor import ApprovalGateError, execute_approved_plan


MAX_PLANNING_ATTEMPTS = 3


@dataclass(frozen=True)
class WorkflowRuntime:
    state: WorkflowState
    raw_dataframe: pd.DataFrame
    transformed_dataframe: pd.DataFrame | None = None
    gold_dataframe: pd.DataFrame | None = None
    user_intent: UserIntent | None = None
    column_alias_map: ColumnAliasMap | None = None


def _replace_state(state: WorkflowState, **updates: object) -> WorkflowState:
    payload = state.model_dump()
    payload.update(updates)
    return WorkflowState.model_validate(payload)


def prepare_workflow(
    source: pd.DataFrame,
    intent: UserIntent,
    planner: Planner,
    reviewer: Reviewer,
) -> WorkflowRuntime:
    """Prepare one bounded plan and stop before human-controlled execution."""

    raw = source.copy(deep=True)
    state = WorkflowState()
    report = diagnose_raw_dataframe(
        raw,
        selected_key_columns=intent.selected_key_columns,
    )
    state = _replace_state(
        state,
        stage=WorkflowStage.DIAGNOSED,
        dataset_identity=report.dataset_identity,
        diagnostic_report=report,
    )
    questions = discover_questions(report) if not intent.user_goal.strip() else ()
    column_alias_map = build_column_alias_map(report, intent)
    initial_context = build_planning_context(
        report,
        intent,
        questions,
        column_alias_map=column_alias_map,
    )
    state = _replace_state(
        state,
        stage=WorkflowStage.INTENT_CAPTURED,
        user_intent=initial_context.user_intent,
        suggested_questions=questions,
    )
    feedback: tuple[str, ...] = ()
    reviews = []
    for attempt in range(1, MAX_PLANNING_ATTEMPTS + 1):
        context = build_planning_context(
            report,
            intent,
            questions,
            previous_review_feedback=feedback,
            provider_context_reference=initial_context.provider_context_reference,
            column_alias_map=column_alias_map,
        )
        state = _replace_state(
            state,
            stage=WorkflowStage.CONTEXT_READY,
            planning_context=context,
            user_intent=context.user_intent,
            planning_attempts=attempt,
            accepted_review=None,
            human_approval=None,
        )
        plan = planner.propose(context, attempt=attempt)
        validation = validate_plan(context, plan)
        state = _replace_state(
            state,
            stage=WorkflowStage.PLANNING,
            transformation_plan=plan,
            plan_validation=validation,
        )
        if not validation.valid:
            feedback = tuple(
                f"{finding.code}: {finding.message}"
                for finding in validation.findings
            )
            if attempt == MAX_PLANNING_ATTEMPTS:
                state = _replace_state(
                    state,
                    stage=WorkflowStage.PLAN_REJECTED,
                    last_error_code="PLAN_VALIDATION_ATTEMPTS_EXHAUSTED",
                )
                return WorkflowRuntime(
                    state=state,
                    raw_dataframe=raw,
                    user_intent=intent,
                    column_alias_map=column_alias_map,
                )
            continue

        verdict = reviewer.review(
            context,
            plan,
            validation,
            previous_feedback=feedback,
            attempt=attempt,
        )
        reviews.append(verdict)
        state = _replace_state(state, review_history=tuple(reviews))
        if verdict.decision is ReviewerDecision.ACCEPT:
            try:
                accepted_review = accept_review(
                    plan,
                    validation,
                    verdict,
                    attempt=attempt,
                )
            except ReviewEvidenceError:
                feedback = ("REVIEW_EVIDENCE_MISMATCH",)
                if attempt == MAX_PLANNING_ATTEMPTS:
                    state = _replace_state(
                        state,
                        stage=WorkflowStage.PLAN_REJECTED,
                        last_error_code="REVIEW_EVIDENCE_ATTEMPTS_EXHAUSTED",
                    )
                    return WorkflowRuntime(
                        state=state,
                        raw_dataframe=raw,
                        user_intent=intent,
                        column_alias_map=column_alias_map,
                    )
                continue
            state = _replace_state(
                state,
                stage=WorkflowStage.AWAITING_APPROVAL,
                accepted_review=accepted_review,
                last_error_code=None,
            )
            return WorkflowRuntime(
                state=state,
                raw_dataframe=raw,
                user_intent=intent,
                column_alias_map=column_alias_map,
            )
        if verdict.decision is ReviewerDecision.REJECT:
            state = _replace_state(
                state,
                stage=WorkflowStage.PLAN_REJECTED,
                last_error_code="PLAN_REJECTED_BY_REVIEWER",
            )
            return WorkflowRuntime(
                state=state,
                raw_dataframe=raw,
                user_intent=intent,
                column_alias_map=column_alias_map,
            )
        feedback = verdict.feedback or verdict.findings
    state = _replace_state(
        state,
        stage=WorkflowStage.PLAN_REJECTED,
        last_error_code="PLANNING_ATTEMPTS_EXHAUSTED",
    )
    return WorkflowRuntime(
        state=state,
        raw_dataframe=raw,
        user_intent=intent,
        column_alias_map=column_alias_map,
    )


def execute_workflow(
    runtime: WorkflowRuntime,
    approval: HumanApproval | None,
    *,
    user_invariants: tuple[QualityInvariant, ...] = (),
) -> WorkflowRuntime:
    """Supported application path for approved execution, QA, and gold gating."""

    state = runtime.state
    if state.stage is not WorkflowStage.AWAITING_APPROVAL:
        return runtime
    if approval is None:
        return runtime
    if approval.decision is HumanDecision.REJECT:
        return WorkflowRuntime(
            state=_replace_state(
                state,
                stage=WorkflowStage.PLAN_REJECTED,
                accepted_review=None,
                human_approval=approval,
                last_error_code="HUMAN_REJECTED_PLAN",
            ),
            raw_dataframe=runtime.raw_dataframe,
            user_intent=runtime.user_intent,
            column_alias_map=runtime.column_alias_map,
        )
    assert state.planning_context is not None
    assert state.transformation_plan is not None
    assert state.plan_validation is not None
    assert state.accepted_review is not None
    try:
        state.accepted_review.require_matching_final_verdict(
            state.review_history,
            current_plan_id=state.transformation_plan.plan_id,
            current_attempt=state.planning_attempts,
        )
    except ValueError:
        return runtime
    planned_ids = tuple(
        operation.operation_id for operation in state.transformation_plan.operations
    )
    if (
        approval.dataset_id != state.transformation_plan.dataset_id
        or approval.dataset_fingerprint != state.transformation_plan.dataset_fingerprint
        or approval.plan_id != state.transformation_plan.plan_id
        or approval.plan_version != state.transformation_plan.version
        or approval.approved_operation_ids != planned_ids
    ):
        return runtime
    state = _replace_state(
        state,
        stage=WorkflowStage.EXECUTING,
        human_approval=approval,
        last_error_code=None,
    )
    try:
        bundle = execute_approved_plan(
            runtime.raw_dataframe,
            state.diagnostic_report,
            state.planning_context,
            runtime.user_intent,
            state.transformation_plan,
            state.plan_validation,
            state.accepted_review,
            approval,
            expected_review_attempt=state.planning_attempts,
        )
    except ApprovalGateError as error:
        del error
        return runtime
    if not bundle.result.success or bundle.dataframe is None:
        return WorkflowRuntime(
            state=_replace_state(
                state,
                stage=WorkflowStage.EXECUTION_FAILED,
                execution_result=bundle.result,
                last_error_code=bundle.result.error_code,
            ),
            raw_dataframe=runtime.raw_dataframe,
            user_intent=runtime.user_intent,
            column_alias_map=runtime.column_alias_map,
        )
    assert state.diagnostic_report is not None
    assert runtime.user_intent is not None
    qa_report = run_quality_assurance(
        runtime.raw_dataframe,
        bundle.dataframe,
        bundle.result,
        state.diagnostic_report,
        state.planning_context,
        runtime.user_intent,
        state.transformation_plan,
        state.plan_validation,
        state.accepted_review,
        approval,
        user_invariants=user_invariants,
    )
    if qa_report.status is QAStatus.PASS:
        stage = WorkflowStage.QA_PASSED
        gold = bundle.dataframe.copy(deep=True)
    elif qa_report.status is QAStatus.WARN:
        stage = WorkflowStage.QA_WARNING
        gold = None
    else:
        stage = WorkflowStage.QA_FAILED
        gold = None
    final_state = _replace_state(
        state,
        stage=stage,
        execution_result=bundle.result,
        qa_report=qa_report,
    )
    return WorkflowRuntime(
        state=final_state,
        raw_dataframe=runtime.raw_dataframe,
        transformed_dataframe=bundle.dataframe,
        gold_dataframe=gold,
        user_intent=runtime.user_intent,
        column_alias_map=runtime.column_alias_map,
    )
