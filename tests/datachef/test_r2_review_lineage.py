from __future__ import annotations

import pandas as pd

from datachef.contracts import (
    DeduplicateByKeysParameters,
    DiagnosticIssueKind,
    DiagnosticResolution,
    KeepPolicy,
    OperationType,
    RenameColumnParameters,
    ReviewerDecision,
    ReviewerVerdict,
    RiskLevel,
    TransformationOperation,
    UserIntent,
    WorkflowStage,
    QAStatus,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.planning import (
    RuleBasedPlanner,
    SequencePlanner,
    SequenceReviewer,
    accept_review,
    create_transformation_plan,
    validate_plan,
)
from datachef.privacy import build_planning_context
from datachef.qa import run_quality_assurance
from datachef.transform.executor import execute_approved_plan
from datachef.workflow import prepare_workflow
from support import accepted_review, human_approval


def _rename(source: str, target: str, operation_id: str = "rename"):
    return TransformationOperation(
        operation_id=operation_id,
        operation_type=OperationType.RENAME_COLUMN,
        target_columns=(source,),
        parameters=RenameColumnParameters(new_name=target),
        user_requirement_ids=(operation_id,),
        rationale="Follow deterministic column lineage.",
        expected_effect="Change one column label.",
        risk=RiskLevel.MEDIUM,
        requires_human_approval=True,
    )


def _dedup(key: str):
    return TransformationOperation(
        operation_id="dedup",
        operation_type=OperationType.DEDUPLICATE_BY_KEYS,
        target_columns=(key,),
        parameters=DeduplicateByKeysParameters(keys=(key,), keep=KeepPolicy.FIRST),
        user_requirement_ids=("key",),
        rationale="Enforce key uniqueness.",
        expected_effect="Keep the first non-null key row.",
        risk=RiskLevel.HIGH,
        requires_human_approval=True,
    )


def _context(source: pd.DataFrame, intent: UserIntent):
    report = diagnose_raw_dataframe(
        source,
        selected_key_columns=intent.selected_key_columns,
    )
    return report, build_planning_context(report, intent, ())


def test_plan_version_and_review_attempt_are_independent() -> None:
    source = pd.DataFrame({"label": [" A "]})
    intent = UserIntent(intent_id="version-attempt")
    _, context = _context(source, intent)
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=2,
        operations=(),
        summary="Plan version two reviewed on the first attempt.",
    )
    validation = validate_plan(context, plan)
    evidence = accept_review(
        plan,
        validation,
        ReviewerVerdict(
            plan_id=plan.plan_id,
            attempt=1,
            decision=ReviewerDecision.ACCEPT,
        ),
        attempt=1,
    )

    assert evidence.plan_version == 2
    assert evidence.attempt == 1


def test_third_review_attempt_can_accept_plan_version_one() -> None:
    source = pd.DataFrame({"label": [" A "]})
    intent = UserIntent(intent_id="third-attempt")
    _, context = _context(source, intent)
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(),
        summary="Stable plan reviewed repeatedly.",
    )
    runtime = prepare_workflow(
        source,
        intent,
        SequencePlanner((plan, plan, plan)),
        SequenceReviewer(
            (
                ReviewerDecision.REVISE,
                ReviewerDecision.REVISE,
                ReviewerDecision.ACCEPT,
            )
        ),
    )

    assert runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
    assert runtime.state.planning_attempts == 3
    assert runtime.state.accepted_review is not None
    assert runtime.state.accepted_review.plan_version == 1
    assert runtime.state.accepted_review.attempt == 3


def test_offline_planner_does_not_derive_plan_version_from_review_attempt() -> None:
    source = pd.DataFrame({"label": ["A"]})
    intent = UserIntent(intent_id="fallback-version")
    _, context = _context(source, intent)

    plan = RuleBasedPlanner().propose(context, attempt=3)

    assert plan.version == 1


def test_renamed_null_key_retains_validation_evidence() -> None:
    source = pd.DataFrame({"key": [1, 1, None, 2], "value": ["a", "b", "c", "d"]})
    intent = UserIntent(
        intent_id="renamed-null-key",
        selected_key_columns=("key",),
        acceptable_row_loss_pct=100,
    )
    _, context = _context(source, intent)
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(_rename("key", "new_key"), _dedup("new_key")),
        summary="Rename then deduplicate a nullable key.",
    )

    validation = validate_plan(context, plan)

    assert validation.valid is False
    assert "NULL_KEYS_UNSAFE" in {finding.code for finding in validation.findings}
    assert validation.row_loss_estimates[-1].estimated_rows == 1


def _run_qa(source: pd.DataFrame, intent: UserIntent, operations):
    report, context = _context(source, intent)
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=tuple(operations),
        summary="R2 lineage QA plan.",
    )
    validation = validate_plan(context, plan)
    review = accepted_review(plan, validation)
    approval = human_approval(plan)
    bundle = execute_approved_plan(
        source,
        report,
        context,
        intent,
        plan,
        validation,
        review,
        approval,
        expected_review_attempt=review.attempt,
    )
    assert bundle.dataframe is not None
    qa = run_quality_assurance(
        source,
        bundle.dataframe,
        bundle.result,
        report,
        context,
        intent,
        plan,
        validation,
        review,
        approval,
    )
    issue = next(
        item for item in report.issues if item.kind is DiagnosticIssueKind.DUPLICATE_KEYS
    )
    comparison = next(
        item for item in qa.diagnostic_comparisons if item.issue_id == issue.issue_id
    )
    return validation, qa, comparison


def test_duplicate_key_comparison_follows_chained_rename_when_unchanged() -> None:
    source = pd.DataFrame({"key": [1, 1, 2], "value": ["a", "b", "c"]})
    intent = UserIntent(
        intent_id="unchanged-lineage",
        selected_key_columns=("key",),
        acceptable_row_loss_pct=100,
    )

    _, _, comparison = _run_qa(
        source,
        intent,
        (_rename("key", "middle", "rename-one"), _rename("middle", "final", "rename-two")),
    )

    assert comparison.status is DiagnosticResolution.UNCHANGED
    assert comparison.before_value == comparison.after_value == 1


def test_rename_then_dedup_resolves_duplicate_key_evidence() -> None:
    source = pd.DataFrame({"key": [1, 1, 2], "value": ["a", "b", "c"]})
    intent = UserIntent(
        intent_id="resolved-lineage",
        selected_key_columns=("key",),
        acceptable_row_loss_pct=40,
    )

    validation, qa, comparison = _run_qa(
        source,
        intent,
        (_rename("key", "new_key"), _dedup("new_key")),
    )

    assert validation.row_loss_estimates[-1].estimated_rows == 1
    assert comparison.status is DiagnosticResolution.RESOLVED
    assert comparison.after_value == 0
    assert qa.status is QAStatus.PASS


def test_composite_key_estimate_follows_one_renamed_component() -> None:
    source = pd.DataFrame(
        {"left": [1, 1, 2], "right": ["A", "A", "B"], "value": [1, 2, 3]}
    )
    intent = UserIntent(
        intent_id="composite-lineage",
        selected_key_columns=("left", "right"),
        acceptable_row_loss_pct=40,
    )
    _, context = _context(source, intent)
    dedup = TransformationOperation(
        operation_id="dedup-composite",
        operation_type=OperationType.DEDUPLICATE_BY_KEYS,
        target_columns=("renamed_left", "right"),
        parameters=DeduplicateByKeysParameters(
            keys=("renamed_left", "right"),
            keep=KeepPolicy.FIRST,
        ),
        user_requirement_ids=("composite",),
        rationale="Deduplicate the approved composite key.",
        expected_effect="Remove one duplicate composite key.",
        risk=RiskLevel.HIGH,
        requires_human_approval=True,
    )
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(_rename("left", "renamed_left"), dedup),
        summary="Composite-key lineage.",
    )

    validation = validate_plan(context, plan)

    assert validation.valid is True
    assert validation.row_loss_estimates[-1].estimated_rows == 1
