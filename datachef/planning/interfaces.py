"""Planner/reviewer protocols and deterministic offline implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from datachef.contracts import (
    NormalizeNumericTextParameters,
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    DeduplicateByKeysParameters,
    DiagnosticIssueKind,
    KeepPolicy,
    OperationType,
    PlanValidationResult,
    PlanningContext,
    ReviewerDecision,
    ReviewerVerdict,
    RiskLevel,
    TransformationOperation,
    TransformationPlan,
)
from datachef.planning.plan import create_transformation_plan


class Planner(Protocol):
    def propose(
        self,
        context: PlanningContext,
        *,
        attempt: int,
    ) -> TransformationPlan: ...


class Reviewer(Protocol):
    def review(
        self,
        context: PlanningContext,
        plan: TransformationPlan,
        validation: PlanValidationResult,
        *,
        previous_feedback: tuple[str, ...],
        attempt: int,
    ) -> ReviewerVerdict: ...


@dataclass
class SequencePlanner:
    plans: tuple[TransformationPlan, ...]
    calls: int = 0

    def propose(self, context: PlanningContext, *, attempt: int) -> TransformationPlan:
        del context, attempt
        index = min(self.calls, len(self.plans) - 1)
        self.calls += 1
        return self.plans[index]


@dataclass
class SequenceReviewer:
    decisions: tuple[ReviewerDecision, ...]
    calls: int = 0

    def review(
        self,
        context: PlanningContext,
        plan: TransformationPlan,
        validation: PlanValidationResult,
        *,
        previous_feedback: tuple[str, ...],
        attempt: int,
    ) -> ReviewerVerdict:
        del context, previous_feedback
        index = min(self.calls, len(self.decisions) - 1)
        decision = self.decisions[index]
        self.calls += 1
        feedback = ("Revise the declarative plan.",) if decision is ReviewerDecision.REVISE else ()
        findings = () if validation.valid else tuple(item.message for item in validation.findings)
        return ReviewerVerdict(
            plan_id=plan.plan_id,
            attempt=attempt,
            decision=decision,
            findings=findings,
            feedback=feedback,
        )


@dataclass
class RuleBasedPlanner:
    """Offline fallback that proposes only diagnosis-grounded MVP operations."""

    calls: int = 0

    def propose(self, context: PlanningContext, *, attempt: int) -> TransformationPlan:
        del attempt
        self.calls += 1
        operations: list[TransformationOperation] = []
        for issue in context.diagnostic_report.issues:
            if issue.kind is DiagnosticIssueKind.CANDIDATE_TYPE_CONVERSION:
                column = issue.affected_columns[0]
                operations.append(
                    TransformationOperation(
                        operation_id=f"op-cast-{column}",
                        operation_type=OperationType.CAST_COLUMN,
                        target_columns=(column,),
                        parameters=CastColumnParameters(
                            target_type=CastTarget.NUMERIC,
                            errors=CastErrorPolicy.COERCE,
                        ),
                        diagnostic_issue_ids=(issue.issue_id,),
                        rationale="The deterministic profile found numeric text.",
                        expected_effect="Convert parseable numeric text to a numeric dtype.",
                        risk=RiskLevel.MEDIUM,
                        requires_human_approval=True,
                    )
                )
            elif issue.kind is DiagnosticIssueKind.CANDIDATE_NUMERIC_TEXT_NOISE:
                # Two operations in order, citing one issue: the column is
                # numeric text carrying symbols, which justifies both stripping
                # the symbols and the cast that becomes possible afterwards.
                column = issue.affected_columns[0]
                operations.append(
                    TransformationOperation(
                        operation_id=f"op-normalize-{column}",
                        operation_type=OperationType.NORMALIZE_NUMERIC_TEXT,
                        target_columns=(column,),
                        parameters=NormalizeNumericTextParameters(),
                        diagnostic_issue_ids=(issue.issue_id,),
                        rationale="The deterministic profile found numeric text carrying symbols.",
                        expected_effect="Strip currency symbols, thousands separators and surrounding whitespace.",
                        risk=RiskLevel.LOW,
                        requires_human_approval=True,
                    )
                )
                operations.append(
                    TransformationOperation(
                        operation_id=f"op-cast-{column}",
                        operation_type=OperationType.CAST_COLUMN,
                        target_columns=(column,),
                        parameters=CastColumnParameters(
                            target_type=CastTarget.NUMERIC,
                            errors=CastErrorPolicy.COERCE,
                        ),
                        diagnostic_issue_ids=(issue.issue_id,),
                        rationale="The normalized text is now parseable as a number.",
                        expected_effect="Convert the stripped numeric text to a numeric dtype.",
                        risk=RiskLevel.MEDIUM,
                        requires_human_approval=True,
                    )
                )
            elif issue.kind is DiagnosticIssueKind.DUPLICATE_KEYS:
                operations.append(
                    TransformationOperation(
                        operation_id="op-deduplicate-keys-" + "-".join(issue.affected_columns),
                        operation_type=OperationType.DEDUPLICATE_BY_KEYS,
                        target_columns=issue.affected_columns,
                        parameters=DeduplicateByKeysParameters(
                            keys=issue.affected_columns,
                            keep=KeepPolicy.FIRST,
                        ),
                        diagnostic_issue_ids=(issue.issue_id,),
                        rationale="The selected key has deterministic duplicate evidence.",
                        expected_effect="Keep the first row for each approved key.",
                        risk=RiskLevel.HIGH,
                        requires_human_approval=True,
                    )
                )
        return create_transformation_plan(
            dataset_id=context.dataset_identity.dataset_id,
            dataset_fingerprint=context.dataset_identity.fingerprint,
            version=1,
            operations=tuple(operations),
            summary="Offline diagnosis-grounded transformation plan.",
        )


@dataclass
class RuleBasedReviewer:
    calls: int = 0

    def review(
        self,
        context: PlanningContext,
        plan: TransformationPlan,
        validation: PlanValidationResult,
        *,
        previous_feedback: tuple[str, ...],
        attempt: int,
    ) -> ReviewerVerdict:
        del context, previous_feedback
        self.calls += 1
        if not validation.valid:
            decision = ReviewerDecision.REVISE
            findings = tuple(item.message for item in validation.findings)
        elif any(
            operation.risk is RiskLevel.HIGH
            and not operation.requires_human_approval
            for operation in plan.operations
        ):
            decision = ReviewerDecision.REVISE
            findings = ("High-risk operations must require human approval.",)
        else:
            decision = ReviewerDecision.ACCEPT
            findings = ()
        return ReviewerVerdict(
            plan_id=plan.plan_id,
            attempt=attempt,
            decision=decision,
            findings=findings,
            feedback=findings,
        )
