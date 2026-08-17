"""Semantic binding between a reviewer verdict and a validated plan."""

from __future__ import annotations

from datachef.contracts import (
    AcceptedReviewEvidence,
    PlanValidationResult,
    ReviewerDecision,
    ReviewerVerdict,
    TransformationPlan,
)


class ReviewEvidenceError(ValueError):
    """Sanitized refusal for stale, foreign, or non-accepting review output."""


def accept_review(
    plan: TransformationPlan,
    validation: PlanValidationResult,
    verdict: ReviewerVerdict,
    *,
    attempt: int,
) -> AcceptedReviewEvidence:
    if not validation.valid or validation.plan_id != plan.plan_id:
        raise ReviewEvidenceError("REVIEW_PLAN_NOT_VALID")
    if verdict.decision is not ReviewerDecision.ACCEPT:
        raise ReviewEvidenceError("REVIEW_NOT_ACCEPTED")
    if verdict.plan_id != plan.plan_id:
        raise ReviewEvidenceError("REVIEW_PLAN_MISMATCH")
    if verdict.attempt != attempt:
        raise ReviewEvidenceError("REVIEW_ATTEMPT_MISMATCH")
    return AcceptedReviewEvidence(
        dataset_id=plan.dataset_id,
        dataset_fingerprint=plan.dataset_fingerprint,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        attempt=attempt,
        validation_plan_id=validation.plan_id,
        validation_valid=True,
        decision=ReviewerDecision.ACCEPT,
    )


__all__ = ["ReviewEvidenceError", "accept_review"]
