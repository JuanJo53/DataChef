from __future__ import annotations

from datetime import datetime, timezone

from datachef.contracts import (
    AcceptedReviewEvidence,
    HumanApproval,
    HumanDecision,
    ReviewerDecision,
    ReviewerVerdict,
    TransformationPlan,
)
from datachef.planning import accept_review


def accepted_review(plan, validation, *, attempt: int = 1) -> AcceptedReviewEvidence:
    verdict = ReviewerVerdict(
        plan_id=plan.plan_id,
        attempt=attempt,
        decision=ReviewerDecision.ACCEPT,
    )
    return accept_review(plan, validation, verdict, attempt=attempt)


def human_approval(plan: TransformationPlan) -> HumanApproval:
    return HumanApproval(
        dataset_id=plan.dataset_id,
        dataset_fingerprint=plan.dataset_fingerprint,
        plan_id=plan.plan_id,
        plan_version=plan.version,
        decision=HumanDecision.APPROVE,
        approved_operation_ids=tuple(
            operation.operation_id for operation in plan.operations
        ),
        decided_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )
