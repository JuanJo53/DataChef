"""Deterministic identity construction for declarative plans."""

from __future__ import annotations

from hashlib import sha256
import json

from datachef.contracts import TransformationOperation, TransformationPlan


def _plan_identity(
    *,
    dataset_id: str,
    dataset_fingerprint: str,
    version: int,
    operations: tuple[TransformationOperation, ...],
    summary: str,
) -> str:
    material = {
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "version": version,
        "operations": [operation.model_dump(mode="json") for operation in operations],
        "summary": summary,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    plan_hash = sha256(encoded.encode("utf-8")).hexdigest()
    return f"plan-{plan_hash[:20]}"


def create_transformation_plan(
    *,
    dataset_id: str,
    dataset_fingerprint: str,
    version: int,
    operations: tuple[TransformationOperation, ...],
    summary: str,
) -> TransformationPlan:
    return TransformationPlan(
        plan_id=_plan_identity(
            dataset_id=dataset_id,
            dataset_fingerprint=dataset_fingerprint,
            version=version,
            operations=operations,
            summary=summary,
        ),
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        version=version,
        operations=operations,
        summary=summary,
    )


def expected_plan_id(plan: TransformationPlan) -> str:
    return _plan_identity(
        dataset_id=plan.dataset_id,
        dataset_fingerprint=plan.dataset_fingerprint,
        version=plan.version,
        operations=plan.operations,
        summary=plan.summary,
    )
