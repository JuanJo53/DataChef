"""Recompute local validation facts from immutable data and local user intent."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datachef.contracts import (
    DiagnosticReport,
    OperationType,
    PlanValidationResult,
    PlanningContext,
    TransformationPlan,
    UserIntent,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.privacy import build_planning_context


@dataclass(frozen=True)
class AuthoritativeValidationFacts:
    """Facts regenerated at a trust boundary instead of accepted as claims."""

    diagnostic_report: DiagnosticReport
    planning_context: PlanningContext
    validation: PlanValidationResult


def _execution_relevant_context(context: PlanningContext) -> tuple[object, ...]:
    """Exclude opaque/provider and question fields that do not affect execution."""

    return (
        context.dataset_identity,
        context.column_schema,
        context.diagnostic_report,
        context.user_intent,
        context.supported_operations,
        context.privacy_manifest,
    )


def context_claim_matches(
    supplied: PlanningContext,
    authoritative: PlanningContext,
) -> bool:
    return _execution_relevant_context(supplied) == _execution_relevant_context(
        authoritative
    )


def recompute_validation_facts(
    source: pd.DataFrame,
    intent: UserIntent,
    plan: TransformationPlan,
) -> AuthoritativeValidationFacts:
    """Diagnose and validate again from source data using only local intent."""

    from datachef.planning.validation import validate_plan

    report = diagnose_raw_dataframe(
        source,
        selected_key_columns=intent.selected_key_columns,
    )
    context = build_planning_context(
        report,
        intent,
        (),
        supported_operations=tuple(OperationType),
        provider_context_reference="local-validation-only",
    )
    return AuthoritativeValidationFacts(
        diagnostic_report=report,
        planning_context=context,
        validation=validate_plan(context, plan),
    )


__all__ = [
    "AuthoritativeValidationFacts",
    "context_claim_matches",
    "recompute_validation_facts",
]
