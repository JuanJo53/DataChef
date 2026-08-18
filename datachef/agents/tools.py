"""The agent action space, which IS the deterministic allow-list.

One tool per executable operation type, plus non-proposal tools. The tools own a
server-side draft: the agent never holds a plan object, never authors an
identity, and never sees a cell value.

Column names are used exactly as they appear in ``context.column_schema``. There
is no de-aliasing and no translation layer: for ordinary columns the alias map is
the identity, and a privacy-aliased column is not executable at all, so every
proposal tool refuses an aliased target at the tool boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from datachef.agents.trace import ToolInvocation
from datachef.contracts import (
    CastColumnParameters,
    CastErrorPolicy,
    CastTarget,
    DeduplicateByKeysParameters,
    DropDuplicateRowsParameters,
    KeepPolicy,
    NormalizeMissingTokensParameters,
    OperationType,
    PlanningContext,
    RenameColumnParameters,
    RiskLevel,
    TransformationOperation,
    TransformationPlan,
    TrimWhitespaceParameters,
)
from datachef.planning import validate_plan
from datachef.planning.plan import create_transformation_plan

# Reason codes reuse the committed validation vocabulary; no new codes are added.
ALIASED = "ALIASED_COLUMN_NOT_EXECUTABLE"
MISSING_COLUMN = "MISSING_COLUMN"
UNKNOWN_ISSUE = "UNKNOWN_DIAGNOSTIC_ISSUE"
EMPTY_TARGETS = "EMPTY_TARGET_COLUMNS"
INVALID_KEY = "INVALID_KEY"


@dataclass
class PlanDraft:
    """Server-side accumulation of proposed operations."""

    context: PlanningContext
    operations: list[TransformationOperation] = field(default_factory=list)
    invocations: list[ToolInvocation] = field(default_factory=list)
    summary: str = "Agent-proposed transformation plan."

    @property
    def known_columns(self) -> frozenset[str]:
        return frozenset(column.name for column in self.context.column_schema)

    @property
    def aliased_columns(self) -> frozenset[str]:
        return frozenset(self.context.privacy_manifest.aliased_columns)

    @property
    def known_issue_ids(self) -> frozenset[str]:
        return frozenset(
            issue.issue_id for issue in self.context.diagnostic_report.issues
        )

    def next_operation_id(self, operation_type: OperationType) -> str:
        """WE own the operation identity; the agent never supplies one."""

        return f"op-{len(self.operations) + 1:03d}-{operation_type.value.lower()}"

    def record(self, invocation: ToolInvocation) -> None:
        self.invocations.append(invocation)

    def build_plan(self) -> TransformationPlan:
        """Identity, dataset binding, and version are computed, never supplied."""

        identity = self.context.dataset_identity
        return create_transformation_plan(
            dataset_id=identity.dataset_id,
            dataset_fingerprint=identity.fingerprint,
            version=1,
            operations=tuple(self.operations),
            summary=self.summary,
        )


class _ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _OperationArgs(_ToolArgs):
    diagnostic_issue_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    risk: RiskLevel = RiskLevel.MEDIUM
    requires_human_approval: bool = True


class TrimWhitespaceArgs(_OperationArgs):
    target_columns: list[str] = Field(min_length=1)


class NormalizeMissingTokensArgs(_OperationArgs):
    target_columns: list[str] = Field(min_length=1)
    tokens: list[str] = Field(min_length=1)
    case_sensitive: bool = False


class CastColumnArgs(_OperationArgs):
    target_columns: list[str] = Field(min_length=1, max_length=1)
    target_type: CastTarget
    errors: CastErrorPolicy = CastErrorPolicy.RAISE


class RenameColumnArgs(_OperationArgs):
    target_columns: list[str] = Field(min_length=1, max_length=1)
    new_name: str = Field(min_length=1)


class DropDuplicateRowsArgs(_OperationArgs):
    keep: KeepPolicy = KeepPolicy.FIRST


class DeduplicateByKeysArgs(_OperationArgs):
    keys: list[str] = Field(min_length=1)
    keep: KeepPolicy = KeepPolicy.FIRST


class FinalizePlanArgs(_ToolArgs):
    summary: str = Field(min_length=1)


class NoArgs(_ToolArgs):
    pass


def guard_columns(draft: PlanDraft, columns: tuple[str, ...]) -> str | None:
    """Aliased first, then existence. Returns a reason code, or None if usable."""

    if any(column in draft.aliased_columns for column in columns):
        return ALIASED
    if any(column not in draft.known_columns for column in columns):
        return MISSING_COLUMN
    return None


def guard_dedup_keys(draft: PlanDraft, keys: tuple[str, ...]) -> str | None:
    """Refuse a key set the estimator cannot price.

    ``key_duplicate_metrics`` holds only nominated key sets, so a dedup on any
    other column would be estimated at zero rows removed. Refusing here makes
    the unpriceable operation unrepresentable rather than merely mispriced.
    """

    known = {
        tuple(metric.key_columns)
        for metric in draft.context.diagnostic_report.key_duplicate_metrics
    }
    return None if tuple(keys) in known else INVALID_KEY


def guard_issues(draft: PlanDraft, issue_ids: tuple[str, ...]) -> str | None:
    if any(issue_id not in draft.known_issue_ids for issue_id in issue_ids):
        return UNKNOWN_ISSUE
    return None


def propose(
    draft: PlanDraft,
    *,
    tool_name: str,
    operation_type: OperationType,
    target_columns: tuple[str, ...],
    parameters: Any,
    args: _OperationArgs,
) -> dict[str, Any]:
    """Shared proposal path: guard, then append with a computed operation_id."""

    issue_ids = tuple(args.diagnostic_issue_ids)
    reason: str | None = None
    if operation_type is not OperationType.DROP_DUPLICATE_ROWS and not target_columns:
        reason = EMPTY_TARGETS
    if reason is None and target_columns:
        reason = guard_columns(draft, target_columns)
    if reason is None and operation_type is OperationType.DEDUPLICATE_BY_KEYS:
        reason = guard_dedup_keys(draft, target_columns)
    if reason is None:
        reason = guard_issues(draft, issue_ids)
    if reason is None and not issue_ids:
        # The operation contract requires a link to an issue or a requirement.
        reason = UNKNOWN_ISSUE
    if reason is not None:
        draft.record(
            ToolInvocation(
                tool_name=tool_name,
                accepted=False,
                reason_code=reason,
                operation_type=operation_type.value,
                target_columns=target_columns,
            )
        )
        return {"accepted": False, "reason_code": reason}

    operation = TransformationOperation(
        operation_id=draft.next_operation_id(operation_type),
        operation_type=operation_type,
        target_columns=target_columns,
        parameters=parameters,
        diagnostic_issue_ids=issue_ids,
        rationale=args.rationale,
        expected_effect=args.expected_effect,
        risk=args.risk,
        requires_human_approval=args.requires_human_approval,
    )
    draft.operations.append(operation)
    draft.record(
        ToolInvocation(
            tool_name=tool_name,
            accepted=True,
            operation_type=operation_type.value,
            target_columns=target_columns,
        )
    )
    return {"accepted": True, "operation_id": operation.operation_id}


def inspect_profile(draft: PlanDraft) -> dict[str, Any]:
    """Aliased schema and diagnostic shape. No cell values, ever."""

    context = draft.context
    report = context.diagnostic_report
    draft.record(ToolInvocation(tool_name="inspect_profile", accepted=True))
    return {
        "columns": [
            {"name": column.name, "dtype": column.dtype}
            for column in context.column_schema
        ],
        "aliased_columns": list(context.privacy_manifest.aliased_columns),
        "row_count": context.dataset_identity.row_count,
        "duplicate_row_count": report.duplicate_row_count,
        "key_duplicate_metrics": [
            {
                "key_columns": list(metric.key_columns),
                "duplicate_row_count": metric.duplicate_row_count,
                "null_key_row_count": metric.null_key_row_count,
            }
            for metric in report.key_duplicate_metrics
        ],
        "issues": [
            {
                "issue_id": issue.issue_id,
                "kind": issue.kind.value,
                "severity": issue.severity.value,
                "affected_columns": list(issue.affected_columns),
                "suggested_operation": (
                    issue.suggested_operation.value
                    if issue.suggested_operation
                    else None
                ),
            }
            for issue in report.issues
        ],
        "supported_operations": [item.value for item in context.supported_operations],
    }


def estimate_current_plan(draft: PlanDraft) -> dict[str, Any]:
    """The deterministic critic, in-loop. Codes and context names only."""

    validation = validate_plan(draft.context, draft.build_plan())
    codes = tuple(finding.code for finding in validation.findings)
    cumulative = round(validation.cumulative_estimated_row_loss_pct, 4)
    draft.record(
        ToolInvocation(
            tool_name="estimate_current_plan",
            accepted=validation.valid,
            critic_finding_codes=codes,
            estimated_row_loss_pct=cumulative,
        )
    )
    return {
        "operation_count": len(draft.operations),
        "valid": validation.valid,
        "cumulative_estimated_row_loss_pct": round(
            validation.cumulative_estimated_row_loss_pct, 4
        ),
        "acceptable_row_loss_pct": draft.context.user_intent.acceptable_row_loss_pct,
        "row_loss_estimates": [
            {
                "operation_id": estimate.operation_id,
                "estimated_rows": estimate.estimated_rows,
                "estimated_pct": round(estimate.estimated_pct, 4),
            }
            for estimate in validation.row_loss_estimates
        ],
        "findings": [
            {"code": finding.code, "operation_id": finding.operation_id}
            for finding in validation.findings
        ],
    }


def discard_last_operation(draft: PlanDraft) -> dict[str, Any]:
    """Let the agent retract a proposal the critic priced as too destructive."""

    if not draft.operations:
        draft.record(
            ToolInvocation(
                tool_name="discard_last_operation",
                accepted=False,
                reason_code=EMPTY_TARGETS,
            )
        )
        return {"accepted": False, "reason_code": EMPTY_TARGETS}
    removed = draft.operations.pop()
    draft.record(
        ToolInvocation(
            tool_name="discard_last_operation",
            accepted=True,
            operation_type=removed.operation_type.value,
            target_columns=removed.target_columns,
        )
    )
    return {"accepted": True, "discarded_operation_id": removed.operation_id}


def finalize_plan(draft: PlanDraft, summary: str) -> dict[str, Any]:
    """Return a handle to the validated draft, or a typed refusal."""

    draft.summary = summary
    validation = validate_plan(draft.context, draft.build_plan())
    accepted = validation.valid
    draft.record(
        ToolInvocation(
            tool_name="finalize_plan",
            accepted=accepted,
            reason_code=None if accepted else validation.findings[0].code,
        )
    )
    if not accepted:
        return {
            "accepted": False,
            "reason_codes": [finding.code for finding in validation.findings],
        }
    return {"accepted": True, "operation_count": len(draft.operations)}


def build_operation_specs() -> tuple[tuple[str, OperationType, type[_OperationArgs]], ...]:
    """The proposal tool surface, one entry per executable operation type."""

    return (
        ("propose_trim_whitespace", OperationType.TRIM_WHITESPACE, TrimWhitespaceArgs),
        (
            "propose_normalize_missing_tokens",
            OperationType.NORMALIZE_MISSING_TOKENS,
            NormalizeMissingTokensArgs,
        ),
        ("propose_cast_column", OperationType.CAST_COLUMN, CastColumnArgs),
        ("propose_rename_column", OperationType.RENAME_COLUMN, RenameColumnArgs),
        (
            "propose_drop_duplicate_rows",
            OperationType.DROP_DUPLICATE_ROWS,
            DropDuplicateRowsArgs,
        ),
        (
            "propose_deduplicate_by_keys",
            OperationType.DEDUPLICATE_BY_KEYS,
            DeduplicateByKeysArgs,
        ),
    )


def apply_operation_args(
    draft: PlanDraft,
    tool_name: str,
    args: _OperationArgs,
) -> dict[str, Any]:
    """Translate validated tool arguments into a guarded proposal."""

    if isinstance(args, TrimWhitespaceArgs):
        return propose(
            draft,
            tool_name=tool_name,
            operation_type=OperationType.TRIM_WHITESPACE,
            target_columns=tuple(args.target_columns),
            parameters=TrimWhitespaceParameters(),
            args=args,
        )
    if isinstance(args, NormalizeMissingTokensArgs):
        return propose(
            draft,
            tool_name=tool_name,
            operation_type=OperationType.NORMALIZE_MISSING_TOKENS,
            target_columns=tuple(args.target_columns),
            parameters=NormalizeMissingTokensParameters(
                tokens=tuple(args.tokens),
                case_sensitive=args.case_sensitive,
            ),
            args=args,
        )
    if isinstance(args, CastColumnArgs):
        return propose(
            draft,
            tool_name=tool_name,
            operation_type=OperationType.CAST_COLUMN,
            target_columns=tuple(args.target_columns),
            parameters=CastColumnParameters(
                target_type=args.target_type,
                errors=args.errors,
            ),
            args=args,
        )
    if isinstance(args, RenameColumnArgs):
        return propose(
            draft,
            tool_name=tool_name,
            operation_type=OperationType.RENAME_COLUMN,
            target_columns=tuple(args.target_columns),
            parameters=RenameColumnParameters(new_name=args.new_name),
            args=args,
        )
    if isinstance(args, DropDuplicateRowsArgs):
        return propose(
            draft,
            tool_name=tool_name,
            operation_type=OperationType.DROP_DUPLICATE_ROWS,
            target_columns=(),
            parameters=DropDuplicateRowsParameters(keep=args.keep),
            args=args,
        )
    if isinstance(args, DeduplicateByKeysArgs):
        keys = tuple(args.keys)
        return propose(
            draft,
            tool_name=tool_name,
            operation_type=OperationType.DEDUPLICATE_BY_KEYS,
            target_columns=keys,
            parameters=DeduplicateByKeysParameters(keys=keys, keep=args.keep),
            args=args,
        )
    raise TypeError("unsupported tool arguments")


__all__ = [
    "ALIASED",
    "CastColumnArgs",
    "DeduplicateByKeysArgs",
    "DropDuplicateRowsArgs",
    "EMPTY_TARGETS",
    "FinalizePlanArgs",
    "INVALID_KEY",
    "MISSING_COLUMN",
    "NoArgs",
    "NormalizeMissingTokensArgs",
    "PlanDraft",
    "RenameColumnArgs",
    "TrimWhitespaceArgs",
    "UNKNOWN_ISSUE",
    "apply_operation_args",
    "build_operation_specs",
    "discard_last_operation",
    "estimate_current_plan",
    "finalize_plan",
    "guard_columns",
    "guard_dedup_keys",
    "inspect_profile",
    "propose",
]
