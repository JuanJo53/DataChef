"""Build minimized, serializable context for deterministic or LLM planners."""

from __future__ import annotations

from hashlib import sha256
from dataclasses import dataclass
import re
from uuid import uuid4

from datachef.contracts import (
    ColumnSchema,
    DatasetIdentity,
    DiagnosticReport,
    KeyDuplicateMetric,
    NullExpectation,
    OperationType,
    PlanningDiagnosticIssue,
    PlanningDiagnosticReport,
    PlanningContext,
    PlanningIntent,
    PlanningQuestion,
    ProviderDiagnosticReport,
    ProviderPlanningIntent,
    ProviderPlanningPayload,
    ProviderPrivacyManifest,
    PrivacyManifest,
    SuggestedQuestion,
    UserIntent,
)


_EMAIL = re.compile(r"(?i)\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
_LONG_IDENTIFIER = re.compile(r"(?<!\w)[A-Za-z0-9_-]{20,}(?!\w)")
_SENSITIVE_COLUMN = re.compile(
    r"(?i)(?:@|email|mail|phone|tel|ssn|dni|passport|name|address|secret|password|api[_-]?key|url|(?:^|_)(?:access|auth|bearer)_token(?:_|$))"
)


@dataclass(frozen=True)
class ColumnAliasMap:
    """Runtime-only source-to-provider aliases; excluded from serialized state."""

    bindings: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, str]:
        return dict(self.bindings)


def sanitize_user_text(value: str) -> str:
    sanitized = _EMAIL.sub("[REDACTED_EMAIL]", value)
    sanitized = _PHONE.sub("[REDACTED_PHONE]", sanitized)
    sanitized = _LONG_IDENTIFIER.sub("[REDACTED_IDENTIFIER]", sanitized)
    return sanitized


def _column_aliases(
    report: DiagnosticReport,
    intent: UserIntent,
) -> dict[str, str]:
    """Alias policy-marked columns plus high-risk labels; keep ordinary references."""

    protected = set(intent.protected_columns)
    profiled_pii = {
        profile.name for profile in report.column_profiles if profile.possible_pii
    }
    source_names = tuple(column.name for column in report.dataset_identity.column_schema)
    prefix_index = 0
    prefix = "__dc_private_"
    while any(name.startswith(prefix) for name in source_names):
        prefix_index += 1
        prefix = f"__dc{prefix_index}_private_"
    return {
        column.name: (
            f"{prefix}{position:03d}__"
            if column.name in protected
            or column.name in profiled_pii
            or _SENSITIVE_COLUMN.search(column.name)
            else column.name
        )
        for position, column in enumerate(
            report.dataset_identity.column_schema,
            start=1,
        )
    }


def build_column_alias_map(
    report: DiagnosticReport,
    intent: UserIntent,
) -> ColumnAliasMap:
    return ColumnAliasMap(tuple(_column_aliases(report, intent).items()))


def _alias_columns(columns: tuple[str, ...], aliases: dict[str, str]) -> tuple[str, ...]:
    return tuple(aliases.get(column, "unavailable_column") for column in columns)


def _safe_issue_id(report_id: str, issue_id: str) -> str:
    digest = sha256(f"{report_id}|{issue_id}".encode("utf-8")).hexdigest()
    return f"planning-issue-{digest[:16]}"


def _planning_intent(intent: UserIntent, aliases: dict[str, str]) -> PlanningIntent:
    requested_operations = tuple(
        operation
        for operation in OperationType
        if operation.value in intent.explicit_requested_transformations
    )
    return PlanningIntent(
        intent_id=f"intent-{sha256(intent.intent_id.encode('utf-8')).hexdigest()[:16]}",
        downstream_use=intent.downstream_use,
        selected_key_columns=_alias_columns(intent.selected_key_columns, aliases),
        protected_columns=_alias_columns(intent.protected_columns, aliases),
        required_columns=_alias_columns(intent.required_columns, aliases),
        acceptable_row_loss_pct=intent.acceptable_row_loss_pct,
        null_expectations=tuple(
            NullExpectation(
                column=aliases.get(expectation.column, "unavailable_column"),
                maximum_null_pct=expectation.maximum_null_pct,
                mandatory=expectation.mandatory,
            )
            for expectation in intent.null_expectations
        ),
        requested_operation_types=requested_operations,
        pii_handling=intent.pii_handling,
    )


def build_planning_context(
    report: DiagnosticReport,
    intent: UserIntent,
    questions: tuple[SuggestedQuestion, ...],
    *,
    supported_operations: tuple[OperationType, ...] = tuple(OperationType),
    previous_review_feedback: tuple[str, ...] = (),
    provider_context_reference: str | None = None,
    column_alias_map: ColumnAliasMap | None = None,
) -> PlanningContext:
    """Create local row-free state; use the provider projection for any LLM call."""

    aliases = (column_alias_map or build_column_alias_map(report, intent)).as_dict()
    null_counts = {
        aliases[profile.name]: int(profile.null_count)
        for profile in report.column_profiles
    }
    safe_identity = DatasetIdentity(
        dataset_id=report.dataset_identity.dataset_id,
        fingerprint=report.dataset_identity.fingerprint,
        row_count=report.dataset_identity.row_count,
        column_count=report.dataset_identity.column_count,
        column_schema=tuple(
            ColumnSchema(name=aliases[column.name], dtype=column.dtype)
            for column in report.dataset_identity.column_schema
        ),
    )
    safe_intent = _planning_intent(intent, aliases)
    safe_report = PlanningDiagnosticReport(
        report_id=report.report_id,
        issues=tuple(
            PlanningDiagnosticIssue(
                issue_id=_safe_issue_id(report.report_id, issue.issue_id),
                kind=issue.kind,
                severity=issue.severity,
                affected_columns=_alias_columns(issue.affected_columns, aliases),
                affected_row_count=next(
                    (
                        evidence.value
                        for evidence in issue.evidence
                        if isinstance(evidence.value, (int, float))
                    ),
                    None,
                ),
                suggested_operation=issue.suggested_operation,
            )
            for issue in report.issues
        ),
        duplicate_row_count=report.duplicate_row_count,
        key_duplicate_metrics=tuple(
            KeyDuplicateMetric(
                key_columns=_alias_columns(metric.key_columns, aliases),
                duplicate_row_count=metric.duplicate_row_count,
                null_key_row_count=metric.null_key_row_count,
            )
            for metric in report.key_duplicate_metrics
        ),
    )
    safe_questions = tuple(
        PlanningQuestion(
            question_id=(
                "question-"
                + sha256(question.question_id.encode("utf-8")).hexdigest()[:16]
            ),
            kind=question.kind,
            relevant_columns=_alias_columns(question.relevant_columns, aliases),
            confidence=question.confidence,
        )
        for question in questions
    )
    manifest = PrivacyManifest(
        raw_rows_included=False,
        row_samples_included=False,
        filenames_included=False,
        suppressed_content=(
            "raw_dataframe",
            "row_samples",
            "unique_values",
            "source_filename",
            "source_path",
            "credentials",
            "free_form_user_goal",
            "free_form_user_questions",
            "diagnostic_explanations",
            "question_explanations",
            "reviewer_feedback",
            "sensitive_column_names",
        ),
        aliased_columns=tuple(
            alias for name, alias in aliases.items() if alias != name
        ),
    )
    identity_material = "|".join(
        (
            report.report_id,
            safe_intent.model_dump_json(),
            ",".join(question.question_id for question in questions),
            ",".join(operation.value for operation in supported_operations),
        )
    )
    context_hash = sha256(identity_material.encode("utf-8")).hexdigest()
    return PlanningContext(
        context_id=f"context-{context_hash[:16]}",
        provider_context_reference=(
            provider_context_reference or f"provider-context-{uuid4()}"
        ),
        dataset_identity=safe_identity,
        column_schema=safe_identity.column_schema,
        null_counts=null_counts,
        diagnostic_report=safe_report,
        user_intent=safe_intent,
        questions=safe_questions,
        supported_operations=supported_operations,
        privacy_manifest=manifest,
    )


def build_provider_planning_payload(
    context: PlanningContext,
) -> ProviderPlanningPayload:
    """Project local planning state into the only contract safe for a provider."""

    return ProviderPlanningPayload(
        context_reference=context.provider_context_reference,
        row_count=context.dataset_identity.row_count,
        column_count=context.dataset_identity.column_count,
        column_schema=context.column_schema,
        null_counts=context.null_counts,
        diagnostic_report=ProviderDiagnosticReport(
            issues=tuple(
                PlanningDiagnosticIssue(
                    issue_id=f"issue_{position:03d}",
                    kind=issue.kind,
                    severity=issue.severity,
                    affected_columns=issue.affected_columns,
                    affected_row_count=issue.affected_row_count,
                    suggested_operation=issue.suggested_operation,
                )
                for position, issue in enumerate(
                    context.diagnostic_report.issues,
                    start=1,
                )
            ),
            duplicate_row_count=context.diagnostic_report.duplicate_row_count,
            key_duplicate_metrics=context.diagnostic_report.key_duplicate_metrics,
        ),
        user_intent=ProviderPlanningIntent(
            downstream_use=context.user_intent.downstream_use,
            selected_key_columns=context.user_intent.selected_key_columns,
            protected_columns=context.user_intent.protected_columns,
            required_columns=context.user_intent.required_columns,
            acceptable_row_loss_pct=context.user_intent.acceptable_row_loss_pct,
            null_expectations=context.user_intent.null_expectations,
            requested_operation_types=context.user_intent.requested_operation_types,
            pii_handling=context.user_intent.pii_handling,
        ),
        questions=tuple(
            PlanningQuestion(
                question_id=f"question_{position:03d}",
                kind=question.kind,
                relevant_columns=question.relevant_columns,
                confidence=question.confidence,
            )
            for position, question in enumerate(context.questions, start=1)
        ),
        supported_operations=context.supported_operations,
        privacy_manifest=ProviderPrivacyManifest(),
    )
