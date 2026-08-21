"""Strict boundary contracts for the offline DataChef workflow."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class WorkflowStage(StrEnum):
    INITIAL = "INITIAL"
    DIAGNOSED = "DIAGNOSED"
    INTENT_CAPTURED = "INTENT_CAPTURED"
    CONTEXT_READY = "CONTEXT_READY"
    PLANNING = "PLANNING"
    PLAN_REJECTED = "PLAN_REJECTED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    EXECUTING = "EXECUTING"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    QA_PASSED = "QA_PASSED"
    QA_WARNING = "QA_WARNING"
    QA_FAILED = "QA_FAILED"


class OperationType(StrEnum):
    TRIM_WHITESPACE = "TRIM_WHITESPACE"
    NORMALIZE_MISSING_TOKENS = "NORMALIZE_MISSING_TOKENS"
    CAST_COLUMN = "CAST_COLUMN"
    RENAME_COLUMN = "RENAME_COLUMN"
    DROP_DUPLICATE_ROWS = "DROP_DUPLICATE_ROWS"
    DEDUPLICATE_BY_KEYS = "DEDUPLICATE_BY_KEYS"
    DROP_COLUMN = "DROP_COLUMN"
    IMPUTE_MISSING = "IMPUTE_MISSING"
    NORMALIZE_NUMERIC_TEXT = "NORMALIZE_NUMERIC_TEXT"
    COMPUTE_COLUMN = "COMPUTE_COLUMN"


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ReviewerDecision(StrEnum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    REJECT = "REJECT"


class HumanDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class QAStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class DownstreamUse(StrEnum):
    BI = "BI"
    ML = "ML"
    ANALYSIS = "ANALYSIS"
    GENERAL_CLEANUP = "GENERAL_CLEANUP"


class PIIHandling(StrEnum):
    NONE = "NONE"
    MASK = "MASK"
    REMOVE = "REMOVE"


class RequestAssessmentStatus(StrEnum):
    PLANNED = "PLANNED"
    ALREADY_SATISFIED = "ALREADY_SATISFIED"
    BLOCKED_UNPLANNED = "BLOCKED_UNPLANNED"


class IssueClassification(StrEnum):
    OBSERVED_DEFECT = "OBSERVED_DEFECT"
    CANDIDATE_CONVERSION = "CANDIDATE_CONVERSION"
    PRIVACY_RISK = "PRIVACY_RISK"


class DiagnosticIssueKind(StrEnum):
    NULL_VALUES = "NULL_VALUES"
    DUPLICATE_ROWS = "DUPLICATE_ROWS"
    DUPLICATE_KEYS = "DUPLICATE_KEYS"
    POSSIBLE_PII = "POSSIBLE_PII"
    CANDIDATE_TYPE_CONVERSION = "CANDIDATE_TYPE_CONVERSION"
    # Numeric text that only fails the plain-numeric test because of noise a
    # NORMALIZE_NUMERIC_TEXT pass removes. Mutually exclusive with
    # CANDIDATE_TYPE_CONVERSION: a column already plain-numeric never gets this.
    CANDIDATE_NUMERIC_TEXT_NOISE = "CANDIDATE_NUMERIC_TEXT_NOISE"
    NULL_KEYS = "NULL_KEYS"
    MISSING_KEY_COLUMN = "MISSING_KEY_COLUMN"


class QuestionKind(StrEnum):
    TREND = "TREND"
    CATEGORY_COMPARISON = "CATEGORY_COMPARISON"
    MISSINGNESS = "MISSINGNESS"
    DUPLICATE_KEYS = "DUPLICATE_KEYS"
    DISTRIBUTION = "DISTRIBUTION"
    RELATIONSHIP = "RELATIONSHIP"


class CastTarget(StrEnum):
    STRING = "STRING"
    NUMERIC = "NUMERIC"
    BOOLEAN = "BOOLEAN"
    DATETIME = "DATETIME"


class CastErrorPolicy(StrEnum):
    RAISE = "RAISE"
    COERCE = "COERCE"


class KeepPolicy(StrEnum):
    FIRST = "FIRST"
    LAST = "LAST"


class ImputeStrategy(StrEnum):
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    MODE = "MODE"
    CONSTANT = "CONSTANT"


class ComputeOperator(StrEnum):
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"


class OperationExecutionStatus(StrEnum):
    APPLIED = "APPLIED"
    FAILED = "FAILED"


class InvariantKind(StrEnum):
    REQUIRED_COLUMN = "REQUIRED_COLUMN"
    MAX_ROW_LOSS = "MAX_ROW_LOSS"
    DTYPE = "DTYPE"
    KEY_UNIQUENESS = "KEY_UNIQUENESS"
    NULL_PCT_MAX = "NULL_PCT_MAX"
    PROTECTED_COLUMN_UNCHANGED = "PROTECTED_COLUMN_UNCHANGED"
    CAST_VALUE_PRESERVATION = "CAST_VALUE_PRESERVATION"
    # Imputation rewrites cells, so it gets its own assertion: nothing that was
    # already populated may change, and the null count must actually fall.
    IMPUTATION_VALUE_PRESERVATION = "IMPUTATION_VALUE_PRESERVATION"
    # Normalization prepares text for a cast; it may never null a value.
    NUMERIC_TEXT_NO_NULLS = "NUMERIC_TEXT_NO_NULLS"
    # A dropped column must take nothing else with it.
    DROPPED_COLUMN_STRUCTURE = "DROPPED_COLUMN_STRUCTURE"
    COMPUTED_COLUMN_ISOLATION = "COMPUTED_COLUMN_ISOLATION"
    PROVENANCE = "PROVENANCE"


class InvariantStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class DiagnosticResolution(StrEnum):
    RESOLVED = "RESOLVED"
    IMPROVED = "IMPROVED"
    UNCHANGED = "UNCHANGED"
    WORSENED = "WORSENED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ColumnSchema(StrictContract):
    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)


class DatasetIdentity(StrictContract):
    dataset_id: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    column_schema: tuple[ColumnSchema, ...]

    @model_validator(mode="after")
    def validate_shape(self) -> "DatasetIdentity":
        if self.column_count != len(self.column_schema):
            raise ValueError("column_count must match schema length")
        if len({column.name for column in self.column_schema}) != len(
            self.column_schema
        ):
            raise ValueError("schema column names must be unique")
        return self


class MetricEvidence(StrictContract):
    metric: str = Field(min_length=1)
    value: int | float | str
    unit: str | None = None


class DiagnosticIssue(StrictContract):
    issue_id: str = Field(min_length=1)
    kind: DiagnosticIssueKind
    classification: IssueClassification
    title: str = Field(min_length=1)
    severity: Severity
    affected_columns: tuple[str, ...] = Field(default_factory=tuple)
    evidence: tuple[MetricEvidence, ...] = Field(default_factory=tuple)
    suggested_operation: OperationType | None = None
    explanation: str = Field(min_length=1)


class ColumnProfile(StrictContract):
    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    sql_type: str = Field(min_length=1)
    null_count: int = Field(ge=0)
    null_pct: float = Field(ge=0, le=100)
    unique_count: int = Field(ge=0)
    zero_count: int = Field(ge=0)
    is_primary_key_candidate: bool
    possible_pii: bool


class KeyDuplicateMetric(StrictContract):
    key_columns: tuple[str, ...] = Field(min_length=1)
    duplicate_row_count: int = Field(ge=0)
    null_key_row_count: int = Field(default=0, ge=0)


class LegacyDiagnosticEvidence(StrictContract):
    health_score: int = Field(ge=0, le=100)
    health_grade: str = Field(min_length=1)
    completeness_pct: float = Field(ge=0, le=100)
    uniqueness_pct: float = Field(ge=0, le=100)
    suggested_primary_key: str | None = None


class DiagnosticReport(StrictContract):
    report_id: str = Field(min_length=1)
    dataset_identity: DatasetIdentity
    column_profiles: tuple[ColumnProfile, ...]
    issues: tuple[DiagnosticIssue, ...] = Field(default_factory=tuple)
    duplicate_row_count: int = Field(ge=0)
    key_duplicate_metrics: tuple[KeyDuplicateMetric, ...] = Field(
        default_factory=tuple
    )
    legacy_evidence: LegacyDiagnosticEvidence


class NullExpectation(StrictContract):
    column: str = Field(min_length=1)
    maximum_null_pct: float = Field(ge=0, le=100)
    mandatory: bool = True


class UserIntent(StrictContract):
    intent_id: str = Field(min_length=1)
    user_goal: str = ""
    downstream_use: DownstreamUse = DownstreamUse.GENERAL_CLEANUP
    selected_key_columns: tuple[str, ...] = Field(default_factory=tuple)
    protected_columns: tuple[str, ...] = Field(default_factory=tuple)
    required_columns: tuple[str, ...] = Field(default_factory=tuple)
    acceptable_row_loss_pct: float = Field(default=0, ge=0, le=100)
    null_expectations: tuple[NullExpectation, ...] = Field(default_factory=tuple)
    explicit_requested_transformations: tuple[str, ...] = Field(
        default_factory=tuple
    )
    questions: tuple[str, ...] = Field(default_factory=tuple)
    pii_handling: PIIHandling = PIIHandling.NONE

    @model_validator(mode="after")
    def validate_column_sets(self) -> "UserIntent":
        protected = set(self.protected_columns)
        if len(protected) != len(self.protected_columns):
            raise ValueError("protected columns must be unique")
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("required columns must be unique")
        if len(set(self.selected_key_columns)) != len(self.selected_key_columns):
            raise ValueError("selected key columns must be unique")
        return self


class PlanningIntent(StrictContract):
    """Provider-safe, structured projection of locally held user intent."""

    intent_id: str = Field(min_length=1)
    downstream_use: DownstreamUse
    selected_key_columns: tuple[str, ...] = Field(default_factory=tuple)
    protected_columns: tuple[str, ...] = Field(default_factory=tuple)
    required_columns: tuple[str, ...] = Field(default_factory=tuple)
    acceptable_row_loss_pct: float = Field(ge=0, le=100)
    null_expectations: tuple[NullExpectation, ...] = Field(default_factory=tuple)
    requested_operation_types: tuple[OperationType, ...] = Field(default_factory=tuple)
    pii_handling: PIIHandling


class SuggestedQuestion(StrictContract):
    question_id: str = Field(min_length=1)
    kind: QuestionKind
    question: str = Field(min_length=1)
    relevant_columns: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    limitations: tuple[str, ...] = Field(default_factory=tuple)


class PrivacyManifest(StrictContract):
    raw_rows_included: bool = False
    row_samples_included: bool = False
    filenames_included: bool = False
    suppressed_content: tuple[str, ...] = Field(default_factory=tuple)
    aliased_columns: tuple[str, ...] = Field(default_factory=tuple)


class PlanningDiagnosticIssue(StrictContract):
    issue_id: str = Field(min_length=1)
    kind: DiagnosticIssueKind
    severity: Severity
    affected_columns: tuple[str, ...] = Field(default_factory=tuple)
    affected_row_count: int | float | None = Field(default=None, ge=0)
    suggested_operation: OperationType | None = None


class PlanningDiagnosticReport(StrictContract):
    report_id: str = Field(min_length=1)
    issues: tuple[PlanningDiagnosticIssue, ...] = Field(default_factory=tuple)
    duplicate_row_count: int = Field(ge=0)
    key_duplicate_metrics: tuple[KeyDuplicateMetric, ...] = Field(default_factory=tuple)


class PlanningQuestion(StrictContract):
    question_id: str = Field(min_length=1)
    kind: QuestionKind
    relevant_columns: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(ge=0, le=1)

class PlanningColumnStatistics(StrictContract):
    column: str
    null_count: int = Field(ge=0)
    zero_count: int = Field(ge=0)


class PlanningContext(StrictContract):
    """Local row-free planning state; never serialize this object to a provider."""

    context_id: str = Field(min_length=1)
    provider_context_reference: str = Field(min_length=1)
    dataset_identity: DatasetIdentity
    column_schema: tuple[ColumnSchema, ...]
    null_counts: dict[str, int] = Field(default_factory=dict)
    diagnostic_report: PlanningDiagnosticReport
    user_intent: PlanningIntent
    questions: tuple[PlanningQuestion, ...] = Field(default_factory=tuple)
    supported_operations: tuple[OperationType, ...]
    privacy_manifest: PrivacyManifest
    column_statistics: tuple[PlanningColumnStatistics, ...] = Field(default_factory=tuple)


class ProviderPrivacyManifest(StrictContract):
    raw_rows_included: Literal[False] = False
    row_samples_included: Literal[False] = False
    filenames_included: Literal[False] = False
    free_form_text_included: Literal[False] = False


class ProviderPlanningIntent(StrictContract):
    downstream_use: DownstreamUse
    selected_key_columns: tuple[str, ...] = Field(default_factory=tuple)
    protected_columns: tuple[str, ...] = Field(default_factory=tuple)
    required_columns: tuple[str, ...] = Field(default_factory=tuple)
    acceptable_row_loss_pct: float = Field(ge=0, le=100)
    null_expectations: tuple[NullExpectation, ...] = Field(default_factory=tuple)
    requested_operation_types: tuple[OperationType, ...] = Field(default_factory=tuple)
    pii_handling: PIIHandling


class ProviderDiagnosticReport(StrictContract):
    issues: tuple[PlanningDiagnosticIssue, ...] = Field(default_factory=tuple)
    duplicate_row_count: int = Field(ge=0)
    key_duplicate_metrics: tuple[KeyDuplicateMetric, ...] = Field(default_factory=tuple)


class ProviderPlanningPayload(StrictContract):
    """The only Phase 1A planning contract permitted across a provider boundary."""

    context_reference: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    column_schema: tuple[ColumnSchema, ...]
    null_counts: dict[str, int] = Field(default_factory=dict)
    diagnostic_report: ProviderDiagnosticReport
    user_intent: ProviderPlanningIntent
    questions: tuple[PlanningQuestion, ...] = Field(default_factory=tuple)
    supported_operations: tuple[OperationType, ...]
    privacy_manifest: ProviderPrivacyManifest
    column_statistics: tuple[PlanningColumnStatistics, ...] = Field(default_factory=tuple)


class TrimWhitespaceParameters(StrictContract):
    kind: Literal["TRIM_WHITESPACE"] = "TRIM_WHITESPACE"


class NormalizeMissingTokensParameters(StrictContract):
    kind: Literal["NORMALIZE_MISSING_TOKENS"] = "NORMALIZE_MISSING_TOKENS"
    tokens: tuple[str, ...] = Field(min_length=1)
    case_sensitive: bool = False


class CastColumnParameters(StrictContract):
    kind: Literal["CAST_COLUMN"] = "CAST_COLUMN"
    target_type: CastTarget
    errors: CastErrorPolicy = CastErrorPolicy.RAISE
    datetime_format: str | None = None
    utc: bool = False
    true_values: tuple[str, ...] = ("true", "1", "yes")
    false_values: tuple[str, ...] = ("false", "0", "no")


class RenameColumnParameters(StrictContract):
    kind: Literal["RENAME_COLUMN"] = "RENAME_COLUMN"
    new_name: str = Field(min_length=1)


class DropDuplicateRowsParameters(StrictContract):
    kind: Literal["DROP_DUPLICATE_ROWS"] = "DROP_DUPLICATE_ROWS"
    keep: KeepPolicy


class DeduplicateByKeysParameters(StrictContract):
    kind: Literal["DEDUPLICATE_BY_KEYS"] = "DEDUPLICATE_BY_KEYS"
    keys: tuple[str, ...] = Field(min_length=1)
    keep: KeepPolicy


class DropColumnParameters(StrictContract):
    """The columns to drop are the operation's target columns.

    Shaped like TrimWhitespaceParameters, the closest precedent: a multi-column
    operation with no options carries no fields of its own, so there is exactly
    one place a column list can live and no way for two lists to disagree.
    """

    kind: Literal["DROP_COLUMN"] = "DROP_COLUMN"


class ImputeMissingParameters(StrictContract):
    kind: Literal["IMPUTE_MISSING"] = "IMPUTE_MISSING"
    strategy: ImputeStrategy
    # Only meaningful for CONSTANT. A closed scalar union, never an expression.
    constant_value: str | int | float | bool | None = None


class NormalizeNumericTextParameters(StrictContract):
    """Named noise classes only: no caller-supplied regex, no expression.

    Parenthesised accounting negatives are one fixed, application-owned grammar;
    they are not an expression supplied by a user. An invalid token remains
    invalid text so the following cast and CAST_VALUE_PRESERVATION still fail
    closed rather than inventing a number.
    """

    kind: Literal["NORMALIZE_NUMERIC_TEXT"] = "NORMALIZE_NUMERIC_TEXT"
    strip_whitespace: bool = True
    strip_currency_symbols: bool = True
    strip_thousands_separators: bool = True


class ComputeColumnParameters(StrictContract):
    """One closed binary numeric derivation; arbitrary expressions do not exist."""

    kind: Literal["COMPUTE_COLUMN"] = "COMPUTE_COLUMN"
    left_column: str = Field(min_length=1)
    right_column: str = Field(min_length=1)
    output_column: str = Field(min_length=1)
    operator: ComputeOperator

    @model_validator(mode="after")
    def validate_column_names(self) -> "ComputeColumnParameters":
        if any(
            not name.strip()
            for name in (self.left_column, self.right_column, self.output_column)
        ):
            raise ValueError("computed-column names must contain non-whitespace text")
        return self


OperationParameters = Annotated[
    TrimWhitespaceParameters
    | NormalizeMissingTokensParameters
    | CastColumnParameters
    | RenameColumnParameters
    | DropDuplicateRowsParameters
    | DeduplicateByKeysParameters
    | DropColumnParameters
    | ImputeMissingParameters
    | NormalizeNumericTextParameters
    | ComputeColumnParameters,
    Field(discriminator="kind"),
]


class TransformationOperation(StrictContract):
    operation_id: str = Field(min_length=1)
    operation_type: OperationType
    target_columns: tuple[str, ...] = Field(default_factory=tuple)
    parameters: OperationParameters
    diagnostic_issue_ids: tuple[str, ...] = Field(default_factory=tuple)
    user_requirement_ids: tuple[str, ...] = Field(default_factory=tuple)
    rationale: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    risk: RiskLevel
    requires_human_approval: bool

    @model_validator(mode="after")
    def validate_parameter_discriminator(self) -> "TransformationOperation":
        if self.parameters.kind != self.operation_type.value:
            raise ValueError("operation type and parameter kind must match")
        if not self.diagnostic_issue_ids and not self.user_requirement_ids:
            raise ValueError("operation must link to an issue or user requirement")
        return self


class RequestedOperation(StrictContract):
    """One typed operation the user explicitly asked for, compiled locally.

    The carrier between the application layer, which compiles a free-form
    objective into typed requests, and the deterministic planner, which must
    account for them. It holds no free-form text: an operation type, the target
    columns, and the same closed parameter union the executor allow-lists.
    """

    request_id: str = Field(min_length=1)
    operation_type: OperationType
    target_columns: tuple[str, ...] = Field(min_length=1)
    parameters: OperationParameters


class RequestAssessment(StrictContract):
    """Typed planning evidence for one explicit user request."""

    request_id: str = Field(min_length=1)
    status: RequestAssessmentStatus
    matched_operation_ids: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_matching_evidence(self) -> "RequestAssessment":
        has_matches = bool(self.matched_operation_ids)
        if (self.status is RequestAssessmentStatus.PLANNED) != has_matches:
            raise ValueError("only planned requests carry matching operations")
        return self


class TransformationPlan(StrictContract):
    plan_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: int = Field(ge=1)
    operations: tuple[TransformationOperation, ...]
    summary: str = Field(min_length=1)


class PlanValidationFinding(StrictContract):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    operation_id: str | None = None
    severity: Severity


class PlanValidationResult(StrictContract):
    plan_id: str = Field(min_length=1)
    valid: bool
    findings: tuple[PlanValidationFinding, ...] = Field(default_factory=tuple)
    row_loss_estimates: tuple["RowLossEstimate", ...] = Field(default_factory=tuple)
    cumulative_estimated_row_loss_pct: float = Field(default=0, ge=0, le=100)

    @model_validator(mode="after")
    def validate_consistency(self) -> "PlanValidationResult":
        if self.valid == bool(self.findings):
            raise ValueError("valid must be true exactly when findings are empty")
        return self


class RowLossEstimate(StrictContract):
    operation_id: str = Field(min_length=1)
    estimated_rows: int = Field(ge=0)
    estimated_pct: float = Field(ge=0, le=100)


class ReviewerVerdict(StrictContract):
    plan_id: str = Field(min_length=1)
    attempt: int = Field(ge=1, le=3)
    decision: ReviewerDecision
    findings: tuple[str, ...] = Field(default_factory=tuple)
    feedback: tuple[str, ...] = Field(default_factory=tuple)


class AcceptedReviewEvidence(StrictContract):
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_id: str = Field(min_length=1)
    plan_version: int = Field(ge=1)
    attempt: int = Field(ge=1, le=3)
    validation_plan_id: str = Field(min_length=1)
    validation_valid: Literal[True] = True
    decision: Literal[ReviewerDecision.ACCEPT] = ReviewerDecision.ACCEPT

    @model_validator(mode="after")
    def validate_plan_binding(self) -> "AcceptedReviewEvidence":
        if self.validation_plan_id != self.plan_id:
            raise ValueError("accepted review validation must reference the same plan")
        return self

    def require_matching_final_verdict(
        self,
        review_history: tuple[ReviewerVerdict, ...],
        *,
        current_plan_id: str,
        current_attempt: int,
    ) -> ReviewerVerdict:
        """Return the recorded final ACCEPT or reject incoherent review history."""

        if not review_history:
            raise ValueError("accepted review requires recorded review history")
        attempts = tuple(verdict.attempt for verdict in review_history)
        if any(current >= following for current, following in zip(attempts, attempts[1:])):
            raise ValueError("review attempts must be recorded in strictly increasing order")
        final_verdict = review_history[-1]
        if any(
            verdict.decision is not ReviewerDecision.REVISE
            for verdict in review_history[:-1]
        ):
            raise ValueError("only revisions may precede the final accepted review")
        if (
            final_verdict.decision is not ReviewerDecision.ACCEPT
            or final_verdict.plan_id != current_plan_id
            or final_verdict.attempt != current_attempt
            or self.plan_id != final_verdict.plan_id
            or self.attempt != final_verdict.attempt
        ):
            raise ValueError("accepted-review receipt must match the final verdict")
        return final_verdict


class HumanApproval(StrictContract):
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_id: str = Field(min_length=1)
    plan_version: int = Field(ge=1)
    decision: HumanDecision
    approved_operation_ids: tuple[str, ...]
    decided_at: datetime


class OperationExecutionRecord(StrictContract):
    operation_id: str = Field(min_length=1)
    status: OperationExecutionStatus
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)
    affected_cell_count: int = Field(ge=0)
    introduced_null_count: int | None = Field(default=None, ge=0)
    # Imputation changes values, so it is measured rather than trusted: how many
    # already-populated cells the handler altered (must be zero) and how many
    # nulls it filled (must be positive). Both are cross-checked against replay.
    changed_non_null_count: int | None = Field(default=None, ge=0)
    filled_null_count: int | None = Field(default=None, ge=0)
    error_code: str | None = None


class ExecutionResult(StrictContract):
    execution_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    plan_version: int = Field(ge=1)
    accepted_review_attempt: int = Field(ge=1, le=3)
    success: bool
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_fingerprint: str | None = None
    before_row_count: int = Field(ge=0)
    after_row_count: int = Field(ge=0)
    before_column_count: int = Field(ge=0)
    after_column_count: int = Field(ge=0)
    operation_records: tuple[OperationExecutionRecord, ...] = Field(
        default_factory=tuple
    )
    error_code: str | None = None

    @model_validator(mode="after")
    def validate_execution_consistency(self) -> "ExecutionResult":
        failed_records = any(
            record.status is OperationExecutionStatus.FAILED
            for record in self.operation_records
        )
        if self.success:
            if self.result_fingerprint is None or self.error_code is not None or failed_records:
                raise ValueError("successful execution must have a result and no failures")
        elif self.result_fingerprint is not None or self.error_code is None:
            raise ValueError("failed execution cannot claim a trusted result")
        if not self.success and not failed_records:
            raise ValueError("failed execution must identify a failed operation")
        if self.operation_records:
            if self.operation_records[0].rows_before != self.before_row_count:
                raise ValueError("first operation must start at the recorded row count")
            if self.operation_records[-1].rows_after != self.after_row_count:
                raise ValueError("last operation must end at the recorded row count")
            for previous, current in zip(
                self.operation_records,
                self.operation_records[1:],
            ):
                if previous.rows_after != current.rows_before:
                    raise ValueError("operation row counts must form a coherent chain")
        return self


class QualityInvariant(StrictContract):
    invariant_id: str = Field(min_length=1)
    kind: InvariantKind
    mandatory: bool = True
    column: str | None = None
    key_columns: tuple[str, ...] = Field(default_factory=tuple)
    expected_dtype: str | None = None
    maximum_pct: float | None = Field(default=None, ge=0, le=100)


class InvariantResult(StrictContract):
    invariant_id: str = Field(min_length=1)
    kind: InvariantKind
    status: InvariantStatus
    mandatory: bool
    explanation: str = Field(min_length=1)
    observed_value: int | float | str | None = None
    expected_value: int | float | str | None = None


class DTypeChange(StrictContract):
    column: str = Field(min_length=1)
    before: str = Field(min_length=1)
    after: str = Field(min_length=1)


class NullCountChange(StrictContract):
    column: str = Field(min_length=1)
    before: int = Field(ge=0)
    after: int = Field(ge=0)


class DiagnosticIssueComparison(StrictContract):
    issue_id: str = Field(min_length=1)
    status: DiagnosticResolution
    before_value: int | float | str | None = None
    after_value: int | float | str | None = None
    explanation: str = Field(min_length=1)


class QAReport(StrictContract):
    qa_report_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    status: QAStatus
    before_row_count: int = Field(ge=0)
    after_row_count: int = Field(ge=0)
    before_column_count: int = Field(ge=0)
    after_column_count: int = Field(ge=0)
    row_loss_pct: float = Field(ge=0, le=100)
    added_columns: tuple[str, ...] = Field(default_factory=tuple)
    removed_columns: tuple[str, ...] = Field(default_factory=tuple)
    renamed_columns: tuple[str, ...] = Field(default_factory=tuple)
    dtype_changes: tuple[DTypeChange, ...] = Field(default_factory=tuple)
    null_count_changes: tuple[NullCountChange, ...] = Field(default_factory=tuple)
    duplicate_rows_before: int = Field(ge=0)
    duplicate_rows_after: int = Field(ge=0)
    duplicate_keys_before: int | None = Field(default=None, ge=0)
    duplicate_keys_after: int | None = Field(default=None, ge=0)
    invariant_results: tuple[InvariantResult, ...] = Field(default_factory=tuple)
    diagnostic_comparisons: tuple[DiagnosticIssueComparison, ...] = Field(
        default_factory=tuple
    )
    execution_failures: tuple[str, ...] = Field(default_factory=tuple)


class WorkflowState(StrictContract):
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    stage: WorkflowStage = WorkflowStage.INITIAL
    dataset_identity: DatasetIdentity | None = None
    diagnostic_report: DiagnosticReport | None = None
    user_intent: PlanningIntent | None = None
    suggested_questions: tuple[SuggestedQuestion, ...] = Field(default_factory=tuple)
    planning_context: PlanningContext | None = None
    transformation_plan: TransformationPlan | None = None
    plan_validation: PlanValidationResult | None = None
    review_history: tuple[ReviewerVerdict, ...] = Field(default_factory=tuple)
    accepted_review: AcceptedReviewEvidence | None = None
    human_approval: HumanApproval | None = None
    execution_result: ExecutionResult | None = None
    qa_report: QAReport | None = None
    planning_attempts: int = Field(default=0, ge=0, le=3)
    last_error_code: str | None = None

    @model_validator(mode="after")
    def validate_stage_evidence(self) -> "WorkflowState":
        diagnosed = {
            WorkflowStage.DIAGNOSED,
            WorkflowStage.INTENT_CAPTURED,
            WorkflowStage.CONTEXT_READY,
            WorkflowStage.PLANNING,
            WorkflowStage.PLAN_REJECTED,
            WorkflowStage.AWAITING_APPROVAL,
            WorkflowStage.EXECUTING,
            WorkflowStage.EXECUTION_FAILED,
            WorkflowStage.QA_PASSED,
            WorkflowStage.QA_WARNING,
            WorkflowStage.QA_FAILED,
        }
        if self.stage in diagnosed:
            if self.dataset_identity is None or self.diagnostic_report is None:
                raise ValueError("diagnosed workflow stages require dataset evidence")
            if self.diagnostic_report.dataset_identity != self.dataset_identity:
                raise ValueError("diagnostic report must match workflow dataset")
        if self.stage in {
            WorkflowStage.INTENT_CAPTURED,
            WorkflowStage.CONTEXT_READY,
            WorkflowStage.PLANNING,
            WorkflowStage.AWAITING_APPROVAL,
            WorkflowStage.EXECUTING,
            WorkflowStage.EXECUTION_FAILED,
            WorkflowStage.QA_PASSED,
            WorkflowStage.QA_WARNING,
            WorkflowStage.QA_FAILED,
        } and self.user_intent is None:
            raise ValueError("intent-aware workflow stages require safe intent evidence")
        if self.stage in {
            WorkflowStage.CONTEXT_READY,
            WorkflowStage.PLANNING,
            WorkflowStage.AWAITING_APPROVAL,
            WorkflowStage.EXECUTING,
            WorkflowStage.EXECUTION_FAILED,
            WorkflowStage.QA_PASSED,
            WorkflowStage.QA_WARNING,
            WorkflowStage.QA_FAILED,
        }:
            if self.planning_context is None or self.planning_attempts < 1:
                raise ValueError("planning stages require context and a positive attempt")
            if (
                self.planning_context.dataset_identity.dataset_id
                != self.dataset_identity.dataset_id
                or self.planning_context.dataset_identity.fingerprint
                != self.dataset_identity.fingerprint
            ):
                raise ValueError("planning context must match workflow dataset")
        planned_stages = {
            WorkflowStage.PLANNING,
            WorkflowStage.AWAITING_APPROVAL,
            WorkflowStage.EXECUTING,
            WorkflowStage.EXECUTION_FAILED,
            WorkflowStage.QA_PASSED,
            WorkflowStage.QA_WARNING,
            WorkflowStage.QA_FAILED,
        }
        if self.stage in planned_stages:
            if self.transformation_plan is None or self.plan_validation is None:
                raise ValueError("planned stages require a plan and validation")
            if self.plan_validation.plan_id != self.transformation_plan.plan_id:
                raise ValueError("plan validation must match the current plan")
        accepted_stages = {
            WorkflowStage.AWAITING_APPROVAL,
            WorkflowStage.EXECUTING,
            WorkflowStage.EXECUTION_FAILED,
            WorkflowStage.QA_PASSED,
            WorkflowStage.QA_WARNING,
            WorkflowStage.QA_FAILED,
        }
        if self.stage in accepted_stages:
            if self.accepted_review is None or not self.plan_validation or not self.plan_validation.valid:
                raise ValueError("accepted stages require valid accepted-review evidence")
            plan = self.transformation_plan
            evidence = self.accepted_review
            if (
                plan is None
                or evidence.plan_id != plan.plan_id
                or evidence.plan_version != plan.version
                or evidence.dataset_id != plan.dataset_id
                or evidence.dataset_fingerprint != plan.dataset_fingerprint
                or evidence.attempt != self.planning_attempts
            ):
                raise ValueError("accepted review evidence must match workflow plan")
            evidence.require_matching_final_verdict(
                self.review_history,
                current_plan_id=plan.plan_id,
                current_attempt=self.planning_attempts,
            )
        if self.stage in {
            WorkflowStage.EXECUTING,
            WorkflowStage.EXECUTION_FAILED,
            WorkflowStage.QA_PASSED,
            WorkflowStage.QA_WARNING,
            WorkflowStage.QA_FAILED,
        }:
            if self.human_approval is None or self.transformation_plan is None:
                raise ValueError("execution stages require human approval")
            approval = self.human_approval
            plan = self.transformation_plan
            if (
                approval.decision is not HumanDecision.APPROVE
                or approval.dataset_id != plan.dataset_id
                or approval.dataset_fingerprint != plan.dataset_fingerprint
                or approval.plan_id != plan.plan_id
                or approval.plan_version != plan.version
                or approval.approved_operation_ids
                != tuple(operation.operation_id for operation in plan.operations)
            ):
                raise ValueError("human approval must match the executing plan")
        if self.stage is WorkflowStage.PLAN_REJECTED and self.accepted_review is not None:
            raise ValueError("rejected workflow cannot retain accepted-review evidence")
        executed_stages = {
            WorkflowStage.EXECUTION_FAILED,
            WorkflowStage.QA_PASSED,
            WorkflowStage.QA_WARNING,
            WorkflowStage.QA_FAILED,
        }
        if self.stage in executed_stages and self.execution_result is None:
            raise ValueError("executed stages require execution evidence")
        if self.execution_result is not None and self.transformation_plan is not None:
            if (
                self.execution_result.plan_id != self.transformation_plan.plan_id
                or self.execution_result.plan_version != self.transformation_plan.version
                or self.execution_result.dataset_id != self.transformation_plan.dataset_id
            ):
                raise ValueError("execution evidence must match workflow plan")
        expected_qa_status = {
            WorkflowStage.QA_PASSED: QAStatus.PASS,
            WorkflowStage.QA_WARNING: QAStatus.WARN,
            WorkflowStage.QA_FAILED: QAStatus.FAIL,
        }
        if self.stage in expected_qa_status:
            if self.qa_report is None or self.qa_report.status is not expected_qa_status[self.stage]:
                raise ValueError("QA stage must match QA report status")
            if (
                self.qa_report.dataset_id != self.dataset_identity.dataset_id
                or self.transformation_plan is None
                or self.qa_report.plan_id != self.transformation_plan.plan_id
            ):
                raise ValueError("QA report must match workflow dataset and plan")
        return self


__all__ = [
    name
    for name, value in globals().items()
    if (isinstance(value, type) and value.__module__ == __name__)
    or name == "OperationParameters"
]
