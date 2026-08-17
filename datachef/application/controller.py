"""Framework-independent coordinator for the offline Phase 1B product flow."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4

import pandas as pd

from datachef.application.models import (
    ApplicationFinding,
    CommandAttempt,
    CommandKind,
    CommandOutcome,
    ParsedDataset,
    RequestedTransformation,
    ScreenId,
    TransitionResult,
    UploadFailure,
    UploadPolicy,
    UploadRequest,
)
from datachef.application.session import (
    ApplicationSession,
    accept_source,
    defensive_runtime_snapshot,
    defensive_session_snapshot,
    navigate,
    new_session,
    record_approval,
    record_command_attempt,
    record_diagnosis,
    record_execution_runtime,
    record_intent,
    record_runtime,
    reset_session,
    set_preview,
)
from datachef.application.uploads import parse_upload, source_metadata_for_upload
from datachef.contracts import (
    CastColumnParameters,
    CastTarget,
    DeduplicateByKeysParameters,
    DiagnosticIssueKind,
    HumanApproval,
    HumanDecision,
    OperationType,
    PIIHandling,
    QAStatus,
    WorkflowState,
    WorkflowStage,
    UserIntent,
)
from datachef.diagnostics import diagnose_raw_dataframe, identify_dataset
from datachef.intent import discover_questions
from datachef.planning import Planner, Reviewer, RuleBasedPlanner, RuleBasedReviewer
from datachef.workflow import (
    WorkflowRuntime,
    execute_workflow,
    prepare_workflow,
    verify_completed_workflow_runtime,
)


UploadParser = Callable[[UploadRequest, UploadPolicy], ParsedDataset | UploadFailure]
PlannerFactory = Callable[[], Planner]
ReviewerFactory = Callable[[], Reviewer]
PrepareService = Callable[[object, UserIntent, Planner, Reviewer], WorkflowRuntime]
ExecuteService = Callable[[WorkflowRuntime, HumanApproval | None], WorkflowRuntime]


_TERMINAL_STAGES = frozenset(
    {
        WorkflowStage.PLAN_REJECTED,
        WorkflowStage.EXECUTION_FAILED,
        WorkflowStage.QA_PASSED,
        WorkflowStage.QA_WARNING,
        WorkflowStage.QA_FAILED,
    }
)


def _finding(
    code: str,
    message: str,
    *,
    blocking: bool,
    request_id: str | None = None,
) -> ApplicationFinding:
    return ApplicationFinding(
        code=code,
        blocking=blocking,
        safe_message=message,
        request_id=request_id,
    )


class DataChefController:
    """Coordinate UI-safe events while keeping Phase 1A authoritative."""

    def __init__(
        self,
        *,
        upload_policy: UploadPolicy | None = None,
        upload_parser: UploadParser = parse_upload,
        planner_factory: PlannerFactory = RuleBasedPlanner,
        reviewer_factory: ReviewerFactory = RuleBasedReviewer,
        prepare_service: PrepareService = prepare_workflow,
        execute_service: ExecuteService = execute_workflow,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        id_factory: Callable[[], str] = lambda: str(uuid4()),
    ) -> None:
        self._session = new_session()
        self._upload_policy = upload_policy or UploadPolicy()
        self._upload_parser = upload_parser
        self._planner_factory = planner_factory
        self._reviewer_factory = reviewer_factory
        self._prepare_service = prepare_service
        self._execute_service = execute_service
        self._clock = clock
        self._id_factory = id_factory

    @property
    def session(self) -> ApplicationSession:
        return defensive_session_snapshot(self._session)

    def _transition(
        self,
        *,
        changed: bool,
        code: str,
        findings: tuple[ApplicationFinding, ...] | None = None,
    ) -> TransitionResult:
        return TransitionResult(
            changed=changed,
            screen=self._session.screen,
            code=code,
            findings=self._session.findings if findings is None else findings,
            revision=self._session.revision,
        )

    def _command_id(self, value: str | None) -> str:
        command_id = value if value is not None else self._id_factory()
        if not command_id or not command_id.strip():
            raise ValueError("command ID must be nonempty")
        return command_id

    def _binding_id(self, *parts: str) -> str:
        digest = sha256()
        for part in parts:
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return f"binding-{digest.hexdigest()}"

    def _plan_binding(self) -> str:
        source = self._session.source
        intent = self._session.intent
        assert source is not None and intent is not None
        requests = "\n".join(
            request.model_dump_json() for request in self._session.requested_transformations
        )
        selected = "\n".join(self._session.selected_question_ids)
        return self._binding_id(
            source.identity.fingerprint,
            intent.model_dump_json(),
            requests,
            selected,
        )

    def _human_binding(self, decision: HumanDecision) -> str:
        runtime = self._session.workflow_runtime
        assert runtime is not None and runtime.state.transformation_plan is not None
        plan = runtime.state.transformation_plan
        return self._binding_id(
            runtime.state.dataset_identity.fingerprint,
            plan.plan_id,
            str(plan.version),
            decision.value,
        )

    def _execution_binding(self) -> str:
        runtime = self._session.workflow_runtime
        approval = self._session.pending_approval
        assert runtime is not None and runtime.state.transformation_plan is not None
        assert approval is not None
        plan = runtime.state.transformation_plan
        return self._binding_id(
            plan.dataset_fingerprint,
            plan.plan_id,
            str(plan.version),
            approval.decision.value,
            "\n".join(approval.approved_operation_ids),
        )

    def _attempt(
        self,
        command_id: str,
        kind: CommandKind,
        binding_id: str,
        outcome: CommandOutcome,
        result_code: str,
    ) -> CommandAttempt:
        return CommandAttempt(
            command_id=command_id,
            kind=kind,
            binding_id=binding_id,
            outcome=outcome,
            result_code=result_code,
        )

    def _is_replayed(
        self,
        attempt: CommandAttempt | None,
        *,
        command_id: str,
        kind: CommandKind,
        binding_id: str,
    ) -> bool:
        return bool(
            attempt is not None
            and attempt.command_id == command_id
            and attempt.kind is kind
            and attempt.binding_id == binding_id
        )

    def _command_attempt(self, command_id: str) -> CommandAttempt | None:
        return next(
            (
                attempt
                for attempt in self._session.command_history
                if attempt.command_id == command_id
            ),
            None,
        )

    def _command_id_conflicts(
        self,
        command_id: str,
    ) -> bool:
        return self._command_attempt(command_id) is not None

    def _frame_matches_source(
        self,
        dataframe: object,
        source: ParsedDataset,
    ) -> bool:
        if not isinstance(dataframe, pd.DataFrame):
            return False
        try:
            return identify_dataset(dataframe) == source.identity
        except Exception:
            return False

    def _state_reconstructs(self, runtime: WorkflowRuntime) -> bool:
        try:
            reconstructed = WorkflowState.model_validate(runtime.state.model_dump())
        except Exception:
            return False
        return reconstructed == runtime.state

    def _prepared_runtime_matches(
        self,
        runtime: object,
        source: ParsedDataset,
        intent: UserIntent,
    ) -> bool:
        if not isinstance(runtime, WorkflowRuntime):
            return False
        if not self._state_reconstructs(runtime):
            return False
        if runtime.state.stage not in {
            WorkflowStage.AWAITING_APPROVAL,
            WorkflowStage.PLAN_REJECTED,
        }:
            return False
        if (
            runtime.state.dataset_identity != source.identity
            or runtime.user_intent != intent
            or not self._frame_matches_source(runtime.raw_dataframe, source)
            or runtime.transformed_dataframe is not None
            or runtime.gold_dataframe is not None
        ):
            return False
        return True

    def _execution_runtime_matches(
        self,
        completed: object,
        prepared: WorkflowRuntime,
        source: ParsedDataset,
        approval: HumanApproval,
    ) -> bool:
        if not isinstance(completed, WorkflowRuntime):
            return False
        if not self._state_reconstructs(completed):
            return False
        state = completed.state
        previous = prepared.state
        plan = previous.transformation_plan
        if plan is None:
            return False
        if state.stage not in {
            WorkflowStage.PLAN_REJECTED,
            WorkflowStage.EXECUTION_FAILED,
            WorkflowStage.QA_PASSED,
            WorkflowStage.QA_WARNING,
            WorkflowStage.QA_FAILED,
        }:
            return False
        if (
            state.dataset_identity != source.identity
            or not self._frame_matches_source(completed.raw_dataframe, source)
            or completed.user_intent != prepared.user_intent
            or state.transformation_plan != plan
            or state.plan_validation != previous.plan_validation
            or state.review_history != previous.review_history
            or state.planning_attempts != previous.planning_attempts
            or state.human_approval != approval
        ):
            return False
        if state.stage is WorkflowStage.PLAN_REJECTED:
            return (
                approval.decision is HumanDecision.REJECT
                and state.accepted_review is None
                and state.execution_result is None
                and state.qa_report is None
                and completed.transformed_dataframe is None
                and completed.gold_dataframe is None
            )
        if (
            approval.decision is not HumanDecision.APPROVE
            or state.accepted_review != previous.accepted_review
            or state.execution_result is None
        ):
            return False
        result = state.execution_result
        if (
            result.dataset_id != source.identity.dataset_id
            or result.source_fingerprint != source.identity.fingerprint
            or result.plan_id != plan.plan_id
            or result.plan_version != plan.version
            or result.accepted_review_attempt != previous.planning_attempts
            or result.before_row_count != source.identity.row_count
            or result.before_column_count != source.identity.column_count
        ):
            return False
        recorded_ids = tuple(record.operation_id for record in result.operation_records)
        planned_ids = tuple(operation.operation_id for operation in plan.operations)
        if state.stage is WorkflowStage.EXECUTION_FAILED:
            return (
                not result.success
                and bool(recorded_ids)
                and recorded_ids == planned_ids[: len(recorded_ids)]
                and completed.transformed_dataframe is None
                and completed.gold_dataframe is None
                and state.qa_report is None
            )
        transformed = completed.transformed_dataframe
        report = state.qa_report
        if (
            not result.success
            or transformed is None
            or report is None
            or recorded_ids != planned_ids
        ):
            return False
        try:
            transformed_identity = identify_dataset(transformed)
        except Exception:
            return False
        if (
            transformed_identity.fingerprint != result.result_fingerprint
            or transformed_identity.row_count != result.after_row_count
            or transformed_identity.column_count != result.after_column_count
            or report.before_row_count != result.before_row_count
            or report.before_column_count != result.before_column_count
            or report.after_row_count != result.after_row_count
            or report.after_column_count != result.after_column_count
        ):
            return False
        if state.stage is WorkflowStage.QA_PASSED:
            if report.status is not QAStatus.PASS or completed.gold_dataframe is None:
                return False
            try:
                gold_identity = identify_dataset(completed.gold_dataframe)
            except Exception:
                return False
            return gold_identity.fingerprint == transformed_identity.fingerprint
        return (
            report.status
            is {
                WorkflowStage.QA_WARNING: QAStatus.WARN,
                WorkflowStage.QA_FAILED: QAStatus.FAIL,
            }[state.stage]
            and completed.gold_dataframe is None
        )

    def load_upload(self, request: UploadRequest) -> TransitionResult:
        expected_metadata = source_metadata_for_upload(request, self._upload_policy)
        if isinstance(expected_metadata, UploadFailure):
            finding = _finding(
                expected_metadata.code.value,
                expected_metadata.safe_message,
                blocking=True,
            )
            return self._transition(
                changed=False,
                code=f"UPLOAD_{expected_metadata.code.value}",
                findings=(finding,),
            )
        if self._session.last_upload_request_id == expected_metadata.request_id:
            return self._transition(changed=False, code="UPLOAD_UNCHANGED")
        try:
            parsed = self._upload_parser(request, self._upload_policy)
        except Exception:
            finding = _finding(
                "PARSER_FAILURE",
                "The selected parser could not read this upload.",
                blocking=True,
            )
            return self._transition(
                changed=False,
                code="UPLOAD_PARSER_FAILURE",
                findings=(finding,),
            )
        if isinstance(parsed, UploadFailure):
            finding = _finding(
                parsed.code.value,
                parsed.safe_message,
                blocking=True,
            )
            return self._transition(
                changed=False,
                code=f"UPLOAD_{parsed.code.value}",
                findings=(finding,),
            )
        if not isinstance(parsed, ParsedDataset):
            finding = _finding(
                "PARSER_FAILURE",
                "The selected parser returned an invalid result.",
                blocking=True,
            )
            return self._transition(
                changed=False,
                code="UPLOAD_PARSER_FAILURE",
                findings=(finding,),
            )
        try:
            identity_matches = identify_dataset(parsed.raw_copy()) == parsed.identity
        except Exception:
            identity_matches = False
        if parsed.metadata != expected_metadata or not identity_matches:
            finding = _finding(
                "PARSER_RESULT_MISMATCH",
                "The parser result does not belong to the current upload request.",
                blocking=True,
            )
            return self._transition(
                changed=False,
                code="UPLOAD_PARSER_RESULT_MISMATCH",
                findings=(finding,),
            )
        self._session, changed = accept_source(self._session, parsed)
        return self._transition(
            changed=changed,
            code="SOURCE_LOADED" if changed else "UPLOAD_UNCHANGED",
        )

    def diagnose(self) -> TransitionResult:
        source = self._session.source
        if source is None:
            return self._transition(changed=False, code="SOURCE_REQUIRED")
        if self._session.display_diagnostic_report is not None:
            return self._transition(changed=False, code="DIAGNOSIS_UNCHANGED")
        try:
            report = diagnose_raw_dataframe(source.raw_copy())
            questions = discover_questions(report, maximum=5)
            self._session = record_diagnosis(self._session, report, questions)
        except Exception:
            finding = _finding(
                "DIAGNOSIS_FAILURE",
                "The deterministic diagnosis could not be completed.",
                blocking=True,
            )
            return self._transition(
                changed=False,
                code="DIAGNOSIS_FAILURE",
                findings=(finding,),
            )
        return self._transition(changed=True, code="DIAGNOSIS_READY")

    def submit_intent(
        self,
        intent: UserIntent,
        requests: tuple[RequestedTransformation, ...],
        *,
        selected_question_ids: tuple[str, ...] = (),
    ) -> TransitionResult:
        if self._session.source is None or self._session.display_diagnostic_report is None:
            return self._transition(changed=False, code="DIAGNOSIS_REQUIRED")
        available_question_ids = {
            item.question_id for item in self._session.suggested_questions
        }
        if not set(selected_question_ids).issubset(available_question_ids):
            finding = _finding(
                "UNKNOWN_QUESTION_SELECTION",
                "One or more selected question suggestions are no longer available.",
                blocking=True,
            )
            return self._transition(
                changed=False,
                code="INTENT_REJECTED",
                findings=(finding,),
            )
        finding_items: list[ApplicationFinding] = []
        if intent.pii_handling in {PIIHandling.MASK, PIIHandling.REMOVE}:
            finding_items.append(
                _finding(
                    "PII_HANDLING_UNSUPPORTED",
                    "PII masking and removal are not executable in the offline Phase 1B slice.",
                    blocking=True,
                )
            )
        if intent.explicit_requested_transformations:
            finding_items.append(
                _finding(
                    "UNTYPED_REQUEST_UNSUPPORTED",
                    "Free-form transformation requests are not executable in the offline slice; use a typed request.",
                    blocking=True,
                )
            )
        findings = tuple(finding_items)
        previous = self._session
        try:
            self._session = record_intent(
                self._session,
                intent,
                requests,
                selected_question_ids=selected_question_ids,
                findings=findings,
            )
        except ValueError:
            finding = _finding(
                "INVALID_INTENT_REQUEST",
                "The submitted intent or transformation requests are inconsistent.",
                blocking=True,
            )
            return self._transition(
                changed=False,
                code="INTENT_REJECTED",
                findings=(finding,),
            )
        return self._transition(
            changed=self._session is not previous,
            code="INTENT_RECORDED",
        )

    def revise_intent(
        self,
        intent: UserIntent,
        requests: tuple[RequestedTransformation, ...],
        *,
        selected_question_ids: tuple[str, ...] = (),
    ) -> TransitionResult:
        return self.submit_intent(
            intent,
            requests,
            selected_question_ids=selected_question_ids,
        )

    def _diagnostic_link_matches(
        self,
        request: RequestedTransformation,
        operation_issue_ids: tuple[str, ...],
        runtime: WorkflowRuntime,
    ) -> bool:
        if request.diagnostic_issue_id is None:
            return True
        if request.diagnostic_issue_id in operation_issue_ids:
            return True
        display_report = self._session.display_diagnostic_report
        context = runtime.state.planning_context
        if display_report is None or context is None:
            return False
        display_issue = next(
            (
                issue
                for issue in display_report.issues
                if issue.issue_id == request.diagnostic_issue_id
            ),
            None,
        )
        if display_issue is None:
            return False
        return any(
            issue.issue_id in operation_issue_ids
            and issue.kind is display_issue.kind
            and issue.affected_columns == request.target_columns
            for issue in context.diagnostic_report.issues
        )

    def _request_is_planned(
        self,
        request: RequestedTransformation,
        runtime: WorkflowRuntime,
    ) -> bool:
        plan = runtime.state.transformation_plan
        if plan is None:
            return False
        for operation in plan.operations:
            if operation.operation_type is not request.operation_type:
                continue
            if operation.target_columns != request.target_columns:
                continue
            if request.operation_type is OperationType.CAST_COLUMN:
                if not (
                    isinstance(operation.parameters, CastColumnParameters)
                    and operation.parameters.target_type is CastTarget.NUMERIC
                    and self._diagnostic_link_matches(
                        request,
                        operation.diagnostic_issue_ids,
                        runtime,
                    )
                ):
                    continue
            elif request.operation_type is OperationType.DEDUPLICATE_BY_KEYS:
                requested = request.parameters
                if not (
                    isinstance(requested, DeduplicateByKeysParameters)
                    and isinstance(operation.parameters, DeduplicateByKeysParameters)
                    and operation.parameters.keys == requested.keys
                    and operation.parameters.keep is requested.keep
                ):
                    continue
            else:
                continue
            return True
        return False

    def _reconcile_requests(
        self,
        runtime: WorkflowRuntime,
    ) -> tuple[ApplicationFinding, ...]:
        findings = list(self._session.findings)
        for request in self._session.requested_transformations:
            if not self._request_is_planned(request, runtime):
                findings.append(
                    _finding(
                        "REQUEST_NOT_PLANNED",
                        "A requested transformation is absent from the canonical plan.",
                        blocking=True,
                        request_id=request.request_id,
                    )
                )
        return tuple(findings)

    def prepare_plan(self, *, command_id: str | None = None) -> TransitionResult:
        resolved_command = self._command_id(command_id)
        source = self._session.source
        intent = self._session.intent
        if source is None or intent is None:
            return self._transition(changed=False, code="INTENT_REQUIRED")
        binding = self._plan_binding()
        existing_attempt = self._command_attempt(resolved_command)
        if self._is_replayed(
            existing_attempt,
            command_id=resolved_command,
            kind=CommandKind.PLAN_PREPARATION,
            binding_id=binding,
        ):
            return self._transition(changed=False, code="PLAN_COMMAND_REPLAYED")
        if self._command_id_conflicts(resolved_command):
            return self._transition(changed=False, code="PLAN_COMMAND_ID_CONFLICT")
        try:
            runtime = self._prepare_service(
                source.raw_copy(),
                intent,
                self._planner_factory(),
                self._reviewer_factory(),
            )
        except Exception:
            attempt = self._attempt(
                resolved_command,
                CommandKind.PLAN_PREPARATION,
                binding,
                CommandOutcome.FAILED,
                "PLANNING_SERVICE_FAILURE",
            )
            self._session = record_command_attempt(self._session, attempt)
            finding = _finding(
                "PLANNING_SERVICE_FAILURE",
                "The deterministic planning workflow could not be completed.",
                blocking=True,
            )
            return self._transition(
                changed=False,
                code="PLANNING_SERVICE_FAILURE",
                findings=(finding,),
            )
        if not self._prepared_runtime_matches(runtime, source, intent):
            attempt = self._attempt(
                resolved_command,
                CommandKind.PLAN_PREPARATION,
                binding,
                CommandOutcome.FAILED,
                "PLANNING_RUNTIME_INVALID",
            )
            self._session = record_command_attempt(self._session, attempt)
            finding = _finding(
                "PLANNING_RUNTIME_INVALID",
                "The planning service returned evidence for a different workflow.",
                blocking=True,
            )
            return self._transition(
                changed=False,
                code="PLANNING_RUNTIME_INVALID",
                findings=(finding,),
            )
        findings = self._reconcile_requests(runtime)
        code = (
            "PLAN_AWAITING_APPROVAL"
            if runtime.state.stage is WorkflowStage.AWAITING_APPROVAL
            else "PLANNING_STOPPED"
        )
        attempt = self._attempt(
            resolved_command,
            CommandKind.PLAN_PREPARATION,
            binding,
            CommandOutcome.SUCCEEDED,
            code,
        )
        self._session = record_runtime(
            self._session,
            runtime,
            findings=findings,
            command_attempt=attempt,
        )
        return self._transition(changed=True, code=code)

    def record_human_decision(
        self,
        decision: HumanDecision,
        *,
        command_id: str | None = None,
    ) -> TransitionResult:
        resolved_command = self._command_id(command_id)
        existing_attempt = self._command_attempt(resolved_command)
        if (
            existing_attempt is not None
            and existing_attempt.kind is not CommandKind.HUMAN_DECISION
        ):
            return self._transition(
                changed=False,
                code="HUMAN_DECISION_COMMAND_ID_CONFLICT",
            )
        runtime = self._session.workflow_runtime
        if existing_attempt is not None:
            if (
                runtime is not None
                and runtime.state.transformation_plan is not None
                and self._is_replayed(
                    existing_attempt,
                    command_id=resolved_command,
                    kind=CommandKind.HUMAN_DECISION,
                    binding_id=self._human_binding(decision),
                )
            ):
                return self._transition(
                    changed=False,
                    code="HUMAN_DECISION_REPLAYED",
                )
            return self._transition(
                changed=False,
                code="HUMAN_DECISION_COMMAND_ID_CONFLICT",
            )
        if runtime is None or runtime.state.stage is not WorkflowStage.AWAITING_APPROVAL:
            return self._transition(changed=False, code="PLAN_NOT_APPROVABLE")
        plan = runtime.state.transformation_plan
        accepted = runtime.state.accepted_review
        if plan is None or accepted is None:
            return self._transition(changed=False, code="PLAN_NOT_APPROVABLE")
        binding = self._human_binding(decision)
        if decision is HumanDecision.APPROVE and any(
            finding.blocking for finding in self._session.findings
        ):
            attempt = self._attempt(
                resolved_command,
                CommandKind.HUMAN_DECISION,
                binding,
                CommandOutcome.FAILED,
                "APPROVAL_BLOCKED",
            )
            self._session = record_command_attempt(self._session, attempt)
            return self._transition(changed=False, code="APPROVAL_BLOCKED")
        try:
            accepted.require_matching_final_verdict(
                runtime.state.review_history,
                current_plan_id=plan.plan_id,
                current_attempt=runtime.state.planning_attempts,
            )
            decided_at = self._clock()
            if decided_at.tzinfo is None or decided_at.utcoffset() is None:
                raise ValueError("clock must return a timezone-aware datetime")
            operation_ids = tuple(
                operation.operation_id for operation in plan.operations
            )
            approval = HumanApproval(
                dataset_id=plan.dataset_id,
                dataset_fingerprint=plan.dataset_fingerprint,
                plan_id=plan.plan_id,
                plan_version=plan.version,
                decision=decision,
                approved_operation_ids=(
                    operation_ids if decision is HumanDecision.APPROVE else ()
                ),
                decided_at=decided_at,
            )
            self._session = record_approval(
                self._session,
                approval,
                command_attempt=self._attempt(
                    resolved_command,
                    CommandKind.HUMAN_DECISION,
                    binding,
                    CommandOutcome.SUCCEEDED,
                    "HUMAN_DECISION_RECORDED",
                ),
            )
        except (ValueError, TypeError):
            attempt = self._attempt(
                resolved_command,
                CommandKind.HUMAN_DECISION,
                binding,
                CommandOutcome.FAILED,
                "PLAN_NOT_APPROVABLE",
            )
            self._session = record_command_attempt(self._session, attempt)
            return self._transition(changed=False, code="PLAN_NOT_APPROVABLE")
        return self._transition(changed=True, code="HUMAN_DECISION_RECORDED")

    def execute_current_plan(
        self,
        *,
        command_id: str | None = None,
    ) -> TransitionResult:
        resolved_command = self._command_id(command_id)
        existing_attempt = self._command_attempt(resolved_command)
        if (
            existing_attempt is not None
            and existing_attempt.kind is not CommandKind.EXECUTION
        ):
            return self._transition(
                changed=False,
                code="EXECUTION_COMMAND_ID_CONFLICT",
            )
        runtime = self._session.workflow_runtime
        source = self._session.source
        approval = self._session.pending_approval
        if runtime is None or source is None or approval is None:
            return self._transition(changed=False, code="APPROVAL_REQUIRED")
        binding = self._execution_binding()
        if self._is_replayed(
            existing_attempt,
            command_id=resolved_command,
            kind=CommandKind.EXECUTION,
            binding_id=binding,
        ):
            return self._transition(changed=False, code="EXECUTION_COMMAND_REPLAYED")
        if self._command_id_conflicts(resolved_command):
            return self._transition(
                changed=False,
                code="EXECUTION_COMMAND_ID_CONFLICT",
            )
        if runtime.state.stage in _TERMINAL_STAGES:
            return self._transition(changed=False, code="WORKFLOW_TERMINAL")
        fresh_runtime = WorkflowRuntime(
            state=runtime.state,
            raw_dataframe=source.raw_copy(),
            transformed_dataframe=runtime.transformed_dataframe,
            gold_dataframe=runtime.gold_dataframe,
            user_intent=runtime.user_intent,
            column_alias_map=runtime.column_alias_map,
        )
        if approval.decision is HumanDecision.REJECT:
            try:
                rejected = execute_workflow(fresh_runtime, approval)
            except Exception:
                attempt = self._attempt(
                    resolved_command,
                    CommandKind.EXECUTION,
                    binding,
                    CommandOutcome.FAILED,
                    "HUMAN_REJECTION_FAILURE",
                )
                self._session = record_command_attempt(self._session, attempt)
                return self._transition(
                    changed=False,
                    code="HUMAN_REJECTION_FAILURE",
                )
            if (
                rejected is fresh_runtime
                or not self._execution_runtime_matches(
                    rejected,
                    runtime,
                    source,
                    approval,
                )
                or rejected.state.stage is not WorkflowStage.PLAN_REJECTED
            ):
                attempt = self._attempt(
                    resolved_command,
                    CommandKind.EXECUTION,
                    binding,
                    CommandOutcome.FAILED,
                    "HUMAN_REJECTION_EVIDENCE_INVALID",
                )
                self._session = record_command_attempt(self._session, attempt)
                return self._transition(
                    changed=False,
                    code="HUMAN_REJECTION_EVIDENCE_INVALID",
                )
            rejected_snapshot = defensive_runtime_snapshot(rejected)
            assert rejected_snapshot is not None
            self._session = record_execution_runtime(
                self._session,
                rejected_snapshot,
                command_attempt=self._attempt(
                    resolved_command,
                    CommandKind.EXECUTION,
                    binding,
                    CommandOutcome.SUCCEEDED,
                    "EXECUTION_COMPLETED",
                ),
            )
            return self._transition(changed=True, code="EXECUTION_COMPLETED")
        try:
            completed = self._execute_service(fresh_runtime, approval)
        except Exception:
            attempt = self._attempt(
                resolved_command,
                CommandKind.EXECUTION,
                binding,
                CommandOutcome.FAILED,
                "EXECUTION_SERVICE_FAILURE",
            )
            self._session = record_command_attempt(self._session, attempt)
            finding = _finding(
                "EXECUTION_SERVICE_FAILURE",
                "The approved deterministic workflow could not be executed.",
                blocking=True,
            )
            return self._transition(
                changed=False,
                code="EXECUTION_SERVICE_FAILURE",
                findings=(finding,),
            )
        if completed is fresh_runtime:
            attempt = self._attempt(
                resolved_command,
                CommandKind.EXECUTION,
                binding,
                CommandOutcome.FAILED,
                "EXECUTION_REFUSED",
            )
            self._session = record_command_attempt(self._session, attempt)
            return self._transition(changed=False, code="EXECUTION_REFUSED")
        verified = None
        if self._execution_runtime_matches(completed, runtime, source, approval):
            verified = verify_completed_workflow_runtime(runtime, completed)
        if verified is None:
            attempt = self._attempt(
                resolved_command,
                CommandKind.EXECUTION,
                binding,
                CommandOutcome.FAILED,
                "EXECUTION_EVIDENCE_INVALID",
            )
            self._session = record_command_attempt(self._session, attempt)
            return self._transition(changed=False, code="EXECUTION_EVIDENCE_INVALID")
        self._session = record_execution_runtime(
            self._session,
            verified,
            command_attempt=self._attempt(
                resolved_command,
                CommandKind.EXECUTION,
                binding,
                CommandOutcome.SUCCEEDED,
                "EXECUTION_COMPLETED",
            ),
        )
        return self._transition(changed=True, code="EXECUTION_COMPLETED")

    def set_preview_enabled(self, enabled: bool) -> TransitionResult:
        previous = self._session
        self._session = set_preview(self._session, enabled)
        return self._transition(
            changed=self._session is not previous,
            code="PREVIEW_UPDATED" if self._session is not previous else "PREVIEW_UNCHANGED",
        )

    def reset(self) -> TransitionResult:
        self._session = reset_session(self._session)
        return self._transition(changed=True, code="SESSION_RESET")


__all__ = ["DataChefController"]
