"""Immutable runtime session state and pure Phase 1B invalidation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace

from datachef.application.models import (
    ApplicationFinding,
    CommandAttempt,
    CommandKind,
    ParsedDataset,
    RequestedTransformation,
    ScreenId,
)
from datachef.contracts import (
    DiagnosticReport,
    HumanApproval,
    SuggestedQuestion,
    UserIntent,
    WorkflowStage,
)
from datachef.workflow import WorkflowRuntime


@dataclass(frozen=True, slots=True, repr=False)
class ApplicationSession:
    """Framework-independent runtime references for one local product session."""

    revision: int = 0
    uploader_generation: int = 0
    screen: ScreenId = ScreenId.UPLOAD
    source: ParsedDataset | None = None
    display_diagnostic_report: DiagnosticReport | None = None
    intent: UserIntent | None = None
    requested_transformations: tuple[RequestedTransformation, ...] = ()
    keep_only_columns: tuple[str, ...] = ()
    suggested_questions: tuple[SuggestedQuestion, ...] = ()
    selected_question_ids: tuple[str, ...] = ()
    workflow_runtime: WorkflowRuntime | None = None
    findings: tuple[ApplicationFinding, ...] = ()
    pending_approval: HumanApproval | None = None
    preview_enabled: bool = False
    last_upload_request_id: str | None = None
    plan_command_attempt: CommandAttempt | None = None
    human_decision_command_attempt: CommandAttempt | None = None
    execution_command_attempt: CommandAttempt | None = None
    command_history: tuple[CommandAttempt, ...] = ()

    def __post_init__(self) -> None:
        if self.revision < 0 or self.uploader_generation < 0:
            raise ValueError("session counters cannot be negative")
        request_ids = tuple(item.request_id for item in self.requested_transformations)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("requested transformation request IDs must be unique")
        semantic_requests = tuple(
            item.model_dump_json(exclude={"request_id"})
            for item in self.requested_transformations
        )
        if len(set(semantic_requests)) != len(semantic_requests):
            raise ValueError("requested transformations must be semantically unique")
        if len(set(self.keep_only_columns)) != len(self.keep_only_columns):
            raise ValueError("keep-only columns must be unique")
        if any(not column.strip() for column in self.keep_only_columns):
            raise ValueError("keep-only columns must contain non-whitespace text")
        if len(set(self.selected_question_ids)) != len(self.selected_question_ids):
            raise ValueError("selected question IDs must be unique")
        if any(not isinstance(item, CommandAttempt) for item in self.command_history):
            raise TypeError("command history must contain typed command attempts")
        command_ids = tuple(item.command_id for item in self.command_history)
        if len(set(command_ids)) != len(command_ids):
            raise ValueError("command history IDs must be unique")
        for latest in (
            self.plan_command_attempt,
            self.human_decision_command_attempt,
            self.execution_command_attempt,
        ):
            if latest is not None and latest not in self.command_history:
                raise ValueError("latest command attempt must exist in command history")

    def __getstate__(self) -> object:
        raise TypeError("application session is runtime-only and cannot be serialized")

    def __repr__(self) -> str:
        workflow_stage = (
            self.workflow_runtime.state.stage.value
            if self.workflow_runtime is not None
            else None
        )
        return (
            "ApplicationSession("
            f"revision={self.revision}, "
            f"uploader_generation={self.uploader_generation}, "
            f"screen={self.screen.value!r}, "
            f"has_source={self.source is not None}, "
            f"has_diagnosis={self.display_diagnostic_report is not None}, "
            f"has_intent={self.intent is not None}, "
            f"workflow_stage={workflow_stage!r}, "
            f"finding_count={len(self.findings)}, "
            f"command_attempt_count={len(self.command_history)}, "
            f"preview_enabled={self.preview_enabled})"
        )

    @property
    def last_plan_command_id(self) -> str | None:
        return self.plan_command_attempt.command_id if self.plan_command_attempt else None

    @property
    def last_human_decision_command_id(self) -> str | None:
        attempt = self.human_decision_command_attempt
        return attempt.command_id if attempt else None

    @property
    def last_execution_command_id(self) -> str | None:
        return self.execution_command_attempt.command_id if self.execution_command_attempt else None


def new_session(*, uploader_generation: int = 0) -> ApplicationSession:
    return ApplicationSession(uploader_generation=uploader_generation)


def _changed(session: ApplicationSession, **updates: object) -> ApplicationSession:
    return replace(session, revision=session.revision + 1, **updates)


def accept_source(
    session: ApplicationSession,
    source: ParsedDataset,
) -> tuple[ApplicationSession, bool]:
    """Accept a new source or return the exact session for the same upload request.

    Accepting a file does not move the user off Upload; running the diagnosis
    does. Uploading a second file therefore rewinds presentation to the start
    of the flow, which matches the evidence being cleared alongside it.
    """

    if session.last_upload_request_id == source.metadata.request_id:
        return session, False
    return (
        _changed(
            session,
            screen=ScreenId.UPLOAD,
            source=source,
            display_diagnostic_report=None,
            intent=None,
            requested_transformations=(),
            keep_only_columns=(),
            suggested_questions=(),
            selected_question_ids=(),
            workflow_runtime=None,
            findings=(),
            pending_approval=None,
            preview_enabled=False,
            last_upload_request_id=source.metadata.request_id,
            plan_command_attempt=None,
            human_decision_command_attempt=None,
            execution_command_attempt=None,
            command_history=(),
        ),
        True,
    )


def record_diagnosis(
    session: ApplicationSession,
    report: DiagnosticReport,
    questions: tuple[SuggestedQuestion, ...],
) -> ApplicationSession:
    if session.source is None:
        raise ValueError("diagnosis requires an accepted source")
    if report.dataset_identity != session.source.identity:
        raise ValueError("diagnosis must match the current source")
    if (
        session.display_diagnostic_report == report
        and session.suggested_questions == questions
    ):
        return session
    return _changed(
        session,
        screen=ScreenId.DIAGNOSE,
        display_diagnostic_report=report,
        suggested_questions=questions,
    )


def record_intent(
    session: ApplicationSession,
    intent: UserIntent,
    requests: tuple[RequestedTransformation, ...],
    *,
    selected_question_ids: tuple[str, ...] = (),
    findings: tuple[ApplicationFinding, ...] = (),
    keep_only_columns: tuple[str, ...] = (),
) -> ApplicationSession:
    request_ids = tuple(request.request_id for request in requests)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("requested transformation request IDs must be unique")
    if len(set(selected_question_ids)) != len(selected_question_ids):
        raise ValueError("selected question IDs must be unique")
    if (
        session.intent == intent
        and session.requested_transformations == requests
        and session.keep_only_columns == keep_only_columns
        and session.selected_question_ids == selected_question_ids
        and session.findings == findings
    ):
        return session
    return _changed(
        session,
        screen=ScreenId.PLAN,
        intent=intent,
        requested_transformations=requests,
        keep_only_columns=keep_only_columns,
        selected_question_ids=selected_question_ids,
        workflow_runtime=None,
        findings=findings,
        pending_approval=None,
        plan_command_attempt=None,
        human_decision_command_attempt=None,
        execution_command_attempt=None,
        command_history=(),
    )


def record_command_attempt(
    session: ApplicationSession,
    attempt: CommandAttempt,
) -> ApplicationSession:
    field = {
        CommandKind.PLAN_PREPARATION: "plan_command_attempt",
        CommandKind.HUMAN_DECISION: "human_decision_command_attempt",
        CommandKind.EXECUTION: "execution_command_attempt",
    }[attempt.kind]
    existing = next(
        (
            item
            for item in session.command_history
            if item.command_id == attempt.command_id
        ),
        None,
    )
    if existing == attempt:
        return session
    if existing is not None:
        raise ValueError("command ID is already bound to another attempt")
    return _changed(
        session,
        **{
            field: attempt,
            "command_history": (*session.command_history, attempt),
        },
    )


def screen_for_workflow_stage(stage: WorkflowStage) -> ScreenId:
    """Map Phase 1A evidence to presentation without authorizing transitions.

    Every stage from execution onwards lands on Results. Quality assurance is
    still the mandatory internal gate — a run that does not pass it withholds
    gold, downloads, and the dashboard exactly as before — but it is no longer
    a place the user is sent, so Results is where both verdicts are read.
    """

    return {
        WorkflowStage.INITIAL: ScreenId.UPLOAD,
        WorkflowStage.DIAGNOSED: ScreenId.DIAGNOSE,
        WorkflowStage.INTENT_CAPTURED: ScreenId.INTENT,
        WorkflowStage.CONTEXT_READY: ScreenId.PLAN,
        WorkflowStage.PLANNING: ScreenId.PLAN,
        WorkflowStage.PLAN_REJECTED: ScreenId.PLAN,
        WorkflowStage.AWAITING_APPROVAL: ScreenId.APPROVAL,
        WorkflowStage.EXECUTING: ScreenId.RESULTS,
        WorkflowStage.EXECUTION_FAILED: ScreenId.RESULTS,
        WorkflowStage.QA_PASSED: ScreenId.RESULTS,
        WorkflowStage.QA_WARNING: ScreenId.RESULTS,
        WorkflowStage.QA_FAILED: ScreenId.RESULTS,
    }[stage]


def furthest_screen_for_workflow_stage(stage: WorkflowStage) -> ScreenId:
    """Furthest screen this evidence unlocks, which is not where it lands.

    Execution lands on Results whatever the verdict, but only a passing run
    earns the dashboard. Keeping that rule here rather than in the shell means
    the sidebar cannot offer a dashboard the controller would refuse to build.
    """

    if stage is WorkflowStage.QA_PASSED:
        return ScreenId.DASHBOARD
    return screen_for_workflow_stage(stage)


def record_runtime(
    session: ApplicationSession,
    runtime: WorkflowRuntime,
    *,
    findings: tuple[ApplicationFinding, ...],
    command_attempt: CommandAttempt,
) -> ApplicationSession:
    if command_attempt.kind is not CommandKind.PLAN_PREPARATION:
        raise ValueError("planning runtime requires a plan command attempt")
    if session.source is None or session.intent is None:
        raise ValueError("planning runtime requires source and intent")
    if runtime.state.dataset_identity != session.source.identity:
        raise ValueError("planning runtime must match the current source")
    screen = screen_for_workflow_stage(runtime.state.stage)
    retained_history = tuple(
        attempt
        for attempt in session.command_history
        if attempt.kind is CommandKind.PLAN_PREPARATION
    )
    existing = next(
        (
            attempt
            for attempt in retained_history
            if attempt.command_id == command_attempt.command_id
        ),
        None,
    )
    if existing is not None and existing != command_attempt:
        raise ValueError("command ID is already bound to another attempt")
    if existing is None:
        retained_history = (*retained_history, command_attempt)
    return _changed(
        session,
        screen=screen,
        workflow_runtime=runtime,
        findings=findings,
        pending_approval=None,
        plan_command_attempt=command_attempt,
        human_decision_command_attempt=None,
        execution_command_attempt=None,
        command_history=retained_history,
    )


def record_approval(
    session: ApplicationSession,
    approval: HumanApproval,
    *,
    command_attempt: CommandAttempt,
) -> ApplicationSession:
    if command_attempt.kind is not CommandKind.HUMAN_DECISION:
        raise ValueError("approval requires a human-decision command attempt")
    existing = next(
        (
            item
            for item in session.command_history
            if item.command_id == command_attempt.command_id
        ),
        None,
    )
    if existing is not None and existing != command_attempt:
        raise ValueError("command ID is already bound to another attempt")
    history = (
        session.command_history
        if existing is not None
        else (*session.command_history, command_attempt)
    )
    return _changed(
        session,
        screen=ScreenId.APPROVAL,
        pending_approval=approval,
        human_decision_command_attempt=command_attempt,
        execution_command_attempt=None,
        command_history=history,
    )


def record_execution_runtime(
    session: ApplicationSession,
    runtime: WorkflowRuntime,
    *,
    command_attempt: CommandAttempt,
) -> ApplicationSession:
    if command_attempt.kind is not CommandKind.EXECUTION:
        raise ValueError("execution runtime requires an execution command attempt")
    screen = screen_for_workflow_stage(runtime.state.stage)
    existing = next(
        (
            item
            for item in session.command_history
            if item.command_id == command_attempt.command_id
        ),
        None,
    )
    if existing is not None and existing != command_attempt:
        raise ValueError("command ID is already bound to another attempt")
    history = (
        session.command_history
        if existing is not None
        else (*session.command_history, command_attempt)
    )
    return _changed(
        session,
        screen=screen,
        workflow_runtime=runtime,
        execution_command_attempt=command_attempt,
        command_history=history,
    )


def defensive_runtime_snapshot(runtime: WorkflowRuntime | None) -> WorkflowRuntime | None:
    if runtime is None:
        return None
    return WorkflowRuntime(
        state=runtime.state,
        raw_dataframe=runtime.raw_dataframe.copy(deep=True),
        transformed_dataframe=(
            runtime.transformed_dataframe.copy(deep=True)
            if runtime.transformed_dataframe is not None
            else None
        ),
        gold_dataframe=(
            runtime.gold_dataframe.copy(deep=True)
            if runtime.gold_dataframe is not None
            else None
        ),
        user_intent=runtime.user_intent,
        column_alias_map=runtime.column_alias_map,
    )


def defensive_session_snapshot(session: ApplicationSession) -> ApplicationSession:
    """Return a public view whose mutable DataFrames cannot alias controller state."""

    return replace(
        session,
        workflow_runtime=defensive_runtime_snapshot(session.workflow_runtime),
    )


def navigate(session: ApplicationSession, screen: ScreenId) -> ApplicationSession:
    if session.screen is screen:
        return session
    return _changed(session, screen=screen)


def set_preview(
    session: ApplicationSession,
    enabled: bool,
) -> ApplicationSession:
    if session.preview_enabled is enabled:
        return session
    return _changed(session, preview_enabled=enabled)


def reset_session(session: ApplicationSession) -> ApplicationSession:
    return new_session(uploader_generation=session.uploader_generation + 1)


__all__ = [
    "ApplicationSession",
    "accept_source",
    "navigate",
    "new_session",
    "record_approval",
    "record_command_attempt",
    "record_diagnosis",
    "record_execution_runtime",
    "record_intent",
    "record_runtime",
    "reset_session",
    "set_preview",
    "defensive_session_snapshot",
    "defensive_runtime_snapshot",
    "furthest_screen_for_workflow_stage",
    "screen_for_workflow_stage",
]
