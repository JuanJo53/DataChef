"""Thin CrewAI Flow adapter over the framework-independent workflow service."""

from __future__ import annotations

from contextlib import contextmanager
import os
import tempfile
from typing import Iterator

from datachef.contracts import (
    HumanApproval,
    QualityInvariant,
    UserIntent,
    WorkflowStage,
    WorkflowState,
)
from datachef.planning import Planner, Reviewer
from datachef.workflow.service import WorkflowRuntime, execute_workflow, prepare_workflow


@contextmanager
def isolated_crewai_runtime() -> Iterator[None]:
    """Keep CrewAI storage temporary and disable offline-test telemetry."""

    names = (
        "CREWAI_STORAGE_DIR",
        "LOCALAPPDATA",
        "CREWAI_DISABLE_TELEMETRY",
        "CREWAI_DISABLE_TRACKING",
        "OTEL_SDK_DISABLED",
    )
    previous = {name: os.environ.get(name) for name in names}
    with tempfile.TemporaryDirectory(prefix="datachef-phase1a-") as storage:
        os.environ["CREWAI_STORAGE_DIR"] = storage
        os.environ["LOCALAPPDATA"] = storage
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
        os.environ["CREWAI_DISABLE_TRACKING"] = "true"
        os.environ["OTEL_SDK_DISABLED"] = "true"
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _create_phase1a_flow(
    runtime: WorkflowRuntime,
    intent: UserIntent,
    planner: Planner,
    reviewer: Reviewer,
    *,
    approval: HumanApproval | None = None,
    user_invariants: tuple[QualityInvariant, ...] = (),
):
    """Internal primitive; caller must establish an isolated CrewAI runtime."""

    from crewai.flow.flow import Flow, listen, router, start
    from pydantic import Field

    class DataChefPhase1AFlow(Flow[WorkflowState]):
        initial_state: WorkflowState = runtime.state
        workflow_runtime: WorkflowRuntime = Field(exclude=True)
        intent_input: UserIntent = Field(exclude=True)
        planner_boundary: object = Field(exclude=True)
        reviewer_boundary: object = Field(exclude=True)
        approval_input: HumanApproval | None = Field(default=None, exclude=True)
        invariant_input: tuple[QualityInvariant, ...] = Field(
            default_factory=tuple,
            exclude=True,
        )

        def _sync(self, updated: WorkflowRuntime) -> None:
            object.__setattr__(self, "workflow_runtime", updated)
            object.__setattr__(self, "_state", updated.state)

        @start()
        def enter(self) -> str:
            return self.state.stage.value

        @router(enter, emit=("BEGIN", "WAIT", "EXECUTE", "TERMINAL"))
        def route_entry(self) -> str:
            if self.state.stage is WorkflowStage.INITIAL:
                return "BEGIN"
            if self.state.stage is WorkflowStage.AWAITING_APPROVAL:
                return "EXECUTE" if self.approval_input is not None else "WAIT"
            return "TERMINAL"

        @listen("BEGIN")
        def prepare(self):
            updated = prepare_workflow(
                self.workflow_runtime.raw_dataframe,
                self.intent_input,
                self.planner_boundary,  # type: ignore[arg-type]
                self.reviewer_boundary,  # type: ignore[arg-type]
            )
            self._sync(updated)
            return updated.state

        @router(prepare, emit=("AWAITING_APPROVAL", "PLAN_REJECTED"))
        def route_prepared(self) -> str:
            return self.state.stage.value

        @listen("EXECUTE")
        def execute(self):
            updated = execute_workflow(
                self.workflow_runtime,
                self.approval_input,
                user_invariants=self.invariant_input,
            )
            self._sync(updated)
            return updated.state

        @router(
            execute,
            emit=(
                "AWAITING_APPROVAL",
                "PLAN_REJECTED",
                "EXECUTION_FAILED",
                "QA_PASSED",
                "QA_WARNING",
                "QA_FAILED",
            ),
        )
        def route_executed(self) -> str:
            return self.state.stage.value

    return DataChefPhase1AFlow(
        initial_state=runtime.state,
        workflow_runtime=runtime,
        intent_input=intent,
        planner_boundary=planner,
        reviewer_boundary=reviewer,
        approval_input=approval,
        invariant_input=user_invariants,
        tracing=False,
        memory=None,
        suppress_flow_events=True,
    )


def run_phase1a_flow(
    runtime: WorkflowRuntime,
    intent: UserIntent,
    planner: Planner,
    reviewer: Reviewer,
    *,
    approval: HumanApproval | None = None,
    user_invariants: tuple[QualityInvariant, ...] = (),
) -> WorkflowRuntime:
    """Run one Flow with cleanup; Phase 1A supports single-process/single-flight use."""

    with isolated_crewai_runtime():
        from crewai.events.event_bus import crewai_event_bus

        try:
            flow = _create_phase1a_flow(
                runtime,
                intent,
                planner,
                reviewer,
                approval=approval,
                user_invariants=user_invariants,
            )
            flow.kickoff()
        except BaseException as primary_error:
            try:
                crewai_event_bus.shutdown(wait=True)
            except Exception as cleanup_error:
                primary_error.add_note(
                    "CrewAI event-bus cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            raise
        crewai_event_bus.shutdown(wait=True)
        return flow.workflow_runtime


__all__ = ["run_phase1a_flow"]
