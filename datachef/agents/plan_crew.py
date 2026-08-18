"""CrewAI wiring for the planning crew.

Mirrors ``datachef/workflow/crewai_flow.py``: ``crewai`` is imported inside
functions so importing this module never loads the framework, the runtime is
isolated through the existing ``isolated_crewai_runtime`` context manager, and
the event bus is shut down on both the success and the failure path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from datachef.agents.tools import (
    FinalizePlanArgs,
    NoArgs,
    PlanDraft,
    apply_operation_args,
    build_operation_specs,
    discard_last_operation,
    estimate_current_plan,
    finalize_plan,
    inspect_profile,
)
from datachef.contracts import PlanningContext, TransformationPlan
from datachef.planning import validate_plan


class PlanEnvelope(BaseModel):
    """The only thing the agent returns: a summary. The plan lives server-side."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class CrewPlanResult:
    plan: TransformationPlan
    draft: PlanDraft


def _make_operation_tool(base_tool_cls, draft: PlanDraft, tool_name, operation_type, args_model):
    tool_description = (
        f"Propose a {operation_type.value} operation. Target columns must be "
        "names returned by inspect_profile. Returns {'accepted': bool} and, when "
        "refused, a reason_code."
    )

    class _OperationTool(base_tool_cls):  # type: ignore[misc, valid-type]
        model_config = ConfigDict(arbitrary_types_allowed=True)
        name: str = tool_name
        description: str = tool_description
        args_schema: type[BaseModel] = args_model

        def _run(self, **kwargs: Any) -> dict[str, Any]:
            return apply_operation_args(draft, tool_name, args_model(**kwargs))

    return _OperationTool()


def build_crew_tools(draft: PlanDraft) -> list[Any]:
    """Build one tool per executable operation, plus the non-proposal tools."""

    from crewai.tools import BaseTool

    tools: list[Any] = [
        _make_operation_tool(BaseTool, draft, name, operation_type, args_model)
        for name, operation_type, args_model in build_operation_specs()
    ]

    class InspectProfileTool(BaseTool):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        name: str = "inspect_profile"
        description: str = (
            "Return the column schema, dtypes, diagnostic issues and severities. "
            "Never returns cell values."
        )
        args_schema: type[BaseModel] = NoArgs

        def _run(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return inspect_profile(draft)

    class EstimateTool(BaseTool):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        name: str = "estimate_current_plan"
        description: str = (
            "Run the deterministic validator over the operations proposed so far "
            "and return findings and estimated row loss. Use this before "
            "finalizing."
        )
        args_schema: type[BaseModel] = NoArgs

        def _run(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return estimate_current_plan(draft)

    class DiscardTool(BaseTool):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        name: str = "discard_last_operation"
        description: str = (
            "Retract the most recently proposed operation, for example after "
            "estimate_current_plan reports it removes too many rows."
        )
        args_schema: type[BaseModel] = NoArgs

        def _run(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return discard_last_operation(draft)

    class FinalizeTool(BaseTool):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        name: str = "finalize_plan"
        description: str = (
            "Validate the accumulated draft and return a handle, or a typed "
            "refusal listing the reason codes."
        )
        args_schema: type[BaseModel] = FinalizePlanArgs

        def _run(self, **kwargs: Any) -> dict[str, Any]:
            return finalize_plan(draft, FinalizePlanArgs(**kwargs).summary)

    tools.extend([InspectProfileTool(), EstimateTool(), DiscardTool(), FinalizeTool()])
    return tools


def build_plan_guardrail(draft: PlanDraft):
    """Task guardrail: reject on any validation finding, with codes only."""

    # No return annotation: CrewAI inspects it, and `from __future__ import
    # annotations` would hand it an unresolvable string.
    def guardrail(output):
        validation = validate_plan(draft.context, draft.build_plan())
        if validation.valid:
            return True, output
        return False, ", ".join(finding.code for finding in validation.findings)

    return guardrail


def build_planning_crew(context: PlanningContext, draft: PlanDraft, llm: Any) -> Any:
    """Construct the sequential planning crew. Caller owns runtime isolation."""

    from crewai import Agent, Crew, Process, Task

    tools = build_crew_tools(draft)
    planner = Agent(
        role="Data preparation planner",
        goal=(
            "Propose the smallest set of allow-listed operations that resolves "
            "the reported diagnostic issues without exceeding the approved row "
            "loss."
        ),
        backstory=(
            "You work only through the provided tools. You never see raw values, "
            "only the column schema and diagnostic codes."
        ),
        tools=tools,
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description=(
            "Call inspect_profile first. Propose operations only for reported "
            "issues. Call estimate_current_plan before finishing; if it reports "
            "findings, retract the offending operation and try a smaller change. "
            "Finish by calling finalize_plan with a one-line summary."
        ),
        expected_output="A one-line summary of the plan you finalized.",
        agent=planner,
        output_pydantic=PlanEnvelope,
        guardrail=build_plan_guardrail(draft),
        guardrail_max_retries=2,
    )
    return Crew(
        agents=[planner],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
        tracing=False,
    )


def run_planning_crew(context: PlanningContext, llm: Any) -> CrewPlanResult:
    """Run one crew inside an isolated runtime and return the server-side plan."""

    from crewai.events.event_bus import crewai_event_bus

    from datachef.workflow.crewai_flow import isolated_crewai_runtime

    draft = PlanDraft(context=context)
    with isolated_crewai_runtime():
        try:
            crew = build_planning_crew(context, draft, llm)
            output = crew.kickoff()
            if isinstance(output, object) and hasattr(output, "pydantic"):
                envelope = getattr(output, "pydantic", None)
                if isinstance(envelope, PlanEnvelope):
                    draft.summary = envelope.summary
        finally:
            crewai_event_bus.shutdown(wait=True)
    return CrewPlanResult(plan=draft.build_plan(), draft=draft)


__all__ = [
    "CrewPlanResult",
    "PlanEnvelope",
    "build_crew_tools",
    "build_plan_guardrail",
    "build_planning_crew",
    "run_planning_crew",
]
