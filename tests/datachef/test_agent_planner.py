from __future__ import annotations

import inspect

import pandas as pd
import pytest

from datachef.agents import AgentPlanner, AgentReviewer
from datachef.agents.plan_crew import CrewPlanResult
from datachef.agents.tools import PlanDraft
from datachef.agents.trace import AgentMode, AgentOutcome
from datachef.contracts import ReviewerDecision, ReviewerVerdict, UserIntent
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.planning import Planner, Reviewer, RuleBasedPlanner, validate_plan
from datachef.planning.plan import create_transformation_plan
from datachef.privacy import build_column_alias_map, build_planning_context
from datachef.workflow import prepare_workflow

LIVE_ENV = {
    "DATACHEF_OFFLINE": "false",
    "GOOGLE_API_KEY": "not-a-real-key",
    "GEMINI_MODEL": "gemini-3.1-flash-lite",
}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 2],
            "region": ["N", "S", "S"],
            "amount": [1.0, 2.0, 2.0],
        }
    )


def _context(row_loss: float = 50.0):
    intent = UserIntent(
        intent_id="intent-agent",
        user_goal="Prepare for analysis.",
        selected_key_columns=("order_id",),
        acceptable_row_loss_pct=row_loss,
    )
    report = diagnose_raw_dataframe(_frame(), selected_key_columns=("order_id",))
    alias_map = build_column_alias_map(report, intent)
    return build_planning_context(report, intent, (), column_alias_map=alias_map), intent


def test_agent_planner_satisfies_the_planner_protocol_structurally() -> None:
    assert inspect.signature(AgentPlanner.propose) == inspect.signature(Planner.propose)
    assert inspect.signature(AgentReviewer.review) == inspect.signature(Reviewer.review)
    # Structural conformance only: no inheritance from the deterministic pair.
    assert [cls.__name__ for cls in AgentPlanner.__mro__] == ["AgentPlanner", "object"]
    assert [cls.__name__ for cls in AgentReviewer.__mro__] == ["AgentReviewer", "object"]
    assert not issubclass(AgentPlanner, RuleBasedPlanner)


def test_offline_default_uses_the_deterministic_planner_and_builds_no_crew() -> None:
    context, _ = _context()
    constructed: list[str] = []
    planner = AgentPlanner(environment={"DATACHEF_OFFLINE": "true"})
    planner._runner = lambda ctx: constructed.append("crew")  # type: ignore[assignment]

    plan = planner.propose(context, attempt=1)

    assert constructed == []
    assert planner.trace.mode is AgentMode.OFFLINE
    assert (
        planner.trace.fallback_reason_code
        is AgentOutcome.AGENT_OFFLINE_USING_DETERMINISTIC_PLANNER
    )
    assert validate_plan(context, plan).valid is True


@pytest.mark.parametrize(
    "environment",
    (
        {"DATACHEF_OFFLINE": "true", "GOOGLE_API_KEY": "k", "GEMINI_MODEL": "m"},
        {"DATACHEF_OFFLINE": "false", "GEMINI_MODEL": "m"},
        {"DATACHEF_OFFLINE": "false", "GOOGLE_API_KEY": "k"},
        {"DATACHEF_OFFLINE": "false", "GOOGLE_API_KEY": "k", "GEMINI_MODEL": "<model>"},
        {},
    ),
)
def test_every_missing_live_condition_falls_back_offline(environment: dict) -> None:
    context, _ = _context()
    planner = AgentPlanner(environment=environment)

    plan = planner.propose(context, attempt=1)

    assert planner.trace.mode is AgentMode.OFFLINE
    assert (
        planner.trace.fallback_reason_code
        is AgentOutcome.AGENT_OFFLINE_USING_DETERMINISTIC_PLANNER
    )
    assert validate_plan(context, plan).valid is True


@pytest.mark.parametrize(
    "failure",
    (
        ConnectionError("network down"),
        TimeoutError("deadline exceeded"),
        RuntimeError("429 quota exceeded for key AIzaSyFAKE"),
        ValueError("malformed model output"),
    ),
)
def test_every_live_failure_funnels_to_the_deterministic_planner(failure) -> None:
    context, _ = _context()

    def exploding(_context):
        raise failure

    planner = AgentPlanner(environment=LIVE_ENV)
    planner._runner = exploding  # type: ignore[assignment]

    plan = planner.propose(context, attempt=1)

    assert planner.trace.mode is AgentMode.LIVE
    assert (
        planner.trace.fallback_reason_code
        is AgentOutcome.AGENT_UNAVAILABLE_USING_DETERMINISTIC_PLANNER
    )
    assert validate_plan(context, plan).valid is True
    rendered = planner.trace.model_dump_json()
    assert "network down" not in rendered
    assert "AIzaSyFAKE" not in rendered
    assert "quota" not in rendered


def test_a_forged_identity_from_the_crew_is_rejected_and_falls_back() -> None:
    context, _ = _context()

    def forging(ctx):
        draft = PlanDraft(context=ctx)
        forged = draft.build_plan().model_copy(update={"plan_id": "plan-agent-authored"})
        return CrewPlanResult(plan=forged, draft=draft)

    planner = AgentPlanner(environment=LIVE_ENV)
    planner._runner = forging  # type: ignore[assignment]

    plan = planner.propose(context, attempt=1)

    assert (
        planner.trace.fallback_reason_code
        is AgentOutcome.AGENT_UNAVAILABLE_USING_DETERMINISTIC_PLANNER
    )
    assert plan.plan_id != "plan-agent-authored"
    assert validate_plan(context, plan).valid is True


def test_an_invalid_crew_plan_is_rejected_whole_with_validation_codes() -> None:
    context, _ = _context(row_loss=0.0)

    def destructive(ctx):
        from datachef.agents.tools import DeduplicateByKeysArgs, apply_operation_args

        draft = PlanDraft(context=ctx)
        issue = next(
            item.issue_id
            for item in ctx.diagnostic_report.issues
            if item.kind.value == "DUPLICATE_KEYS"
        )
        apply_operation_args(
            draft,
            "propose_deduplicate_by_keys",
            DeduplicateByKeysArgs(
                keys=["order_id"],
                diagnostic_issue_ids=[issue],
                rationale="r",
                expected_effect="e",
            ),
        )
        return CrewPlanResult(plan=draft.build_plan(), draft=draft)

    planner = AgentPlanner(environment=LIVE_ENV)
    planner._runner = destructive  # type: ignore[assignment]

    plan = planner.propose(context, attempt=1)

    assert (
        planner.trace.fallback_reason_code
        is AgentOutcome.AGENT_CONTRACT_VIOLATION_USING_DETERMINISTIC_PLANNER
    )
    attempt = planner.trace.attempts[-1]
    assert "ROW_LOSS_THRESHOLD" in attempt.validation_codes
    # Never a partial plan: the agent draft is discarded whole and the
    # deterministic proposal is returned instead, identity and all.
    deterministic = RuleBasedPlanner().propose(context, attempt=1)
    assert plan == deterministic


def test_an_accepted_crew_plan_is_returned_and_traced() -> None:
    context, _ = _context(row_loss=50.0)

    def compliant(ctx):
        from datachef.agents.tools import DeduplicateByKeysArgs, apply_operation_args

        draft = PlanDraft(context=ctx)
        issue = next(
            item.issue_id
            for item in ctx.diagnostic_report.issues
            if item.kind.value == "DUPLICATE_KEYS"
        )
        apply_operation_args(
            draft,
            "propose_deduplicate_by_keys",
            DeduplicateByKeysArgs(
                keys=["order_id"],
                diagnostic_issue_ids=[issue],
                rationale="Deterministic duplicate evidence.",
                expected_effect="Keep the first row per key.",
            ),
        )
        return CrewPlanResult(plan=draft.build_plan(), draft=draft)

    planner = AgentPlanner(environment=LIVE_ENV)
    planner._runner = compliant  # type: ignore[assignment]

    plan = planner.propose(context, attempt=1)

    assert validate_plan(context, plan).valid is True
    assert planner.trace.fallback_reason_code is None
    attempt = planner.trace.attempts[-1]
    assert attempt.outcome_code is AgentOutcome.AGENT_PLAN_ACCEPTED
    assert attempt.tool_invocations[0].accepted is True
    assert attempt.elapsed_ms >= 0


def test_the_trace_contains_no_cell_values_or_exception_text() -> None:
    context, _ = _context()

    def exploding(_ctx):
        raise RuntimeError("boom at C:/secret/path.csv while reading 'N' and 2.0")

    planner = AgentPlanner(environment=LIVE_ENV)
    planner._runner = exploding  # type: ignore[assignment]
    planner.propose(context, attempt=1)

    rendered = planner.trace.model_dump_json()

    for leak in ("boom", "C:/secret", "path.csv", "'N'"):
        assert leak not in rendered
    assert "AGENT_UNAVAILABLE_USING_DETERMINISTIC_PLANNER" in rendered


def test_agent_reviewer_falls_back_with_a_typed_code() -> None:
    context, _ = _context()
    draft = PlanDraft(context=context)
    plan = draft.build_plan()
    validation = validate_plan(context, plan)

    reviewer = AgentReviewer(environment=LIVE_ENV)
    reviewer._runner = lambda *_: (_ for _ in ()).throw(ConnectionError("down"))  # type: ignore[assignment]

    verdict = reviewer.review(
        context, plan, validation, previous_feedback=(), attempt=1
    )

    assert isinstance(verdict, ReviewerVerdict)
    assert verdict.plan_id == plan.plan_id
    assert (
        reviewer.trace.fallback_reason_code
        is AgentOutcome.AGENT_UNAVAILABLE_USING_DETERMINISTIC_REVIEWER
    )
    assert "down" not in reviewer.trace.model_dump_json()


def test_agents_drop_into_prepare_workflow_unchanged() -> None:
    _, intent = _context()
    planner = AgentPlanner(environment={})
    reviewer = AgentReviewer(environment={})

    runtime = prepare_workflow(_frame(), intent, planner, reviewer)

    assert runtime.state.stage.value == "AWAITING_APPROVAL"
    assert runtime.state.transformation_plan is not None
    assert planner.trace.mode is AgentMode.OFFLINE
    assert reviewer.trace.mode is AgentMode.OFFLINE
