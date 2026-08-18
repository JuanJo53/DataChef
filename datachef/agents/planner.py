"""Agent planner and reviewer that fall back to the deterministic pair.

Both satisfy the existing ``Planner`` and ``Reviewer`` Protocols structurally, so
they drop into the controller's ``planner_factory`` / ``reviewer_factory`` with
no change anywhere in ``datachef/application``.

Every failure mode — offline, missing credential, network, timeout, quota,
malformed output, contract violation, crew exception — funnels through one
handler to the deterministic implementation and is recorded as a typed code. A
single broad ``except`` is deliberate: every one of those failures has the same
safe answer, and none of them may surface exception text to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from datachef.agents.llm import decide_live_model
from datachef.agents.trace import (
    AgentMode,
    AgentOutcome,
    AgentTrace,
    AttemptTrace,
)
from datachef.contracts import (
    PlanningContext,
    PlanValidationResult,
    ReviewerDecision,
    ReviewerVerdict,
    TransformationPlan,
)
from datachef.planning import RuleBasedPlanner, RuleBasedReviewer, validate_plan
from datachef.planning.plan import expected_plan_id

DEFAULT_TIMEOUT_SECONDS = 90.0


@dataclass
class AgentPlanner:
    """Live planner with a deterministic fallback and a log-safe trace."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    fallback: RuleBasedPlanner = field(default_factory=RuleBasedPlanner)
    trace: AgentTrace = field(
        default_factory=lambda: AgentTrace(mode=AgentMode.OFFLINE, model_configured=False)
    )
    environment: dict[str, str] | None = None
    _runner: object | None = None

    def propose(
        self,
        context: PlanningContext,
        *,
        attempt: int,
    ) -> TransformationPlan:
        started = time.monotonic()
        decision = decide_live_model(self.environment)
        self.trace = self.trace.model_copy(
            update={
                "mode": AgentMode.LIVE if decision.permitted else AgentMode.OFFLINE,
                "model_configured": decision.model_configured,
            }
        )
        if not decision.permitted:
            return self._fall_back(
                context,
                attempt=attempt,
                started=started,
                outcome=AgentOutcome.AGENT_OFFLINE_USING_DETERMINISTIC_PLANNER,
                invocations=(),
            )

        notices: tuple[AgentOutcome, ...] = ()
        try:
            plan, invocations, notices = self._run_live(context, decision.model_name)
            self._assert_contract(context, plan)
        except Exception:  # noqa: BLE001 - every failure has the same safe answer
            return self._fall_back(
                context,
                attempt=attempt,
                started=started,
                outcome=AgentOutcome.AGENT_UNAVAILABLE_USING_DETERMINISTIC_PLANNER,
                invocations=(),
            )

        validation = validate_plan(context, plan)
        if not validation.valid:
            return self._fall_back(
                context,
                attempt=attempt,
                started=started,
                outcome=(
                    AgentOutcome.AGENT_CONTRACT_VIOLATION_USING_DETERMINISTIC_PLANNER
                ),
                invocations=invocations,
                validation_codes=tuple(
                    finding.code for finding in validation.findings
                ),
                notices=notices,
            )

        self.trace = self.trace.with_attempt(
            AttemptTrace(
                attempt=attempt,
                agent="planner",
                outcome_code=AgentOutcome.AGENT_PLAN_ACCEPTED,
                tool_invocations=invocations,
                notices=notices,
                elapsed_ms=self._elapsed(started),
            )
        )
        return plan

    def _run_live(self, context: PlanningContext, model_name: str | None):
        from datachef.agents.llm import build_llm
        from datachef.agents.plan_crew import run_planning_crew

        runner = self._runner
        if runner is None:
            llm = build_llm(model_name or "")
            result = run_planning_crew(context, llm, self.timeout_seconds)
        else:
            result = runner(context)  # type: ignore[operator]
        notices = (
            (AgentOutcome.AGENT_RUNTIME_CLEANUP_DEFERRED,)
            if getattr(result, "cleanup_deferred", False)
            else ()
        )
        return result.plan, tuple(result.draft.invocations), notices

    def _assert_contract(self, context: PlanningContext, plan: TransformationPlan) -> None:
        """We re-derive identity ourselves; the agent never authors one."""

        identity = context.dataset_identity
        if plan.dataset_id != identity.dataset_id:
            raise ValueError("dataset id mismatch")
        if plan.dataset_fingerprint != identity.fingerprint:
            raise ValueError("dataset fingerprint mismatch")
        if plan.plan_id != expected_plan_id(plan):
            raise ValueError("plan identity mismatch")

    def _fall_back(
        self,
        context: PlanningContext,
        *,
        attempt: int,
        started: float,
        outcome: AgentOutcome,
        invocations: tuple,
        validation_codes: tuple[str, ...] = (),
        notices: tuple[AgentOutcome, ...] = (),
    ) -> TransformationPlan:
        self.trace = self.trace.model_copy(update={"fallback_reason_code": outcome})
        self.trace = self.trace.with_attempt(
            AttemptTrace(
                attempt=attempt,
                agent="planner",
                outcome_code=outcome,
                tool_invocations=invocations,
                validation_codes=validation_codes,
                notices=notices,
                elapsed_ms=self._elapsed(started),
            )
        )
        return self.fallback.propose(context, attempt=attempt)

    @staticmethod
    def _elapsed(started: float) -> float:
        return round(max(0.0, (time.monotonic() - started) * 1000.0), 3)


@dataclass
class AgentReviewer:
    """Live reviewer with the same single-funnel fallback shape."""

    fallback: RuleBasedReviewer = field(default_factory=RuleBasedReviewer)
    trace: AgentTrace = field(
        default_factory=lambda: AgentTrace(mode=AgentMode.OFFLINE, model_configured=False)
    )
    environment: dict[str, str] | None = None
    _runner: object | None = None

    def review(
        self,
        context: PlanningContext,
        plan: TransformationPlan,
        validation: PlanValidationResult,
        *,
        previous_feedback: tuple[str, ...],
        attempt: int,
    ) -> ReviewerVerdict:
        started = time.monotonic()
        decision = decide_live_model(self.environment)
        self.trace = self.trace.model_copy(
            update={
                "mode": AgentMode.LIVE if decision.permitted else AgentMode.OFFLINE,
                "model_configured": decision.model_configured,
            }
        )
        if not decision.permitted:
            return self._fall_back(
                context,
                plan,
                validation,
                previous_feedback=previous_feedback,
                attempt=attempt,
                started=started,
                outcome=AgentOutcome.AGENT_OFFLINE_USING_DETERMINISTIC_REVIEWER,
            )
        try:
            runner = self._runner
            if runner is None:
                raise RuntimeError("no live reviewer configured")
            verdict = runner(context, plan, validation)  # type: ignore[operator]
            if not isinstance(verdict, ReviewerVerdict) or verdict.plan_id != plan.plan_id:
                raise ValueError("reviewer contract violation")
        except Exception:  # noqa: BLE001 - one safe answer for every failure
            return self._fall_back(
                context,
                plan,
                validation,
                previous_feedback=previous_feedback,
                attempt=attempt,
                started=started,
                outcome=AgentOutcome.AGENT_UNAVAILABLE_USING_DETERMINISTIC_REVIEWER,
            )
        self.trace = self.trace.with_attempt(
            AttemptTrace(
                attempt=attempt,
                agent="reviewer",
                outcome_code=AgentOutcome.AGENT_REVIEW_ACCEPTED,
                elapsed_ms=AgentPlanner._elapsed(started),
            )
        )
        return verdict

    def _fall_back(
        self,
        context: PlanningContext,
        plan: TransformationPlan,
        validation: PlanValidationResult,
        *,
        previous_feedback: tuple[str, ...],
        attempt: int,
        started: float,
        outcome: AgentOutcome,
    ) -> ReviewerVerdict:
        self.trace = self.trace.model_copy(update={"fallback_reason_code": outcome})
        self.trace = self.trace.with_attempt(
            AttemptTrace(
                attempt=attempt,
                agent="reviewer",
                outcome_code=outcome,
                validation_codes=tuple(
                    finding.code for finding in validation.findings
                ),
                elapsed_ms=AgentPlanner._elapsed(started),
            )
        )
        return self.fallback.review(
            context,
            plan,
            validation,
            previous_feedback=previous_feedback,
            attempt=attempt,
        )


__all__ = ["AgentPlanner", "AgentReviewer", "DEFAULT_TIMEOUT_SECONDS"]
