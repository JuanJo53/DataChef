"""Typed, log-safe trace of what the planning crew did.

The trace records codes, tool names, and column names as they appear in the
planning context. It never carries cell values, free-form model prose, or
exception text: every explanatory field is a closed enum or a code string.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StrictTraceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AgentMode(StrEnum):
    LIVE = "LIVE"
    OFFLINE = "OFFLINE"


class AgentOutcome(StrEnum):
    """Every terminal outcome an agent stage can report."""

    AGENT_PLAN_ACCEPTED = "AGENT_PLAN_ACCEPTED"
    AGENT_REVIEW_ACCEPTED = "AGENT_REVIEW_ACCEPTED"
    AGENT_OFFLINE_USING_DETERMINISTIC_PLANNER = (
        "AGENT_OFFLINE_USING_DETERMINISTIC_PLANNER"
    )
    AGENT_UNAVAILABLE_USING_DETERMINISTIC_PLANNER = (
        "AGENT_UNAVAILABLE_USING_DETERMINISTIC_PLANNER"
    )
    AGENT_CONTRACT_VIOLATION_USING_DETERMINISTIC_PLANNER = (
        "AGENT_CONTRACT_VIOLATION_USING_DETERMINISTIC_PLANNER"
    )
    AGENT_RUNTIME_CLEANUP_DEFERRED = "AGENT_RUNTIME_CLEANUP_DEFERRED"
    AGENT_OFFLINE_USING_DETERMINISTIC_REVIEWER = (
        "AGENT_OFFLINE_USING_DETERMINISTIC_REVIEWER"
    )
    AGENT_UNAVAILABLE_USING_DETERMINISTIC_REVIEWER = (
        "AGENT_UNAVAILABLE_USING_DETERMINISTIC_REVIEWER"
    )


class ToolInvocation(StrictTraceModel):
    """One tool call, as the tool boundary judged it."""

    tool_name: str = Field(min_length=1)
    accepted: bool
    reason_code: str | None = None
    operation_type: str | None = None
    target_columns: tuple[str, ...] = ()
    # What the deterministic critic replied, when this call was an estimate.
    critic_finding_codes: tuple[str, ...] = ()
    estimated_row_loss_pct: float | None = None


class AttemptTrace(StrictTraceModel):
    attempt: int = Field(ge=1)
    agent: str = Field(min_length=1)
    outcome_code: AgentOutcome
    tool_invocations: tuple[ToolInvocation, ...] = ()
    validation_codes: tuple[str, ...] = ()
    notices: tuple[AgentOutcome, ...] = ()
    elapsed_ms: float = Field(ge=0)


class AgentTrace(StrictTraceModel):
    mode: AgentMode
    model_configured: bool
    fallback_reason_code: AgentOutcome | None = None
    attempts: tuple[AttemptTrace, ...] = ()

    def with_attempt(self, attempt: "AttemptTrace") -> "AgentTrace":
        return self.model_copy(update={"attempts": (*self.attempts, attempt)})


__all__ = [
    "AgentMode",
    "AgentOutcome",
    "AgentTrace",
    "AttemptTrace",
    "StrictTraceModel",
    "ToolInvocation",
]
