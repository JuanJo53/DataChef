"""Offline dashboard handoff that adapts the existing rule-based team builder.

Phase 1B does not derive its own semantic roles, KPIs, or chart suggestions.
The deterministic teammate builder in `crew.dashboard_agent` is the chart
recommendation source; this module gates it behind verified gold evidence,
calls it offline, validates the dictionary it returns, and hands the result
over as a runtime-only object.

The legacy module is imported lazily inside the successful construction path,
so importing `datachef.application` never loads it, CrewAI, a Google SDK, or
Streamlit.
"""

from __future__ import annotations

import copy
from enum import StrEnum
from hashlib import sha256
from typing import Any

import pandas as pd
from pydantic import Field

from datachef.application.artifacts import ArtifactFailure, gold_evidence_failure
from datachef.application.models import StrictApplicationModel
from datachef.contracts import (
    DownstreamUse,
    InvariantStatus,
    SuggestedQuestion,
    UserIntent,
)
from datachef.workflow import WorkflowRuntime


LEGACY_TITLE = "Sales Dashboard"
DASHBOARD_TITLE = "DataChef Dashboard"

_ALLOWED_CHART_TYPES = frozenset({"line", "bar", "pie", "histogram"})
_ALLOWED_AGGREGATIONS = frozenset({"sum", "count"})
_REQUIRED_SPEC_KEYS = frozenset(
    {"title", "engine", "meta", "roles", "kpis", "charts", "insights"}
)


class DashboardFailureCode(StrEnum):
    GOLD_UNAVAILABLE = "GOLD_UNAVAILABLE"
    EVIDENCE_INCOMPLETE = "EVIDENCE_INCOMPLETE"
    GOLD_EVIDENCE_MISMATCH = "GOLD_EVIDENCE_MISMATCH"
    DASHBOARD_SPEC_INVALID = "DASHBOARD_SPEC_INVALID"
    DASHBOARD_BUILDER_FAILURE = "DASHBOARD_BUILDER_FAILURE"


class DashboardFailure(StrictApplicationModel):
    code: DashboardFailureCode
    safe_message: str = Field(min_length=1)
    suggested_action: str = Field(min_length=1)


_SAFE_MESSAGES = {
    DashboardFailureCode.GOLD_UNAVAILABLE: (
        "No verified gold table is available to visualize.",
        "Complete an approved run whose quality assurance passed, then open the dashboard.",
    ),
    DashboardFailureCode.EVIDENCE_INCOMPLETE: (
        "The completed run does not carry the evidence a dashboard must cite.",
        "Re-run the approved plan so plan, execution, and quality evidence exist.",
    ),
    DashboardFailureCode.GOLD_EVIDENCE_MISMATCH: (
        "The gold table does not match its recorded execution evidence.",
        "Re-run the approved plan; do not chart this result.",
    ),
    DashboardFailureCode.DASHBOARD_SPEC_INVALID: (
        "The deterministic dashboard specification is not usable for this table.",
        "Re-run the approved plan; if it persists, report the dataset shape.",
    ),
    DashboardFailureCode.DASHBOARD_BUILDER_FAILURE: (
        "The deterministic dashboard specification could not be produced.",
        "Re-run the approved plan; if it persists, report the dataset shape.",
    ),
}


def _failure(code: DashboardFailureCode) -> DashboardFailure:
    message, action = _SAFE_MESSAGES[code]
    return DashboardFailure(code=code, safe_message=message, suggested_action=action)


def _translated(blocked: ArtifactFailure) -> DashboardFailure:
    try:
        return _failure(DashboardFailureCode(blocked.code.value))
    except ValueError:
        return _failure(DashboardFailureCode.GOLD_UNAVAILABLE)


class DashboardContext(StrictApplicationModel):
    """Safe local presentation metadata; never provider context or an artifact."""

    handoff_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    result_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_id: str = Field(min_length=1)
    qa_report_id: str = Field(min_length=1)
    downstream_use: DownstreamUse
    authored_questions: tuple[str, ...] = Field(default_factory=tuple)
    selected_questions: tuple[SuggestedQuestion, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class DashboardHandoff:
    """Runtime-only owner of gold, safe context, and the validated legacy spec."""

    __slots__ = ("__context", "__gold", "__spec")

    def __init__(
        self,
        *,
        context: DashboardContext,
        gold: pd.DataFrame,
        spec: dict[str, Any],
    ) -> None:
        if not isinstance(gold, pd.DataFrame):
            raise TypeError("dashboard handoff requires a gold DataFrame")
        object.__setattr__(self, "_DashboardHandoff__context", context)
        object.__setattr__(self, "_DashboardHandoff__gold", gold.copy(deep=True))
        object.__setattr__(self, "_DashboardHandoff__spec", copy.deepcopy(spec))

    def __getattribute__(self, name: str) -> object:
        if name in {"_gold", "_DashboardHandoff__gold", "_DashboardHandoff__spec"}:
            raise AttributeError("dashboard handoff backing storage is not public")
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("dashboard handoff is immutable")

    @property
    def context(self) -> DashboardContext:
        return object.__getattribute__(self, "_DashboardHandoff__context")

    def gold_frame(self) -> pd.DataFrame:
        """Return a fresh deep copy; callers never receive the stored frame."""

        return object.__getattribute__(self, "_DashboardHandoff__gold").copy(deep=True)

    def dashboard_spec(self) -> dict[str, Any]:
        """Return a fresh deep copy of the validated rule-based specification."""

        return copy.deepcopy(object.__getattribute__(self, "_DashboardHandoff__spec"))

    def __repr__(self) -> str:
        return f"DashboardHandoff(context={self.context!r}, gold=<private>, spec=<private>)"

    def __getstate__(self) -> object:
        raise TypeError("dashboard handoff is runtime-only and cannot be serialized")


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _roles_are_valid(roles: object, columns: frozenset[str]) -> bool:
    if not isinstance(roles, dict):
        return False
    if set(roles) != {"date", "measures", "dimensions"}:
        return False
    date = roles["date"]
    if date is not None and (not _is_text(date) or date not in columns):
        return False
    for key in ("measures", "dimensions"):
        values = roles[key]
        if not isinstance(values, list):
            return False
        if any(not _is_text(item) or item not in columns for item in values):
            return False
    return True


def _kpis_are_valid(kpis: object) -> bool:
    if not isinstance(kpis, list) or not kpis:
        return False
    for kpi in kpis:
        if not isinstance(kpi, dict):
            return False
        if not _is_text(kpi.get("label")) or not _is_text(kpi.get("format")):
            return False
        if not _is_number(kpi.get("value")):
            return False
    return True


def _charts_are_valid(charts: object, columns: frozenset[str]) -> bool:
    if not isinstance(charts, list):
        return False
    for chart in charts:
        if not isinstance(chart, dict):
            return False
        if chart.get("type") not in _ALLOWED_CHART_TYPES:
            return False
        if not _is_text(chart.get("title")):
            return False
        dimension = chart.get("x")
        if not _is_text(dimension) or dimension not in columns:
            return False
        measure = chart.get("y")
        if measure is not None and (not _is_text(measure) or measure not in columns):
            return False
        if "agg" in chart and chart["agg"] not in _ALLOWED_AGGREGATIONS:
            return False
    return True


def _spec_is_valid(spec: object, gold: pd.DataFrame) -> bool:
    """Validate the dictionary the legacy builder actually returned."""

    if not isinstance(spec, dict):
        return False
    if "error" in spec:
        return False
    if not _REQUIRED_SPEC_KEYS.issubset(spec):
        return False
    if spec.get("engine") != "rule-based":
        return False
    if not _is_text(spec.get("title")):
        return False
    meta = spec.get("meta")
    if not isinstance(meta, dict):
        return False
    rows, column_count = meta.get("rows"), meta.get("columns")
    if not isinstance(rows, int) or isinstance(rows, bool):
        return False
    if not isinstance(column_count, int) or isinstance(column_count, bool):
        return False
    if rows != int(gold.shape[0]) or column_count != int(gold.shape[1]):
        return False
    columns = frozenset(str(name) for name in gold.columns)
    if not _roles_are_valid(spec.get("roles"), columns):
        return False
    if not _kpis_are_valid(spec.get("kpis")):
        return False
    if not _charts_are_valid(spec.get("charts"), columns):
        return False
    insights = spec.get("insights")
    if not isinstance(insights, list) or any(
        not isinstance(item, str) for item in insights
    ):
        return False
    return True


def _retitled(spec: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(spec)
    if copied.get("title") == LEGACY_TITLE:
        copied["title"] = DASHBOARD_TITLE
    return copied


def _warnings(runtime: WorkflowRuntime) -> tuple[str, ...]:
    report = runtime.state.qa_report
    assert report is not None
    messages: list[str] = []
    if report.row_loss_pct > 0:
        messages.append(
            f"The approved plan removed {report.row_loss_pct:.2f}% of the source rows."
        )
    if report.removed_columns:
        messages.append(
            f"{len(report.removed_columns)} column(s) were removed by the approved plan."
        )
    if report.duplicate_rows_after > 0:
        messages.append("The gold table still contains duplicate rows.")
    if report.duplicate_keys_after:
        messages.append("The gold table still contains duplicate key values.")
    for invariant in report.invariant_results:
        if invariant.status is InvariantStatus.WARN:
            messages.append(f"Quality warning on invariant {invariant.kind.value}.")
    return tuple(messages)


def _handoff_id(result_fingerprint: str, plan_id: str, qa_report_id: str) -> str:
    digest = sha256()
    for part in (result_fingerprint, plan_id, qa_report_id):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return f"handoff-{digest.hexdigest()}"


def build_dashboard_handoff(
    runtime: object,
    intent: object,
    *,
    selected_questions: tuple[SuggestedQuestion, ...] = (),
) -> DashboardHandoff | DashboardFailure:
    """Gate on verified gold, then adapt the deterministic team dashboard builder."""

    blocked = gold_evidence_failure(runtime)
    if blocked is not None:
        return _translated(blocked)
    if not isinstance(intent, UserIntent):
        return _failure(DashboardFailureCode.EVIDENCE_INCOMPLETE)
    assert isinstance(runtime, WorkflowRuntime)
    state = runtime.state
    result = state.execution_result
    report = state.qa_report
    plan = state.transformation_plan
    identity = state.dataset_identity
    assert result is not None and report is not None
    assert plan is not None and identity is not None
    assert result.result_fingerprint is not None
    assert runtime.gold_dataframe is not None

    gold = runtime.gold_dataframe
    from crew.dashboard_agent.dashboard_agent import build_dashboard_spec

    gold_copy = gold.copy(deep=True)
    try:
        spec = build_dashboard_spec(gold_copy, use_llm=False)
    except Exception:
        return _failure(DashboardFailureCode.DASHBOARD_BUILDER_FAILURE)
    if not _spec_is_valid(spec, gold):
        return _failure(DashboardFailureCode.DASHBOARD_SPEC_INVALID)

    context = DashboardContext(
        handoff_id=_handoff_id(
            result.result_fingerprint,
            plan.plan_id,
            report.qa_report_id,
        ),
        dataset_id=identity.dataset_id,
        result_fingerprint=result.result_fingerprint,
        plan_id=plan.plan_id,
        qa_report_id=report.qa_report_id,
        downstream_use=intent.downstream_use,
        authored_questions=intent.questions,
        selected_questions=selected_questions,
        warnings=_warnings(runtime),
    )
    return DashboardHandoff(context=context, gold=gold, spec=_retitled(spec))


__all__ = [
    "DASHBOARD_TITLE",
    "DashboardContext",
    "DashboardFailure",
    "DashboardFailureCode",
    "DashboardHandoff",
    "build_dashboard_handoff",
]
