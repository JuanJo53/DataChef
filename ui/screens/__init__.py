"""Screen modules mapped 1:1 onto the committed ScreenId enum.

Routing is driven entirely by ``controller.session.screen``. The shell never
picks a stage itself, and no screen grants a capability: each one re-derives
what it may show from controller return values.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from datachef.application import RequestedTransformation, ScreenId
from datachef.contracts import (
    CastColumnParameters,
    CastTarget,
    DeduplicateByKeysParameters,
    KeepPolicy,
    OperationType,
)


def build_transformation_requests(
    cast_columns: list[str],
    dedup_keys: list[str],
) -> tuple[RequestedTransformation, ...]:
    """Build the typed requests the offline slice can execute."""

    built: list[RequestedTransformation] = []
    for column in cast_columns:
        built.append(
            RequestedTransformation(
                request_id=f"request-cast-{column}",
                operation_type=OperationType.CAST_COLUMN,
                target_columns=(column,),
                parameters=CastColumnParameters(target_type=CastTarget.NUMERIC),
            )
        )
    if dedup_keys:
        keys = tuple(dedup_keys)
        built.append(
            RequestedTransformation(
                request_id="request-deduplicate-by-keys",
                operation_type=OperationType.DEDUPLICATE_BY_KEYS,
                target_columns=keys,
                parameters=DeduplicateByKeysParameters(keys=keys, keep=KeepPolicy.FIRST),
            )
        )
    return tuple(built)


def render_diagnosis(session: Any) -> None:
    """Render the deterministic diagnosis the controller already produced.

    Shared so the screen the user actually lands on after diagnosing shows this
    evidence. Values are read off session evidence; nothing is derived here.
    """

    report = session.display_diagnostic_report
    if report is None:
        return
    st.markdown("### Deterministic diagnosis")
    evidence = report.legacy_evidence
    left, middle, right = st.columns(3)
    left.metric("Health score", evidence.health_score)
    middle.metric("Grade", evidence.health_grade)
    right.metric("Duplicate rows", report.duplicate_row_count)
    if not report.issues:
        st.success("No diagnostic issues were detected.")
        return
    st.markdown("#### Issues")
    for issue in report.issues:
        columns = ", ".join(issue.affected_columns) or "—"
        st.markdown(
            f"- `{issue.severity.value}` **{issue.title}** "
            f"({issue.kind.value}) — columns: {columns}"
        )


def render_validation(state_evidence: Any, intent: Any) -> None:
    """Show the validation facts the application already produced.

    This renders evidence only: codes, messages, operation IDs, and numbers.
    It never recommends a threshold or interprets the outcome.
    """

    validation = state_evidence.plan_validation
    if validation is None:
        return
    st.markdown("### Validation")
    left, right = st.columns(2)
    left.metric(
        "Estimated cumulative row loss",
        f"{validation.cumulative_estimated_row_loss_pct:.2f}%",
    )
    if intent is not None:
        right.metric(
            "Your acceptable row loss",
            f"{intent.acceptable_row_loss_pct:.2f}%",
        )
    if validation.row_loss_estimates:
        st.markdown("#### Estimated row removal per operation")
        for estimate in validation.row_loss_estimates:
            st.markdown(
                f"- `{estimate.operation_id}` — {estimate.estimated_rows} row(s), "
                f"{estimate.estimated_pct:.2f}%"
            )
    if validation.findings:
        st.markdown("#### Validation findings")
        for finding in validation.findings:
            operation = f" (`{finding.operation_id}`)" if finding.operation_id else ""
            st.markdown(
                f"- `{finding.severity.value}` **{finding.code}**{operation} — "
                f"{finding.message}"
            )


def render_findings(findings: Any) -> None:
    """Render sanitized application findings only; never exception text."""

    for finding in findings or ():
        line = f"**{finding.code}** — {finding.safe_message}"
        if finding.request_id:
            line = f"{line} _(request {finding.request_id})_"
        if finding.blocking:
            st.error(line)
        else:
            st.warning(line)


def render_result(result: Any) -> None:
    """Render the controller's own outcome code; never a recomputed status."""

    if result is None:
        return
    st.caption(f"Last controller outcome: `{result.code}`")


def render_failure(failure: Any) -> None:
    """Render a sanitized artifact/dashboard refusal."""

    if failure is None:
        return
    st.warning(f"**{failure.code.value}** — {failure.safe_message}")
    st.caption(failure.suggested_action)


def has_blocking(findings: Any) -> bool:
    """Report whether the controller flagged a blocking finding."""

    return any(finding.blocking for finding in findings or ())


def render_screen(screen: ScreenId, controller: Any, state: Any) -> None:
    from ui.screens import approval, intent, plan, qa, results, upload

    renderers = {
        ScreenId.UPLOAD: upload.render,
        ScreenId.DIAGNOSE: upload.render,
        ScreenId.INTENT: intent.render,
        ScreenId.PLAN: plan.render,
        ScreenId.APPROVAL: approval.render,
        ScreenId.QA: qa.render,
        ScreenId.RESULTS: results.render,
    }
    renderers[screen](controller, state)


__all__ = [
    "build_transformation_requests",
    "has_blocking",
    "render_diagnosis",
    "render_failure",
    "render_findings",
    "render_result",
    "render_validation",
    "render_screen",
]
