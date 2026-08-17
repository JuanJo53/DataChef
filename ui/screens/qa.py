"""Quality stage: render the quality verdict. No downloads, no dashboard here."""

from __future__ import annotations

from typing import Any

import streamlit as st

from datachef.contracts import WorkflowStage
from ui import state as ui_state
from ui.screens import render_findings, render_result


_STAGE_MESSAGES = {
    WorkflowStage.EXECUTING: (
        "The approved plan is still running.",
        "info",
    ),
    WorkflowStage.EXECUTION_FAILED: (
        "The approved plan did not finish. No gold table was produced, so "
        "downloads and the dashboard stay closed.",
        "error",
    ),
    WorkflowStage.QA_WARNING: (
        "Quality assurance returned a warning. Gold was withheld, so downloads "
        "and the dashboard stay closed.",
        "warning",
    ),
    WorkflowStage.QA_FAILED: (
        "Quality assurance failed. Gold was withheld, so downloads and the "
        "dashboard stay closed.",
        "error",
    ),
}


def _render_report(report: Any) -> None:
    st.markdown("### Quality report")
    first, second, third, fourth = st.columns(4)
    first.metric("Rows before", report.before_row_count)
    second.metric("Rows after", report.after_row_count)
    third.metric("Columns after", report.after_column_count)
    fourth.metric("Row loss", f"{report.row_loss_pct:.2f}%")
    st.caption(f"Report `{report.qa_report_id}` · status **{report.status.value}**")

    failing = [item for item in report.invariant_results if item.status.value != "PASS"]
    if failing:
        st.markdown("#### Invariants that did not pass")
        for invariant in failing:
            st.markdown(
                f"- `{invariant.status.value}` **{invariant.kind.value}** — "
                f"{invariant.explanation}"
            )
    if report.execution_failures:
        st.markdown("#### Recorded operation failures")
        for code in report.execution_failures:
            st.markdown(f"- `{code}`")


def render(controller: Any, state: Any) -> None:
    st.header("5 · Quality")
    session = controller.session
    runtime = session.workflow_runtime
    if runtime is None:
        st.info("Run an approved plan first.")
        return

    evidence = runtime.state
    message, level = _STAGE_MESSAGES.get(
        evidence.stage,
        ("This run is not in a quality stage.", "info"),
    )
    getattr(st, level)(message)
    if evidence.last_error_code:
        st.caption(f"Reported code: `{evidence.last_error_code}`")

    if evidence.execution_result is not None:
        result = evidence.execution_result
        st.markdown("### Execution")
        st.caption(
            f"`{result.execution_id}` · success **{result.success}** · "
            f"{len(result.operation_records)} operation record(s)"
        )
        for record in result.operation_records:
            st.markdown(
                f"- `{record.status.value}` **{record.operation_id}** — "
                f"{record.rows_before} → {record.rows_after} rows"
            )

    if evidence.qa_report is not None:
        _render_report(evidence.qa_report)

    render_findings(session.findings)
    render_result(ui_state.last_result(state))
