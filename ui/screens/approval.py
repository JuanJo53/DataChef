"""Approval stage: record the human decision, then run the approved plan."""

from __future__ import annotations

from typing import Any

import streamlit as st

from datachef.contracts import DiagnosticIssueKind, HumanDecision
from ui import state as ui_state
from ui.screens import has_blocking, render_findings, render_result


# The offline slice executes exactly these two operation families, so any other
# diagnostic finding has no executable operation behind it.
_EXECUTABLE_ISSUE_KINDS = frozenset(
    {
        DiagnosticIssueKind.CANDIDATE_TYPE_CONVERSION,
        DiagnosticIssueKind.DUPLICATE_KEYS,
    }
)


def _unaddressable_kinds(report: Any) -> tuple[str, ...]:
    """Name the reported finding kinds this offline slice cannot act on."""

    if report is None:
        return ()
    return tuple(
        sorted(
            {
                issue.kind.value
                for issue in report.issues
                if issue.kind not in _EXECUTABLE_ISSUE_KINDS
            }
        )
    )


def _render_plan_summary(evidence: Any, report: Any) -> None:
    plan = evidence.transformation_plan
    if plan is None:
        return
    st.markdown("### What you are approving")
    st.caption(f"`{plan.plan_id}` · version {plan.version} — {plan.summary}")
    if not plan.operations:
        st.info("The reviewed plan is empty; execution will copy the table unchanged.")
        unaddressable = _unaddressable_kinds(report)
        if unaddressable:
            st.caption(
                "The diagnosis reported "
                + ", ".join(unaddressable)
                + ". This offline slice has no executable operation for those, so "
                "the plan leaves them as they are."
            )
    for index, operation in enumerate(plan.operations, start=1):
        columns = ", ".join(operation.target_columns) or "—"
        # Risk is read straight off the plan operation and shown as text.
        st.markdown(
            f"{index}. **{operation.operation_type.value}** on {columns} — "
            f"risk **{operation.risk.value}** — {operation.expected_effect}"
        )
    accepted = evidence.accepted_review
    if accepted is not None:
        st.caption(
            f"Reviewer accepted attempt {accepted.attempt} for plan "
            f"`{accepted.plan_id}` v{accepted.plan_version}."
        )


def _render_approval_record(approval: Any) -> None:
    st.success(
        f"Decision recorded: **{approval.decision.value}** for plan "
        f"`{approval.plan_id}` v{approval.plan_version} "
        f"({len(approval.approved_operation_ids)} operation(s) approved)."
    )


def render(controller: Any, state: Any) -> None:
    st.header("4 · Approve")
    session = controller.session
    runtime = session.workflow_runtime
    if runtime is None:
        st.info("Prepare a plan first.")
        return

    _render_plan_summary(runtime.state, session.display_diagnostic_report)
    render_findings(session.findings)

    blocking = has_blocking(session.findings)
    if blocking:
        st.error(
            "Approval is disabled while a blocking finding is open. Revise your "
            "objective or requests to clear it."
        )

    if session.pending_approval is None:
        approve, reject = st.columns(2)
        if approve.button(
            "Approve this plan",
            key=ui_state.APPROVE_WIDGET,
            type="primary",
            disabled=blocking,
            use_container_width=True,
        ):
            result = ui_state.remember_result(
                state,
                controller.record_human_decision(
                    HumanDecision.APPROVE,
                    command_id=ui_state.command_id(
                        state,
                        ui_state.human_command_slot(HumanDecision.APPROVE),
                    ),
                ),
            )
            render_findings(result.findings)
            st.rerun()
        if reject.button(
            "Reject this plan",
            key=ui_state.REJECT_WIDGET,
            use_container_width=True,
        ):
            result = ui_state.remember_result(
                state,
                controller.record_human_decision(
                    HumanDecision.REJECT,
                    command_id=ui_state.command_id(
                        state,
                        ui_state.human_command_slot(HumanDecision.REJECT),
                    ),
                ),
            )
            render_findings(result.findings)
            st.rerun()
        render_result(ui_state.last_result(state))
        return

    _render_approval_record(session.pending_approval)
    if st.button(
        "Run the approved plan",
        key=ui_state.EXECUTE_WIDGET,
        type="primary",
    ):
        result = ui_state.remember_result(
            state,
            controller.execute_current_plan(
                command_id=ui_state.command_id(state, ui_state.EXECUTION_COMMAND),
            ),
        )
        render_findings(result.findings)
        st.rerun()
    render_result(ui_state.last_result(state))
