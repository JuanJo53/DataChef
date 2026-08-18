"""Plan stage: ask the controller to prepare a plan, then render its evidence."""

from __future__ import annotations

from typing import Any

import streamlit as st

from datachef.contracts import OperationType, WorkflowStage
from ui import state as ui_state
from ui.screens import (
    build_transformation_requests,
    render_findings,
    render_result,
    render_validation,
)


def _render_operations(plan: Any) -> None:
    st.markdown("### Canonical plan")
    st.caption(f"`{plan.plan_id}` · version {plan.version} — {plan.summary}")
    if not plan.operations:
        st.info(
            "The reviewed plan is empty: the deterministic planner found nothing "
            "that needed changing."
        )
        return
    for index, operation in enumerate(plan.operations, start=1):
        columns = ", ".join(operation.target_columns) or "—"
        # Risk is read straight off the plan operation and shown as text, never
        # as colour alone and never inside a collapsed label the reader can miss.
        st.markdown(
            f"{index}. **{operation.operation_type.value}** on {columns} — "
            f"risk **{operation.risk.value}**"
        )
        with st.expander(f"Operation {index} detail"):
            st.markdown(f"**Why:** {operation.rationale}")
            st.markdown(f"**Expected effect:** {operation.expected_effect}")
            st.caption(f"Operation `{operation.operation_id}`")


def _rejection_message(state_evidence: Any) -> str:
    """Name the terminal code the application reported, whatever that code is."""

    code = state_evidence.last_error_code or "PLANNING_STOPPED"
    message = f"**PLANNING_STOPPED** — the application reported `{code}`."
    verdicts = len(state_evidence.review_history)
    if verdicts:
        return (
            f"{message} {verdicts} plan review verdict(s) were recorded and none "
            "produced an approvable plan."
        )
    return f"{message} Planning stopped before a plan could be offered for approval."


def _current_request_columns(session: Any) -> tuple[list[str], list[str]]:
    casts: list[str] = []
    dedups: list[str] = []
    for request in session.requested_transformations:
        if request.operation_type is OperationType.CAST_COLUMN:
            casts.extend(request.target_columns)
        elif request.operation_type is OperationType.DEDUPLICATE_BY_KEYS:
            dedups.extend(request.target_columns)
    return casts, dedups


def _render_revise_form(controller: Any, state: Any, session: Any) -> None:
    """Revise the recorded intent in place, without discarding the dataset."""

    intent = session.intent
    if intent is None or session.source is None:
        return
    columns = [column.name for column in session.source.identity.column_schema]
    cast_default, dedup_default = _current_request_columns(session)

    with st.expander("Revise your objective and plan again", expanded=True):
        st.caption(
            "Revising keeps the uploaded dataset and its diagnosis. It clears the "
            "prepared plan and the recorded command history for this dataset."
        )
        row_loss = st.slider(
            "Acceptable row loss (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(intent.acceptable_row_loss_pct),
            step=1.0,
            key=ui_state.REVISE_ROW_LOSS_WIDGET,
        )
        key_columns = st.multiselect(
            "Key columns",
            columns,
            default=[column for column in intent.selected_key_columns if column in columns],
            key=ui_state.REVISE_KEY_COLUMNS_WIDGET,
        )
        cast_columns = st.multiselect(
            "Cast these columns to numeric",
            columns,
            default=[column for column in cast_default if column in columns],
            key=ui_state.REVISE_CAST_REQUEST_WIDGET,
        )
        dedup_keys = st.multiselect(
            "Deduplicate rows by these keys",
            columns,
            default=[column for column in dedup_default if column in columns],
            key=ui_state.REVISE_DEDUP_REQUEST_WIDGET,
        )
        if not st.button(
            "Revise objective",
            key=ui_state.REVISE_SUBMIT_WIDGET,
            type="primary",
        ):
            return
        try:
            revised = intent.model_copy(
                update={
                    "acceptable_row_loss_pct": float(row_loss),
                    "selected_key_columns": tuple(key_columns),
                }
            )
            requests = build_transformation_requests(cast_columns, dedup_keys)
        except (ValueError, TypeError):
            st.error(
                "**INVALID_INTENT_REQUEST** — the revised intent or requests are "
                "inconsistent. Adjust the selections and try again."
            )
            return
        result = ui_state.remember_result(
            state,
            controller.revise_intent(
                revised,
                requests,
                selected_question_ids=tuple(session.selected_question_ids),
            ),
        )
        if result.changed:
            ui_state.clear_action_commands(state)
        render_findings(result.findings)
        st.rerun()


def _render_reviews(state_evidence: Any) -> None:
    if not state_evidence.review_history:
        return
    st.markdown("### Reviewer history")
    for verdict in state_evidence.review_history:
        st.markdown(
            f"- attempt {verdict.attempt}: **{verdict.decision.value}**"
        )


def render(controller: Any, state: Any) -> None:
    st.header("3 · Plan")
    session = controller.session
    if session.intent is None:
        st.info("Describe what you need before a plan can be prepared.")
        return

    if st.button(
        "Prepare plan",
        key=ui_state.PREPARE_PLAN_WIDGET,
        type="primary",
    ):
        result = ui_state.remember_result(
            state,
            controller.prepare_plan(
                command_id=ui_state.command_id(state, ui_state.PLAN_COMMAND),
            ),
        )
        render_findings(result.findings)
        st.rerun()

    runtime = session.workflow_runtime
    if runtime is None:
        render_findings(session.findings)
        render_result(ui_state.last_result(state))
        st.info("Prepare a plan to continue.")
        return

    evidence = runtime.state
    if evidence.stage is WorkflowStage.PLAN_REJECTED:
        st.error(_rejection_message(evidence))
        if evidence.transformation_plan is not None:
            _render_operations(evidence.transformation_plan)
        render_validation(evidence, session.intent)
        _render_reviews(evidence)
        render_findings(session.findings)
        _render_revise_form(controller, state, session)
        render_result(ui_state.last_result(state))
        return

    if evidence.transformation_plan is not None:
        _render_operations(evidence.transformation_plan)
    render_validation(evidence, session.intent)
    _render_reviews(evidence)
    render_findings(session.findings)
    render_result(ui_state.last_result(state))
