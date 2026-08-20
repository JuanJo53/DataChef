"""Intent stage: collect typed intent and typed requests, submit, show findings."""

from __future__ import annotations

from typing import Any

import streamlit as st

from datachef.contracts import DownstreamUse, UserIntent
from ui import state as ui_state
from ui.screens import (
    build_transformation_requests,
    render_diagnosis,
    render_findings,
    render_result,
)


def _column_names(session: Any) -> list[str]:
    identity = session.source.identity
    return [column.name for column in identity.column_schema]


def _render_suggested_questions(session: Any) -> list[str]:
    if not session.suggested_questions:
        return []
    labels = {
        f"{item.question}  ·  {item.kind.value}": item.question_id
        for item in session.suggested_questions
    }
    chosen = st.multiselect(
        "Deterministic suggested questions to carry into the dashboard",
        list(labels),
        key=ui_state.SUGGESTED_QUESTIONS_WIDGET,
    )
    return [labels[label] for label in chosen]


def render(controller: Any, state: Any) -> None:
    st.header("3 · Objective")
    session = controller.session
    if session.source is None or session.display_diagnostic_report is None:
        st.info("Upload a dataset and run the diagnosis first.")
        return

    # The diagnosis is produced on the previous stage but the controller lands
    # the user here, so this is where that evidence has to be readable.
    render_diagnosis(session)

    columns = _column_names(session)

    st.markdown("### What you need from this table")
    goal = st.text_area(
        "What should this table be good for?",
        key=ui_state.GOAL_WIDGET,
        placeholder="Prepare the table for analysis.",
        height=90,
    )
    left, right = st.columns(2)
    downstream = left.selectbox(
        "Downstream use",
        list(DownstreamUse),
        format_func=lambda item: item.value,
        key=ui_state.DOWNSTREAM_WIDGET,
    )
    row_loss = right.slider(
        "Acceptable row loss (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key=ui_state.ROW_LOSS_WIDGET,
    )
    required_columns = st.multiselect(
        "Columns that must survive",
        columns,
        key=ui_state.REQUIRED_COLUMNS_WIDGET,
        help="DataChef refuses any plan that would drop one of these columns.",
    )
    key_columns = st.multiselect(
        "Which column tells you two rows are the same record?",
        columns,
        key=ui_state.KEY_COLUMNS_WIDGET,
        help=(
            "If two rows share this value, DataChef treats them as duplicates — "
            "for example order_id or email. Leave empty if you're not sure."
        ),
    )
    st.markdown("### Typed transformation requests")
    st.caption(
        "This offline slice executes numeric casts. The planner must account "
        "for every request you make here."
    )
    cast_columns = st.multiselect(
        "Cast these columns to numeric",
        columns,
        key=ui_state.CAST_REQUEST_WIDGET,
    )
    # A separate "deduplicate by these keys" control asked the same question as
    # "which column tells you two rows are the same record?" above, in different
    # words. The key columns already drive deduplication, so the duplicate input
    # is gone; the request contract and the submit handler are unchanged.

    # Questions drive the dashboard, not the cleaning plan, so they sit below
    # every cleaning field in their own section.
    st.markdown("### Business questions for the dashboard")
    st.caption(
        "Optional. These shape the dashboard you get at the end — they do not "
        "change how your data is cleaned."
    )
    authored = st.text_area(
        "Your analytical questions (one per line)",
        key=ui_state.QUESTIONS_WIDGET,
        height=80,
    )
    selected_question_ids = _render_suggested_questions(session)

    render_findings(session.findings)

    if st.button(
        "Save objective and build a plan",
        key=ui_state.SUBMIT_INTENT_WIDGET,
        type="primary",
    ):
        try:
            intent = UserIntent(
                intent_id=f"intent-{session.source.identity.dataset_id}",
                user_goal=goal or "",
                downstream_use=downstream,
                selected_key_columns=tuple(key_columns),
                required_columns=tuple(required_columns),
                acceptable_row_loss_pct=float(row_loss),
                questions=tuple(
                    line.strip() for line in (authored or "").splitlines() if line.strip()
                ),
            )
            requests = build_transformation_requests(cast_columns, [])
        except (ValueError, TypeError):
            st.error(
                "**INVALID_INTENT_REQUEST** — the submitted intent or requests "
                "are inconsistent. Adjust the selections and try again."
            )
            return
        result = ui_state.remember_result(
            state,
            controller.submit_intent(
                intent,
                requests,
                selected_question_ids=tuple(selected_question_ids),
            ),
        )
        if result.changed:
            ui_state.clear_action_commands(state)
        render_findings(result.findings)
        st.rerun()

    render_result(ui_state.last_result(state))
