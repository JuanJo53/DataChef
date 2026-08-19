"""Dashboard stage: the analytical view of verified gold, and only of that.

Results answers "what happened to my data?". This screen answers "what does my
data say?", and it is a separate screen because those are separate questions.

It grants nothing. The charts exist only if ``controller.build_dashboard_handoff()``
returns a handoff, which it does only for a run whose quality assurance passed
and whose gold table still matches its execution evidence. Reaching this screen
by any other route — navigating forward, refreshing, arriving before approval —
renders the controller's own refusal and no data at all.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from datachef.application import DashboardHandoff
from ui import state as ui_state
from ui.charts import render_charts
from ui.screens import render_failure, render_findings, render_result


def _render_dashboard(handoff: DashboardHandoff, preview_enabled: bool) -> None:
    context = handoff.context
    st.caption(
        f"Handoff `{context.handoff_id[:24]}…` · plan `{context.plan_id}` · "
        f"QA `{context.qa_report_id}`"
    )
    for warning in context.warnings:
        st.warning(warning)
    frame = handoff.gold_frame()
    render_charts({"spec": handoff.dashboard_spec(), "data": frame})
    if context.authored_questions or context.selected_questions:
        st.markdown("#### Questions carried into this view")
        for question in context.authored_questions:
            st.markdown(f"- {question}")
        for suggested in context.selected_questions:
            st.markdown(f"- {suggested.question}")
    if preview_enabled:
        st.markdown("#### Local gold preview")
        st.caption("Presentation only; never part of evidence or the manifest.")
        st.dataframe(frame.head(10), use_container_width=True)


def render(controller: Any, state: Any) -> None:
    st.header("7 · Dashboard")
    session = controller.session

    handoff = controller.build_dashboard_handoff()
    if isinstance(handoff, DashboardHandoff):
        st.caption(
            "Built from the verified gold table only. Every chart is drawn "
            "locally from a deterministic specification."
        )
        _render_dashboard(handoff, session.preview_enabled)
    else:
        render_failure(handoff)

    render_findings(session.findings)
    render_result(ui_state.last_result(state))
