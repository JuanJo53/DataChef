"""Results stage: downloads and dashboard, both only as the controller allows.

Nothing here decides that a run passed. The bundle exists only if
``controller.build_artifacts()`` returns one, and the dashboard exists only if
``controller.build_dashboard_handoff()`` returns one. Any refusal is rendered as
its own sanitized message with no download controls at all.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from datachef.application import ArtifactSet, DashboardHandoff
from ui import state as ui_state
from ui.charts import render_charts
from ui.screens import render_failure, render_findings, render_result


_DOWNLOAD_LABELS = {
    "CLEANED_CSV": "Download cleaned CSV",
    "CLEANED_PARQUET": "Download cleaned Parquet",
    "TRANSFORMATION_PLAN_JSON": "Download transformation plan",
    "QA_REPORT_JSON": "Download QA report",
    "EXECUTION_CHANGE_LOG_JSON": "Download execution change log",
    "PIPELINE_SCRIPT_PY": "Download reusable pipeline script",
    "MANIFEST_JSON": "Download manifest",
}


def _render_downloads(bundle: ArtifactSet) -> None:
    st.markdown("### Download the verified bundle")
    st.caption(
        "Every file is served exactly as the application produced it. The "
        "manifest records a SHA-256 for each of the other six artifacts."
    )
    for artifact in bundle.artifacts():
        kind = artifact.kind.value
        left, right = st.columns([2, 3])
        left.download_button(
            _DOWNLOAD_LABELS.get(kind, kind),
            data=artifact.content,
            file_name=artifact.filename,
            mime=artifact.media_type,
            key=f"datachef_w_download_{kind}",
            use_container_width=True,
        )
        right.caption(
            f"`{artifact.filename}` · {artifact.media_type} · "
            f"{artifact.byte_size} bytes · sha256 `{artifact.sha256[:16]}…`"
        )


def _render_dashboard(handoff: DashboardHandoff, preview_enabled: bool) -> None:
    context = handoff.context
    st.markdown("### Dashboard")
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
    st.header("6 · Results")
    session = controller.session

    bundle = controller.build_artifacts()
    if isinstance(bundle, ArtifactSet):
        st.success(
            "Quality assurance passed and the gold table matched its execution "
            "evidence, so the download bundle is available."
        )
        _render_downloads(bundle)
    else:
        render_failure(bundle)

    handoff = controller.build_dashboard_handoff()
    if isinstance(handoff, DashboardHandoff):
        _render_dashboard(handoff, session.preview_enabled)
    else:
        render_failure(handoff)

    render_findings(session.findings)
    render_result(ui_state.last_result(state))
