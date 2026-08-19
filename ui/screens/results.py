"""Results stage: downloads and dashboard, both only as the controller allows.

Nothing here decides that a run passed. The bundle exists only if
``controller.build_artifacts()`` returns one, and the dashboard exists only if
``controller.build_dashboard_handoff()`` returns one. Any refusal is rendered as
its own sanitized message with no download controls at all.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

import pandas as pd

from datachef.application import ArtifactSet, DashboardHandoff, DashboardSummary
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




def _render_readiness(summary: DashboardSummary) -> None:
    """Top band: did it work, how much moved, is it ready to model.

    Every number is read off the deterministic summary. Nothing is recomputed
    here, so the screen cannot disagree with the QA report it is describing.
    """

    st.markdown("### Data readiness")
    if summary.modelling_ready:
        st.success(summary.readiness_headline)
    else:
        st.warning(summary.readiness_headline)

    rows, columns, missing, duplicates = st.columns(4)
    rows.metric(
        "Rows",
        f"{summary.rows_after:,}",
        delta=f"-{summary.rows_removed:,}" if summary.rows_removed else "unchanged",
        delta_color="off",
    )
    columns.metric(
        "Columns",
        f"{summary.columns_after:,}",
        delta=(
            f"-{len(summary.removed_columns)}"
            if summary.removed_columns
            else "unchanged"
        ),
        delta_color="off",
    )
    missing.metric(
        "Missing values",
        f"{summary.nulls_after_total:,}",
        delta=f"-{summary.nulls_filled:,}" if summary.nulls_filled else "unchanged",
        delta_color="off",
    )
    duplicates.metric(
        "Duplicate rows",
        f"{summary.duplicate_rows_after:,}",
        delta=(
            f"-{summary.duplicate_rows_before - summary.duplicate_rows_after:,}"
            if summary.duplicate_rows_before > summary.duplicate_rows_after
            else "unchanged"
        ),
        delta_color="off",
    )

    if summary.target_column is None:
        st.caption(
            "No target column was named in the objective, so modelling readiness "
            "is reported for the table as a whole."
        )
    elif summary.target_is_usable:
        st.caption(
            f"Target `{summary.target_column}` survived the plan and carries no "
            "missing values."
        )
    else:
        st.caption(
            f"Target `{summary.target_column}` is not usable yet: it was removed "
            "or still carries missing values."
        )


def _render_change_detail(summary: DashboardSummary) -> None:
    """Middle band: what happened to each column, and which operations did it."""

    st.markdown("### What changed")
    missingness = pd.DataFrame(
        [
            {
                "column": item.column,
                "dtype": item.dtype,
                "missing before": item.nulls_before,
                "missing after": item.nulls_after,
                "distinct": item.distinct_count,
                "target": "yes" if item.is_target else "",
            }
            for item in summary.columns
        ]
    )
    st.caption("Missing values per surviving column, before and after the plan.")
    st.dataframe(missingness, use_container_width=True, hide_index=True)

    if summary.operations:
        st.caption("Operations the human approved, in the order they ran.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "operation": item.operation_type.value,
                        "columns": ", ".join(item.target_columns) or "whole row",
                        "detail": item.detail,
                        "rows": f"{item.rows_before} -> {item.rows_after}",
                    }
                    for item in summary.operations
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Removed columns**")
        if summary.removed_columns:
            for column in summary.removed_columns:
                st.markdown(f"- `{column}`")
        else:
            st.caption("None; every column survived.")
    with right:
        st.markdown("**Rows and duplicates**")
        st.markdown(
            f"- Rows {summary.rows_before:,} to {summary.rows_after:,} "
            f"({summary.row_loss_pct:.2f}% removed)"
        )
        st.markdown(
            f"- Duplicate rows {summary.duplicate_rows_before:,} to "
            f"{summary.duplicate_rows_after:,}"
        )
        if summary.duplicate_keys_before is not None:
            st.markdown(
                f"- Duplicate keys {summary.duplicate_keys_before:,} to "
                f"{summary.duplicate_keys_after:,}"
            )

    for message in summary.unresolved_issues:
        st.warning(message)


def _render_summary(summary: DashboardSummary) -> None:
    _render_readiness(summary)
    _render_change_detail(summary)


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

    summary = controller.build_dashboard_summary()
    if isinstance(summary, DashboardSummary):
        _render_summary(summary)

    handoff = controller.build_dashboard_handoff()
    if isinstance(handoff, DashboardHandoff):
        _render_dashboard(handoff, session.preview_enabled)
    else:
        render_failure(handoff)

    render_findings(session.findings)
    render_result(ui_state.last_result(state))
