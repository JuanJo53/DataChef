"""Diagnostics stage: read the deterministic diagnosis back to the user.

This screen computes nothing. Every number on it was produced by
``DataChefController.diagnose()`` and stored as evidence, and this module only
chooses how to arrange it. A richer diagnostic view is somebody else's work;
this one exists so the diagnosis has a place of its own before the objective is
written, instead of arriving as a footnote on the upload screen.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

import pandas as pd

from datachef.application import ScreenId
from ui import state as ui_state
from ui.screens import render_diagnosis, render_findings, render_result


def _render_shape(report: Any) -> None:
    """The four numbers that describe the table as it arrived."""

    identity = report.dataset_identity
    evidence = report.legacy_evidence
    rows, columns, complete, duplicates = st.columns(4)
    rows.metric("Rows", f"{identity.row_count:,}")
    columns.metric("Columns", f"{identity.column_count:,}")
    complete.metric("Complete values", f"{evidence.completeness_pct:.2f}%")
    duplicates.metric("Duplicate rows", f"{report.duplicate_row_count:,}")

    unique, key = st.columns(2)
    unique.metric("Unique rows", f"{evidence.uniqueness_pct:.2f}%")
    if evidence.suggested_primary_key:
        key.metric("Suggested key", evidence.suggested_primary_key)
    else:
        key.metric("Suggested key", "none found")


def _render_columns(report: Any) -> None:
    """Per-column completeness, straight off the recorded profiles."""

    st.markdown("### Every column as it arrived")
    st.caption(
        "Missing counts, distinct values, and the type each column was read "
        "as. Nothing here has been changed yet."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "column": profile.name,
                    "type": profile.dtype,
                    "missing": profile.null_count,
                    "missing %": round(profile.null_pct, 2),
                    "distinct": profile.unique_count,
                    "zeros": profile.zero_count,
                    "key candidate": "yes" if profile.is_primary_key_candidate else "",
                }
                for profile in report.column_profiles
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_key_duplicates(report: Any) -> None:
    if not report.key_duplicate_metrics:
        return
    st.markdown("### Duplicate keys")
    for metric in report.key_duplicate_metrics:
        keys = ", ".join(metric.key_columns)
        st.markdown(
            f"- `{keys}` — {metric.duplicate_row_count:,} duplicate row(s), "
            f"{metric.null_key_row_count:,} row(s) with a missing key"
        )


def render(controller: Any, state: Any) -> None:
    st.header("2 · Diagnostics")
    session = controller.session

    if session.source is None:
        st.info("Upload a dataset first.")
        render_result(ui_state.last_result(state))
        return

    report = session.display_diagnostic_report
    if report is None:
        st.info(
            "This dataset has not been diagnosed yet. Run the deterministic "
            "diagnosis on the upload screen."
        )
        render_findings(session.findings)
        render_result(ui_state.last_result(state))
        return

    st.caption(
        "A deterministic read of the file you uploaded. No plan has been made "
        "and nothing has been changed."
    )
    _render_shape(report)
    render_diagnosis(session)
    _render_columns(report)
    _render_key_duplicates(report)

    st.markdown("---")
    if st.button(
        "Continue to the objective",
        key=ui_state.CONTINUE_TO_INTENT_WIDGET,
        type="primary",
    ):
        controller.navigate(ScreenId.INTENT)
        st.rerun()

    render_findings(session.findings)
    render_result(ui_state.last_result(state))
