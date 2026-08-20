"""Diagnostics stage: read the deterministic diagnosis back to the user.

This screen computes nothing. Every number on it was produced by
``DataChefController.diagnose()`` and stored as evidence, and this module only
chooses how to arrange it. A richer diagnostic view is somebody else's work;
this one exists so the diagnosis has a place of its own before the objective is
written, instead of arriving as a footnote on the upload screen.
"""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from datachef.application import ScreenId
from ui import state as ui_state
from ui.screens import render_findings, render_result


def _metric_card(
    label: str,
    value: str,
    description: str,
    tone: str = "green",
) -> None:
    """Render one Data Health metric card."""

    html = (
        f'<div class="diagnostic-metric-card metric-{tone}">'
        f'<div class="diagnostic-metric-label">'
        f'<span class="metric-dot"></span>'
        f'{escape(label)}'
        f'</div>'
        f'<div class="diagnostic-metric-value">'
        f'{escape(str(value))}'
        f'</div>'
        f'</div>'
        f'<div class="diagnostic-metric-description">'
        f'{escape(description)}'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def _render_data_health(report: Any) -> None:
    """Render the main Data Health metrics."""

    evidence = report.legacy_evidence
    profiles = report.column_profiles

    columns_with_nulls = sum(
        1 for profile in profiles if profile.null_count > 0
    )

    pii_columns = sum(
        1 for profile in profiles if profile.possible_pii
    )

    st.markdown("## Data health")

    cols = st.columns(5, gap="small")

    with cols[0]:
        _metric_card(
            "Health score",
            f"{evidence.health_score}/100 ({evidence.health_grade})",
            "Weighted completeness + uniqueness.",
            "green",
        )

    with cols[1]:
        _metric_card(
            "Completeness",
            f"{evidence.completeness_pct:.1f}%",
            "Share of non-null cells.",
            "blue",
        )

    with cols[2]:
        _metric_card(
            "Duplicate rows",
            f"{report.duplicate_row_count:,}",
            "Fully duplicated rows.",
            "purple",
        )

    with cols[3]:
        _metric_card(
            "Cols with nulls",
            f"{columns_with_nulls:,}",
            "Columns containing missing values.",
            "red",
        )

    with cols[4]:
        _metric_card(
            "PII columns",
            f"{pii_columns:,}",
            "Potential personally identifiable information.",
            "green",
        )


def _get_affected_rows(issue: Any) -> int | float | str:
    """Try to obtain the affected row count from diagnostic evidence."""

    for item in issue.evidence:
        metric_name = item.metric.lower()

        if (
            "row" in metric_name
            or "count" in metric_name
            or "duplicate" in metric_name
            or "null" in metric_name
        ):
            return item.value

    return "—"


def _get_percentage(issue: Any) -> str | None:
    """Find a percentage value in the issue evidence if one exists."""

    for item in issue.evidence:
        metric_name = item.metric.lower()

        if (
            "pct" in metric_name
            or "percent" in metric_name
            or item.unit == "%"
        ):
            try:
                return f"{float(item.value):.1f}%"
            except (TypeError, ValueError):
                return str(item.value)

    return None


def _suggestion_text(issue: Any) -> str:
    """Human-friendly text for the suggested operation."""

    if issue.suggested_operation is None:
        return "Review the affected data"

    mapping = {
        "DROP_DUPLICATES": "Drop duplicates",
        "DEDUPLICATE_BY_KEYS": "Drop duplicate keys",
        "IMPUTE": "Fill with default / review missing values",
        "CAST_COLUMN": "Convert column type",
        "NORMALIZE_NUMERIC_TEXT": "Normalize numeric values",
        "DROP_COLUMN": "Remove column",
        "MASK_PII": "Mask sensitive values",
    }

    value = issue.suggested_operation.value

    return mapping.get(
        value,
        value.replace("_", " ").title(),
    )


def _severity_class(severity: str) -> str:
    severity = severity.lower()

    if severity == "high":
        return "severity-high"

    if severity == "medium":
        return "severity-medium"

    return "severity-low"


def _render_issue_card(issue: Any) -> None:
    """Render a single detected issue."""

    severity = issue.severity.value
    severity_css = _severity_class(severity)

    affected_rows = _get_affected_rows(issue)
    percentage = _get_percentage(issue)

    columns = ", ".join(issue.affected_columns)

    title = issue.title
    if columns and columns not in title:
        title = f"{title} in '{columns}'"

    explanation = issue.explanation
    suggestion = _suggestion_text(issue)

    percentage_text = (
        f"{escape(percentage)} of rows are affected. "
        if percentage
        else ""
    )

    html = (
        '<div class="diagnostic-issue-card">'
        '<div class="issue-main-line">'
        f'<span class="issue-title">{escape(title)}</span>'
        '<span class="issue-separator">·</span>'
        '<span class="issue-label">severity:</span>'
        f'<span class="severity-badge {severity_css}">'
        f'{escape(severity.title())}'
        '</span>'
        '<span class="issue-separator">·</span>'
        '<span class="issue-label">affected rows:</span>'
        f'<span class="issue-row-count">{escape(str(affected_rows))}</span>'
        '</div>'
        '<div class="issue-description">'
        f'{percentage_text}'
        f'{escape(explanation)}'
        '<span class="issue-arrow"> → </span>'
        f'<span class="issue-suggestion">'
        f'suggested: {escape(suggestion)}'
        '</span>'
        '</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def _render_issues(report: Any) -> None:
    """Render all diagnostic issues."""

    issues = report.issues

    st.markdown(f"## Detected issues ({len(issues)})")

    if not issues:
        st.markdown(
            """
            <div class="diagnostic-success-card">
                <span>●</span>
                No diagnostic issues were detected.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for issue in issues:
        _render_issue_card(issue)


def _render_columns(report: Any) -> None:
    """Render detailed per-column profile."""

    with st.expander("View column details"):
        st.caption(
            "Missing values, distinct values, inferred types and key candidates."
        )

        dataframe = pd.DataFrame(
            [
                {
                    "Column": profile.name,
                    "Type": profile.dtype,
                    "Missing": profile.null_count,
                    "Missing %": round(profile.null_pct, 2),
                    "Distinct": profile.unique_count,
                    "Zeros": profile.zero_count,
                    "PII": "Yes" if profile.possible_pii else "",
                    "Key candidate": (
                        "Yes"
                        if profile.is_primary_key_candidate
                        else ""
                    ),
                }
                for profile in report.column_profiles
            ]
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )


def _render_key_duplicates(report: Any) -> None:
    """Render duplicate-key information only when available."""

    if not report.key_duplicate_metrics:
        return

    with st.expander("View duplicate key details"):
        for metric in report.key_duplicate_metrics:
            keys = ", ".join(metric.key_columns)

            st.markdown(
                f"""
                **{keys}**

                Duplicate rows: `{metric.duplicate_row_count:,}`  
                Rows with missing key: `{metric.null_key_row_count:,}`
                """
            )


def render(controller: Any, state: Any) -> None:
    """Render Diagnostics screen."""

    session = controller.session

    if session.source is None:
        st.info("Upload a dataset first.")
        render_result(ui_state.last_result(state))
        return

    report = session.display_diagnostic_report

    if report is None:
        st.info(
            "This dataset has not been diagnosed yet. "
            "Run the deterministic diagnosis on the upload screen."
        )
        render_findings(session.findings)
        render_result(ui_state.last_result(state))
        return

    # --------------------------------------------------
    # Main diagnostics UI
    # --------------------------------------------------

    _render_data_health(report)

    st.markdown(
        '<div class="diagnostic-section-space"></div>',
        unsafe_allow_html=True,
    )

    _render_issues(report)

    st.markdown(
        '<div class="diagnostic-section-space"></div>',
        unsafe_allow_html=True,
    )

    _render_columns(report)
    _render_key_duplicates(report)

    st.markdown(
        '<div class="diagnostic-bottom-space"></div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # Navigation
    # --------------------------------------------------

    _, button_column = st.columns([4, 1])

    with button_column:
        if st.button(
            "Continue to objective →",
            key=ui_state.CONTINUE_TO_INTENT_WIDGET,
            type="primary",
            use_container_width=True,
        ):
            controller.navigate(ScreenId.INTENT)
            st.rerun()

    render_findings(session.findings)
    render_result(ui_state.last_result(state))