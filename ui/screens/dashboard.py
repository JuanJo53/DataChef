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

import pandas as pd
import plotly.express as px
import streamlit as st

from datachef.application import (
    ChartAggregation,
    ChartCategoryPolicy,
    ChartCategoryTransform,
    ChartRanking,
    DashboardHandoff,
    QuestionChartType,
    QuestionResolutionStatus,
    QuestionSource,
)
from ui import state as ui_state
from ui.charts import render_charts
from ui.screens import render_failure, render_findings, render_result


_QUESTION_FAILURE_MESSAGES = {
    "QUESTION_COLUMN_UNAVAILABLE": "A referenced column is not available in verified gold.",
    "QUESTION_AGGREGATION_AMBIGUOUS": "Name one category and the measure to aggregate.",
    "QUESTION_KIND_UNSUPPORTED": "This question type has no deterministic chart mapping yet.",
    "QUESTION_COLUMNS_AMBIGUOUS": "Name the exact columns the chart should compare.",
    "QUESTION_RANKING_LIMIT_INVALID": "Use a top-N value of at least one.",
    "QUESTION_INTENT_AMBIGUOUS": "State a trend, distribution, relationship, comparison, or aggregation.",
}

_CATEGORY_PLACEHOLDERS = frozenset({"unknown", "null", "n/a", "na"})


def _bar_plot_data(frame: Any, chart: Any) -> tuple[Any, str, bool]:
    """Build one local aggregate view without mutating verified gold."""

    columns = [chart.x_column]
    if chart.y_column is not None:
        columns.append(chart.y_column)
    working = frame.loc[:, columns].copy(deep=True)
    category_column = chart.x_column
    if chart.category_transform is ChartCategoryTransform.DAY_OF_WEEK:
        category_column = "__datachef_weekday__"
        working[category_column] = pd.to_datetime(
            working[chart.x_column],
            errors="coerce",
            format="mixed",
            dayfirst=True,
        ).dt.day_name()

    if chart.category_policy is ChartCategoryPolicy.EXCLUDE_PLACEHOLDERS:
        category = working[category_column]
        normalized = category.astype("string").str.strip().str.casefold()
        keep = category.notna() & normalized.ne("") & ~normalized.isin(
            _CATEGORY_PLACEHOLDERS
        )
        working = working.loc[keep].copy()

    if chart.aggregation is ChartAggregation.COUNT:
        plotted = working[category_column].value_counts(dropna=False).reset_index()
        plotted.columns = [category_column, "count"]
        value_column = "count"
    else:
        aggregation = {
            ChartAggregation.SUM: "sum",
            ChartAggregation.MEAN: "mean",
            ChartAggregation.MEDIAN: "median",
            ChartAggregation.MAX: "max",
        }[chart.aggregation]
        plotted = (
            working.groupby(category_column, dropna=False)[chart.y_column]
            .agg(aggregation)
            .reset_index()
        )
        value_column = chart.y_column
    if chart.ranking is ChartRanking.DESCENDING:
        plotted = plotted.sort_values(
            value_column, ascending=False, kind="mergesort"
        ).head(chart.limit)
    lengths = plotted[category_column].dropna().astype(str).str.len()
    horizontal = bool(len(lengths) and int(lengths.max()) > 40)
    return plotted, value_column, horizontal


def _render_resolutions(
    resolutions: tuple[Any, ...],
    display_questions: tuple[str, ...],
    frame: Any,
) -> None:
    for resolution in resolutions:
        if resolution.status is not QuestionResolutionStatus.RESOLVED:
            label = (
                "Question"
                if resolution.source is QuestionSource.AUTHORED
                else "Recommendation"
            )
            st.warning(
                f"{label} {resolution.source_index + 1} could not be answered: "
                + _QUESTION_FAILURE_MESSAGES.get(
                    resolution.reason_code,
                    "The question is outside the deterministic chart vocabulary.",
                )
            )
            if resolution.source_index < len(display_questions):
                st.text(display_questions[resolution.source_index])
            continue

        chart = resolution.chart
        assert chart is not None
        st.markdown(f"#### {chart.title}")
        if chart.chart_type is QuestionChartType.HISTOGRAM:
            figure = px.histogram(frame, x=chart.x_column)
        elif chart.chart_type is QuestionChartType.SCATTER:
            figure = px.scatter(frame, x=chart.x_column, y=chart.y_column)
        elif chart.chart_type is QuestionChartType.BOX:
            figure = px.box(frame, x=chart.x_column, y=chart.y_column)
        elif chart.chart_type is QuestionChartType.LINE:
            ordered = frame.sort_values(chart.x_column)
            figure = px.line(
                ordered, x=chart.x_column, y=chart.y_column, markers=True
            )
        else:
            assert chart.chart_type is QuestionChartType.BAR
            plotted, value_column, horizontal = _bar_plot_data(frame, chart)
            category_column = (
                "__datachef_weekday__"
                if chart.category_transform is ChartCategoryTransform.DAY_OF_WEEK
                else chart.x_column
            )
            if horizontal:
                figure = px.bar(
                    plotted,
                    x=value_column,
                    y=category_column,
                    orientation="h",
                )
            else:
                figure = px.bar(
                    plotted,
                    x=category_column,
                    y=value_column,
                )
        st.plotly_chart(
            figure,
            use_container_width=True,
            key=f"question_{chart.spec_id}",
        )


def _render_question_charts(handoff: DashboardHandoff, frame: Any) -> None:
    context = handoff.context
    authored = context.authored_question_resolutions
    if authored:
        st.markdown("### Charts answering your questions")
        _render_resolutions(authored, context.authored_questions, frame)

    st.markdown("### Other recommended charts / diagnostics")
    recommended = context.recommended_question_resolutions
    if recommended:
        _render_resolutions(
            recommended,
            tuple(question.question for question in context.selected_questions),
            frame,
        )


def _render_dashboard(handoff: DashboardHandoff, preview_enabled: bool) -> None:
    context = handoff.context
    st.caption(
        f"Handoff `{context.handoff_id[:24]}…` · plan `{context.plan_id}` · "
        f"QA `{context.qa_report_id}`"
    )
    for warning in context.warnings:
        st.warning(warning)
    frame = handoff.gold_frame()
    _render_question_charts(handoff, frame)
    render_charts({"spec": handoff.dashboard_spec(), "data": frame})
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
