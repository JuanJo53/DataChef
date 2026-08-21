from __future__ import annotations

import pandas as pd
import pytest

from datachef.application.question_charts import (
    ChartAggregation,
    ChartCategoryPolicy,
    ChartCategoryTransform,
    ChartRanking,
    QuestionChartType,
    QuestionResolutionStatus,
    QuestionSource,
    compile_question_charts,
)
from datachef.contracts import QuestionKind, SuggestedQuestion


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observed_on": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "region": ["North", "South"],
            "sales": [10.0, 20.0],
            "cost": [4.0, 7.0],
            "private_value": ["CELL_CANARY_ONE", "CELL_CANARY_TWO"],
        }
    )


@pytest.mark.parametrize(
    ("question", "chart_type", "aggregation"),
    [
        ("Show the trend of sales over observed_on", QuestionChartType.LINE, None),
        ("Show the distribution of sales", QuestionChartType.HISTOGRAM, None),
        ("Show the relationship between sales and cost", QuestionChartType.SCATTER, None),
        ("Compare sales across region", QuestionChartType.BOX, None),
        ("Show total sales by region", QuestionChartType.BAR, ChartAggregation.SUM),
        ("Show average sales by region", QuestionChartType.BAR, ChartAggregation.MEAN),
        ("Show median sales by region", QuestionChartType.BAR, ChartAggregation.MEDIAN),
        ("Show count by region", QuestionChartType.BAR, ChartAggregation.COUNT),
    ],
)
def test_authored_questions_compile_to_closed_grounded_specs(
    question, chart_type, aggregation
) -> None:
    result = compile_question_charts(_frame(), (question,), ())

    assert len(result) == 1
    assert result[0].status is QuestionResolutionStatus.RESOLVED
    assert result[0].chart is not None
    assert result[0].chart.chart_type is chart_type
    assert result[0].chart.aggregation is aggregation


@pytest.mark.parametrize(
    ("kind", "columns", "chart_type"),
    [
        (QuestionKind.TREND, ("observed_on", "sales"), QuestionChartType.LINE),
        (QuestionKind.DISTRIBUTION, ("sales",), QuestionChartType.HISTOGRAM),
        (QuestionKind.RELATIONSHIP, ("sales", "cost"), QuestionChartType.SCATTER),
        (QuestionKind.CATEGORY_COMPARISON, ("region", "sales"), QuestionChartType.BOX),
    ],
)
def test_selected_suggestions_use_typed_question_kind(kind, columns, chart_type) -> None:
    suggestion = SuggestedQuestion(
        question_id="suggestion-1",
        kind=kind,
        question="Locally authored display text.",
        relevant_columns=columns,
        rationale="Deterministic schema roles support it.",
        confidence=0.9,
    )

    result = compile_question_charts(_frame(), (), (suggestion,))

    assert result[0].status is QuestionResolutionStatus.RESOLVED
    assert result[0].chart.chart_type is chart_type


def test_ambiguous_and_missing_column_questions_are_explicit() -> None:
    result = compile_question_charts(
        _frame(),
        (
            "What should I know?",
            "Show the distribution of missing_measure",
            "Compare sales and cost and region somehow",
        ),
        (),
    )

    assert [item.status for item in result] == [
        QuestionResolutionStatus.QUESTION_NEEDS_INPUT,
        QuestionResolutionStatus.QUESTION_UNSUPPORTED,
        QuestionResolutionStatus.QUESTION_NEEDS_INPUT,
    ]
    assert all(item.chart is None for item in result)


def test_compilation_is_deterministic_and_serialized_specs_contain_no_cell_values() -> None:
    questions = ("Show the relationship between sales and cost",)
    first = compile_question_charts(_frame(), questions, ())
    second = compile_question_charts(_frame(), questions, ())

    assert first == second
    serialized = "".join(item.model_dump_json() for item in first)
    assert "CELL_CANARY" not in serialized
    assert "private_value" not in serialized
    assert questions[0] not in serialized


def test_unsupported_suggestion_kind_remains_visible() -> None:
    suggestion = SuggestedQuestion(
        question_id="missingness-1",
        kind=QuestionKind.MISSINGNESS,
        question="Where is data missing?",
        relevant_columns=("sales",),
        rationale="A missingness question.",
        confidence=0.8,
    )

    result = compile_question_charts(_frame(), (), (suggestion,))

    assert result[0].status is QuestionResolutionStatus.QUESTION_UNSUPPORTED
    assert result[0].reason_code == "QUESTION_KIND_UNSUPPORTED"


@pytest.mark.parametrize(
    "question",
    (
        "What titles have the highest prices?",
        "Which titles have the highest prices?",
        "What are the most expensive titles?",
        "Which products cost the most?",
        "Top titles by price",
    ),
)
def test_highest_price_questions_compile_to_closed_descending_top_n(question) -> None:
    frame = pd.DataFrame(
        {"title": ["A", "B"], "price": [10.0, 20.0], "stars": [4.0, 5.0]}
    )

    result = compile_question_charts(frame, (question,), ())

    resolution = result[0]
    assert resolution.source is QuestionSource.AUTHORED
    assert resolution.status is QuestionResolutionStatus.RESOLVED
    assert resolution.chart is not None
    assert resolution.chart.chart_type is QuestionChartType.BAR
    assert resolution.chart.x_column == "title"
    assert resolution.chart.y_column == "price"
    assert resolution.chart.aggregation is ChartAggregation.MAX
    assert resolution.chart.ranking is ChartRanking.DESCENDING
    assert resolution.chart.limit == 10


def test_authored_relationship_and_automatic_suggestion_keep_separate_identity() -> None:
    frame = pd.DataFrame(
        {"title": ["A", "B"], "price": [10.0, 20.0], "stars": [4.0, 5.0]}
    )
    suggestion = SuggestedQuestion(
        question_id="automatic-missingness",
        kind=QuestionKind.MISSINGNESS,
        question="Which columns have the largest missing-data rates?",
        relevant_columns=("price",),
        rationale="Aggregate missingness evidence.",
        confidence=0.9,
    )

    result = compile_question_charts(
        frame,
        (
            "What titles have the highest prices?",
            "What's the relationship between price and stars?",
        ),
        (suggestion,),
    )

    assert [item.source for item in result] == [
        QuestionSource.AUTHORED,
        QuestionSource.AUTHORED,
        QuestionSource.AUTOMATIC,
    ]
    assert [item.source_index for item in result] == [0, 1, 0]
    scatter = result[1].chart
    assert scatter is not None
    assert scatter.chart_type is QuestionChartType.SCATTER
    assert (scatter.x_column, scatter.y_column) == ("price", "stars")
    assert result[2].reason_code == "QUESTION_KIND_UNSUPPORTED"


@pytest.mark.parametrize(
    ("question", "category", "measure", "limit"),
    [
        ("What stores have the most weekly sales?", "Store", "Weekly_Sales", 10),
        ("What are the top 5 most profitable segments?", "Segment", "Profit", 5),
        ("In what countries do we sell the most by quantity?", "Country", "Quantity", 10),
        ("Which regions have the largest total sales?", "Region", "Sales", 10),
    ],
)
def test_aggregate_ranking_questions_compile_with_closed_sum_semantics(
    question, category, measure, limit
) -> None:
    frame = pd.DataFrame(
        {
            "Store": [1, 2],
            "Weekly_Sales": [10.0, 20.0],
            "Segment": ["A", "B"],
            "Profit": [1.0, 2.0],
            "Country": ["X", "Y"],
            "Quantity": [3.0, 4.0],
            "Region": ["N", "S"],
            "Sales": [5.0, 6.0],
        }
    )

    resolution = compile_question_charts(frame, (question,), ())[0]

    assert resolution.status is QuestionResolutionStatus.RESOLVED
    assert resolution.chart is not None
    assert resolution.chart.x_column == category
    assert resolution.chart.y_column == measure
    assert resolution.chart.aggregation is ChartAggregation.SUM
    assert resolution.chart.ranking is ChartRanking.DESCENDING
    assert resolution.chart.limit == limit


def test_day_of_week_ranking_uses_a_closed_derived_category() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-08-17", "2026-08-18", "2026-08-24"]),
            "Weekly_Sales": [10.0, 20.0, 30.0],
            "Store": [1, 1, 2],
        }
    )

    resolution = compile_question_charts(
        frame,
        ("On what day of the week do we have the most sales?",),
        (),
    )[0]

    assert resolution.status is QuestionResolutionStatus.RESOLVED
    assert resolution.chart is not None
    assert resolution.chart.x_column == "Date"
    assert resolution.chart.y_column == "Weekly_Sales"
    assert resolution.chart.category_transform is ChartCategoryTransform.DAY_OF_WEEK
    assert resolution.chart.aggregation is ChartAggregation.SUM
    assert resolution.chart.ranking is ChartRanking.DESCENDING


def test_top_five_title_ranking_has_exact_limit_and_placeholder_policy() -> None:
    frame = pd.DataFrame(
        {
            "title": ["A", "B"],
            "stars": [5.0, 4.0],
        }
    )

    resolution = compile_question_charts(
        frame,
        ("What are the top 5 titles by stars ratings that aren't unknown or nulls?",),
        (),
    )[0]

    assert resolution.status is QuestionResolutionStatus.RESOLVED
    assert resolution.chart is not None
    assert resolution.chart.limit == 5
    assert resolution.chart.category_policy is ChartCategoryPolicy.EXCLUDE_PLACEHOLDERS


def test_invalid_zero_ranking_limit_fails_closed_without_compiler_exception() -> None:
    frame = pd.DataFrame({"title": ["A"], "stars": [5.0]})

    resolution = compile_question_charts(
        frame,
        ("Top 0 titles by stars",),
        (),
    )[0]

    assert resolution.status is QuestionResolutionStatus.QUESTION_NEEDS_INPUT
    assert resolution.reason_code == "QUESTION_RANKING_LIMIT_INVALID"


def test_dropped_column_and_natural_alias_failures_are_distinguished() -> None:
    frame = pd.DataFrame({"Store": [1], "Weekly_Sales": [10.0], "Date": ["2026-01-01"]})

    available = compile_question_charts(
        frame,
        ("What stores have the most weekly sales?",),
        (),
        unavailable_columns=("CPI", "Unemployment"),
    )[0]
    dropped = compile_question_charts(
        frame,
        ("Which stores have the highest CPI?",),
        (),
        unavailable_columns=("CPI", "Unemployment"),
    )[0]

    assert available.status is QuestionResolutionStatus.RESOLVED
    assert dropped.status is QuestionResolutionStatus.QUESTION_UNSUPPORTED
    assert dropped.reason_code == "QUESTION_COLUMN_UNAVAILABLE"


def test_ranking_renderer_enforces_exact_top_n_filters_placeholders_and_orients_long_labels() -> None:
    from ui.screens.dashboard import _bar_plot_data

    frame = pd.DataFrame(
        {
            "title": [
                "Unknown",
                "",
                None,
                "A very long product title that exceeds forty characters 1",
                "A very long product title that exceeds forty characters 2",
                "A very long product title that exceeds forty characters 3",
                "A very long product title that exceeds forty characters 4",
                "A very long product title that exceeds forty characters 5",
                "A very long product title that exceeds forty characters 6",
            ],
            "stars": [9.0, 9.0, 9.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    resolution = compile_question_charts(
        frame,
        ("Top 5 titles by stars not including unknown or nulls",),
        (),
    )[0]
    assert resolution.chart is not None

    plotted, value_column, horizontal = _bar_plot_data(frame, resolution.chart)

    assert len(plotted) == 5
    assert value_column == "stars"
    assert horizontal is True
    assert not plotted["title"].isna().any()
    assert "unknown" not in {str(value).strip().casefold() for value in plotted["title"]}
