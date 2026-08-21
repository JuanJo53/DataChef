"""Deterministic, local question-to-chart compilation for verified gold.

The compiler inspects schema metadata only. Free-form question text is used
locally for exact column and closed intent matching, but is never copied into
the serialized resolution or chart contracts.
"""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
import json
import re

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from pydantic import Field, model_validator

from datachef.application.models import StrictApplicationModel
from datachef.contracts import QuestionKind, SuggestedQuestion


class QuestionResolutionStatus(StrEnum):
    RESOLVED = "RESOLVED"
    QUESTION_NEEDS_INPUT = "QUESTION_NEEDS_INPUT"
    QUESTION_UNSUPPORTED = "QUESTION_UNSUPPORTED"


class QuestionSource(StrEnum):
    AUTHORED = "AUTHORED"
    AUTOMATIC = "AUTOMATIC"


class QuestionChartType(StrEnum):
    LINE = "LINE"
    HISTOGRAM = "HISTOGRAM"
    SCATTER = "SCATTER"
    BOX = "BOX"
    BAR = "BAR"


class ChartAggregation(StrEnum):
    SUM = "SUM"
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    COUNT = "COUNT"
    MAX = "MAX"


class ChartRanking(StrEnum):
    DESCENDING = "DESCENDING"


class ChartCategoryTransform(StrEnum):
    DAY_OF_WEEK = "DAY_OF_WEEK"


class ChartCategoryPolicy(StrEnum):
    EXCLUDE_PLACEHOLDERS = "EXCLUDE_PLACEHOLDERS"


class QuestionChartSpec(StrictApplicationModel):
    spec_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    chart_type: QuestionChartType
    x_column: str = Field(min_length=1)
    y_column: str | None = Field(default=None, min_length=1)
    aggregation: ChartAggregation | None = None
    ranking: ChartRanking | None = None
    limit: int | None = Field(default=None, ge=1, le=50)
    category_transform: ChartCategoryTransform | None = None
    category_policy: ChartCategoryPolicy | None = None
    title: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_closed_shape(self) -> "QuestionChartSpec":
        if self.chart_type in {
            QuestionChartType.LINE,
            QuestionChartType.SCATTER,
            QuestionChartType.BOX,
        } and self.y_column is None:
            raise ValueError("this chart type requires a y column")
        if self.chart_type is QuestionChartType.HISTOGRAM and self.y_column is not None:
            raise ValueError("histograms use one numeric column")
        if self.chart_type is QuestionChartType.BAR:
            if self.aggregation is None:
                raise ValueError("bar charts require an explicit aggregation")
            if self.aggregation is not ChartAggregation.COUNT and self.y_column is None:
                raise ValueError("numeric aggregations require a y column")
        elif self.aggregation is not None:
            raise ValueError("only bar charts carry aggregations")
        if (self.ranking is None) != (self.limit is None):
            raise ValueError("ranking and limit must be supplied together")
        if self.ranking is not None and (
            self.chart_type is not QuestionChartType.BAR
            or self.aggregation is None
        ):
            raise ValueError("ranking is limited to closed aggregate bar charts")
        if (
            self.category_transform is not None
            and self.chart_type is not QuestionChartType.BAR
        ):
            raise ValueError("category transforms are limited to bar charts")
        if (
            self.category_policy is not None
            and self.chart_type is not QuestionChartType.BAR
        ):
            raise ValueError("category policies are limited to bar charts")
        return self


class QuestionResolution(StrictApplicationModel):
    question_id: str = Field(min_length=1)
    source: QuestionSource
    source_index: int = Field(ge=0)
    status: QuestionResolutionStatus
    reason_code: str = Field(min_length=1)
    chart: QuestionChartSpec | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> "QuestionResolution":
        if (self.status is QuestionResolutionStatus.RESOLVED) != (self.chart is not None):
            raise ValueError("only resolved questions carry a chart")
        return self


_AGGREGATIONS: tuple[tuple[re.Pattern[str], ChartAggregation], ...] = (
    (re.compile(r"\b(?:sum|total)\b", re.IGNORECASE), ChartAggregation.SUM),
    (re.compile(r"\b(?:average|mean)\b", re.IGNORECASE), ChartAggregation.MEAN),
    (re.compile(r"\bmedian\b", re.IGNORECASE), ChartAggregation.MEDIAN),
    (re.compile(r"\bcount\b", re.IGNORECASE), ChartAggregation.COUNT),
)
_TREND = re.compile(r"\b(?:trend|over\s+time|change\s+over)\b", re.IGNORECASE)
_DISTRIBUTION = re.compile(r"\b(?:distribution|histogram)\b", re.IGNORECASE)
_RELATIONSHIP = re.compile(r"\b(?:relationship|correlation|versus|vs\.?\b)\b", re.IGNORECASE)
_COMPARISON = re.compile(r"\b(?:compare|comparison|across|leads?|by)\b", re.IGNORECASE)
_RANKING = re.compile(
    r"\b(?:highest|largest|most\b|top\s+\d*|most\s+expensive|cost\s+the\s+most)\b",
    re.IGNORECASE,
)
_TOP_N = re.compile(r"\btop\s+(\d+)\b", re.IGNORECASE)
_DAY_OF_WEEK = re.compile(r"\bday\s+of\s+the\s+week\b|\bweekday\b", re.IGNORECASE)
_PLACEHOLDER_EXCLUSION = re.compile(
    r"\b(?:unknown|nulls?|n/?a|empty|blank)\b", re.IGNORECASE
)
_TIME_NAME = re.compile(r"(?:^|_)(?:date|time|day|month|year|.*_on)$", re.IGNORECASE)

_PLURALS = {
    "stores": "store",
    "segments": "segment",
    "countries": "country",
    "titles": "title",
    "products": "product",
    "ratings": "rating",
}
_SEMANTIC_COLUMN_TOKENS = {
    "profitable": ("profit",),
    "profitability": ("profit",),
    "products": ("title", "product"),
    "product": ("title", "product"),
    "ratings": ("stars", "rating"),
    "rating": ("stars", "rating"),
    "expensive": ("price", "cost"),
    "cost": ("price", "cost"),
    "sales": ("sales",),
}
_MEASURE_TOKENS = frozenset(
    {"sales", "profit", "quantity", "price", "cost", "stars", "rating", "amount", "revenue"}
)


def _question_id(text: str, index: int) -> str:
    digest = sha256(f"{index}\x00{text}".encode("utf-8")).hexdigest()
    return f"authored-question-{digest[:16]}"


def _identifier_tokens(value: str) -> tuple[str, ...]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return tuple(
        _PLURALS.get(token.casefold(), token.casefold())
        for token in re.findall(r"[0-9A-Za-z]+", spaced)
    )


def _mentioned_columns(
    question: str,
    columns: tuple[str, ...],
    *,
    include_semantic: bool = True,
) -> tuple[str, ...]:
    """Resolve exact, normalized, then unique closed semantic references."""

    words = tuple(
        (match.start(), match.end(), _PLURALS.get(match.group(0).casefold(), match.group(0).casefold()))
        for match in re.finditer(r"[0-9A-Za-z]+", question)
    )
    candidates: list[tuple[int, int, int, str]] = []
    for column in columns:
        literal = re.search(
            r"(?<![0-9A-Za-z_])" + re.escape(column) + r"s?(?![0-9A-Za-z_])",
            question,
            flags=re.IGNORECASE,
        )
        if literal is not None:
            candidates.append((literal.start(), literal.end(), 0, column))
        tokens = _identifier_tokens(column)
        if not tokens:
            continue
        for start in range(0, len(words) - len(tokens) + 1):
            window = tuple(item[2] for item in words[start : start + len(tokens)])
            if window == tokens:
                candidates.append(
                    (words[start][0], words[start + len(tokens) - 1][1], 1, column)
                )

    if include_semantic:
        for index, (_, _, word) in enumerate(words):
            semantic = _SEMANTIC_COLUMN_TOKENS.get(word)
            if semantic is None:
                continue
            matching = [
                column
                for column in columns
                if any(token in _identifier_tokens(column) for token in semantic)
            ]
            exact_semantic = [
                column
                for column in matching
                if len(_identifier_tokens(column)) == 1
            ]
            chosen = exact_semantic if len(exact_semantic) == 1 else matching
            if len(chosen) == 1:
                candidates.append((words[index][0], words[index][1], 2, chosen[0]))

    selected: list[tuple[int, int, str]] = []
    for start, end, level, column in sorted(
        candidates,
        key=lambda item: (item[0], item[2], -(item[1] - item[0])),
    ):
        if any(start < existing_end and end > existing_start for existing_start, existing_end, _ in selected):
            continue
        if column in {item[2] for item in selected}:
            continue
        selected.append((start, end, column))
    return tuple(column for _, _, column in sorted(selected))


def _is_time_column(frame: pd.DataFrame, column: str) -> bool:
    return bool(
        is_datetime64_any_dtype(frame[column].dtype)
        or _TIME_NAME.search(column)
        or {"date", "time", "day", "month", "year"}.intersection(
            _identifier_tokens(column)
        )
    )


def _make_chart(
    question_id: str,
    source: QuestionSource,
    source_index: int,
    chart_type: QuestionChartType,
    x_column: str,
    y_column: str | None = None,
    aggregation: ChartAggregation | None = None,
    ranking: ChartRanking | None = None,
    limit: int | None = None,
    category_transform: ChartCategoryTransform | None = None,
    category_policy: ChartCategoryPolicy | None = None,
) -> QuestionResolution:
    material = json.dumps(
        {
            "question_id": question_id,
            "chart_type": chart_type.value,
            "x_column": x_column,
            "y_column": y_column,
            "aggregation": aggregation.value if aggregation else None,
            "ranking": ranking.value if ranking else None,
            "limit": limit,
            "category_transform": category_transform.value if category_transform else None,
            "category_policy": category_policy.value if category_policy else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    spec_id = f"question-chart-{sha256(material.encode('utf-8')).hexdigest()[:16]}"
    label = "Question" if source is QuestionSource.AUTHORED else "Recommendation"
    title = f"{label} {source_index + 1} · {chart_type.value.title()}"
    return QuestionResolution(
        question_id=question_id,
        source=source,
        source_index=source_index,
        status=QuestionResolutionStatus.RESOLVED,
        reason_code="QUESTION_RESOLVED",
        chart=QuestionChartSpec(
            spec_id=spec_id,
            question_id=question_id,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            aggregation=aggregation,
            ranking=ranking,
            limit=limit,
            category_transform=category_transform,
            category_policy=category_policy,
            title=title,
        ),
    )


def _unresolved(
    question_id: str,
    source: QuestionSource,
    source_index: int,
    status: QuestionResolutionStatus,
    reason_code: str,
) -> QuestionResolution:
    return QuestionResolution(
        question_id=question_id,
        source=source,
        source_index=source_index,
        status=status,
        reason_code=reason_code,
    )


def _compile_kind(
    frame: pd.DataFrame,
    question_id: str,
    source: QuestionSource,
    source_index: int,
    kind: QuestionKind,
    columns: tuple[str, ...],
    aggregation: ChartAggregation | None = None,
    ranking: ChartRanking | None = None,
    limit: int | None = None,
) -> QuestionResolution:
    available = frozenset(str(column) for column in frame.columns)
    if not columns or any(column not in available for column in columns):
        return _unresolved(
            question_id,
            source,
            source_index,
            QuestionResolutionStatus.QUESTION_UNSUPPORTED,
            "QUESTION_COLUMN_UNAVAILABLE",
        )
    numeric = tuple(column for column in columns if is_numeric_dtype(frame[column].dtype))
    nonnumeric = tuple(column for column in columns if column not in numeric)
    time_columns = tuple(column for column in columns if _is_time_column(frame, column))

    if aggregation is ChartAggregation.COUNT:
        if len(nonnumeric) == 1 and len(columns) == 1:
            return _make_chart(
                question_id, source, source_index, QuestionChartType.BAR,
                nonnumeric[0], aggregation=aggregation,
            )
        return _unresolved(
            question_id, source, source_index, QuestionResolutionStatus.QUESTION_NEEDS_INPUT,
            "QUESTION_AGGREGATION_AMBIGUOUS",
        )
    if aggregation is not None:
        if len(numeric) == 1 and len(nonnumeric) == 1 and len(columns) == 2:
            return _make_chart(
                question_id, source, source_index, QuestionChartType.BAR,
                nonnumeric[0], numeric[0], aggregation, ranking, limit,
            )
        return _unresolved(
            question_id, source, source_index, QuestionResolutionStatus.QUESTION_NEEDS_INPUT,
            "QUESTION_AGGREGATION_AMBIGUOUS",
        )
    if kind is QuestionKind.TREND and len(time_columns) == 1 and len(numeric) == 1:
        return _make_chart(
            question_id, source, source_index, QuestionChartType.LINE,
            time_columns[0], numeric[0],
        )
    if kind is QuestionKind.DISTRIBUTION and len(numeric) == 1 and len(columns) == 1:
        return _make_chart(
            question_id, source, source_index, QuestionChartType.HISTOGRAM, numeric[0]
        )
    if kind is QuestionKind.RELATIONSHIP and len(numeric) == 2 and len(columns) == 2:
        return _make_chart(
            question_id, source, source_index, QuestionChartType.SCATTER,
            numeric[0], numeric[1],
        )
    if (
        kind is QuestionKind.CATEGORY_COMPARISON
        and len(numeric) == 1
        and len(nonnumeric) == 1
        and len(columns) == 2
    ):
        return _make_chart(
            question_id, source, source_index, QuestionChartType.BOX,
            nonnumeric[0], numeric[0],
        )
    if kind in {QuestionKind.MISSINGNESS, QuestionKind.DUPLICATE_KEYS}:
        return _unresolved(
            question_id, source, source_index, QuestionResolutionStatus.QUESTION_UNSUPPORTED,
            "QUESTION_KIND_UNSUPPORTED",
        )
    return _unresolved(
        question_id, source, source_index, QuestionResolutionStatus.QUESTION_NEEDS_INPUT,
        "QUESTION_COLUMNS_AMBIGUOUS",
    )


def _compile_authored(
    frame: pd.DataFrame,
    question: str,
    source_index: int,
    unavailable_columns: tuple[str, ...] = (),
) -> QuestionResolution:
    question_id = _question_id(question, source_index)
    available = tuple(str(column) for column in frame.columns)
    columns = _mentioned_columns(question, available)
    unavailable = _mentioned_columns(
        question,
        unavailable_columns,
        include_semantic=False,
    )
    if unavailable:
        return _unresolved(
            question_id,
            QuestionSource.AUTHORED,
            source_index,
            QuestionResolutionStatus.QUESTION_UNSUPPORTED,
            "QUESTION_COLUMN_UNAVAILABLE",
        )
    aggregation = next(
        (aggregation for pattern, aggregation in _AGGREGATIONS if pattern.search(question)),
        None,
    )
    if _DAY_OF_WEEK.search(question):
        time_columns = tuple(
            column for column in available if _is_time_column(frame, column)
        )
        measures = tuple(
            column
            for column in columns
            if is_numeric_dtype(frame[column].dtype) and column not in time_columns
        )
        if len(time_columns) == 1 and len(measures) == 1:
            return _make_chart(
                question_id,
                QuestionSource.AUTHORED,
                source_index,
                QuestionChartType.BAR,
                time_columns[0],
                measures[0],
                ChartAggregation.SUM,
                ChartRanking.DESCENDING,
                7,
                ChartCategoryTransform.DAY_OF_WEEK,
            )
        return _unresolved(
            question_id,
            QuestionSource.AUTHORED,
            source_index,
            QuestionResolutionStatus.QUESTION_NEEDS_INPUT,
            "QUESTION_COLUMNS_AMBIGUOUS",
        )
    if _RANKING.search(question):
        numeric = tuple(
            column for column in columns if is_numeric_dtype(frame[column].dtype)
        )
        semantic_measures = tuple(
            column
            for column in numeric
            if _MEASURE_TOKENS.intersection(_identifier_tokens(column))
        )
        measure_candidates = semantic_measures or numeric
        if len(measure_candidates) == 1:
            measure = measure_candidates[0]
            categories = tuple(column for column in columns if column != measure)
        else:
            measure = ""
            categories = ()
        if measure and len(categories) == 1:
            lowered = question.casefold()
            sum_semantics = bool(
                re.search(r"\b(?:sum|total|profitable|profitability|sell|sales|quantity)\b", lowered)
                and not re.search(r"\b(?:highest\s+prices?|most\s+expensive|stars?|ratings?)\b", lowered)
            )
            top_match = _TOP_N.search(question)
            if top_match is not None and int(top_match.group(1)) < 1:
                return _unresolved(
                    question_id,
                    QuestionSource.AUTHORED,
                    source_index,
                    QuestionResolutionStatus.QUESTION_NEEDS_INPUT,
                    "QUESTION_RANKING_LIMIT_INVALID",
                )
            limit = min(int(top_match.group(1)), 50) if top_match else 10
            return _make_chart(
                question_id,
                QuestionSource.AUTHORED,
                source_index,
                QuestionChartType.BAR,
                categories[0],
                measure,
                ChartAggregation.SUM if sum_semantics else ChartAggregation.MAX,
                ChartRanking.DESCENDING,
                limit,
                category_policy=(
                    ChartCategoryPolicy.EXCLUDE_PLACEHOLDERS
                    if _PLACEHOLDER_EXCLUSION.search(question)
                    else None
                ),
            )
        return _unresolved(
            question_id,
            QuestionSource.AUTHORED,
            source_index,
            QuestionResolutionStatus.QUESTION_NEEDS_INPUT,
            "QUESTION_COLUMNS_AMBIGUOUS",
        )
    if aggregation is not None:
        return _compile_kind(
            frame, question_id, QuestionSource.AUTHORED, source_index, QuestionKind.CATEGORY_COMPARISON,
            columns, aggregation,
        )
    if _TREND.search(question):
        kind = QuestionKind.TREND
    elif _DISTRIBUTION.search(question):
        kind = QuestionKind.DISTRIBUTION
    elif _RELATIONSHIP.search(question):
        kind = QuestionKind.RELATIONSHIP
    elif _COMPARISON.search(question):
        kind = QuestionKind.CATEGORY_COMPARISON
    else:
        return _unresolved(
            question_id, QuestionSource.AUTHORED, source_index,
            QuestionResolutionStatus.QUESTION_NEEDS_INPUT,
            "QUESTION_INTENT_AMBIGUOUS",
        )
    return _compile_kind(
        frame, question_id, QuestionSource.AUTHORED, source_index, kind, columns
    )


def compile_question_charts(
    gold: pd.DataFrame,
    authored_questions: tuple[str, ...],
    selected_questions: tuple[SuggestedQuestion, ...],
    *,
    unavailable_columns: tuple[str, ...] = (),
) -> tuple[QuestionResolution, ...]:
    """Compile questions against the current verified schema without reading cells."""

    if not isinstance(gold, pd.DataFrame):
        raise TypeError("question compilation requires a DataFrame")
    results = [
        _compile_authored(gold, question, index, unavailable_columns)
        for index, question in enumerate(authored_questions)
    ]
    for relative_index, question in enumerate(selected_questions):
        results.append(
            _compile_kind(
                gold,
                question.question_id,
                QuestionSource.AUTOMATIC,
                relative_index,
                question.kind,
                question.relevant_columns,
            )
        )
    return tuple(results)


__all__ = [
    "ChartAggregation",
    "ChartCategoryPolicy",
    "ChartCategoryTransform",
    "ChartRanking",
    "QuestionChartSpec",
    "QuestionChartType",
    "QuestionResolution",
    "QuestionResolutionStatus",
    "QuestionSource",
    "compile_question_charts",
]
