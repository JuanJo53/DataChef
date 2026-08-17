"""Bounded, schema-grounded analytical question suggestions."""

from __future__ import annotations

from hashlib import sha256

import pandas as pd

from datachef.contracts import (
    DiagnosticIssueKind,
    DiagnosticReport,
    QuestionKind,
    SuggestedQuestion,
)


def _question_id(kind: QuestionKind, columns: tuple[str, ...]) -> str:
    digest = sha256(f"{kind.value}|{'|'.join(columns)}".encode()).hexdigest()
    return f"question-{digest[:12]}"


def _is_identifier(name: str) -> bool:
    lowered = name.lower()
    return lowered == "id" or lowered.endswith("_id")


def discover_questions(
    report: DiagnosticReport,
    *,
    maximum: int = 5,
) -> tuple[SuggestedQuestion, ...]:
    """Suggest answerable question shapes without inferring domain semantics."""

    maximum = max(0, min(maximum, 5))
    numeric = tuple(
        profile.name
        for profile in report.column_profiles
        if pd.api.types.is_numeric_dtype(profile.dtype)
        and not _is_identifier(profile.name)
    )
    temporal = tuple(
        profile.name
        for profile in report.column_profiles
        if pd.api.types.is_datetime64_any_dtype(profile.dtype)
        or any(token in profile.name.lower() for token in ("date", "time", "year"))
    )
    categorical = tuple(
        profile.name
        for profile in report.column_profiles
        if not pd.api.types.is_numeric_dtype(profile.dtype)
        and profile.name not in temporal
        and not profile.possible_pii
    )
    suggestions: list[SuggestedQuestion] = []

    def add(
        kind: QuestionKind,
        question: str,
        columns: tuple[str, ...],
        rationale: str,
        confidence: float,
        limitations: tuple[str, ...],
    ) -> None:
        if columns and len(suggestions) < maximum:
            suggestions.append(
                SuggestedQuestion(
                    question_id=_question_id(kind, columns),
                    kind=kind,
                    question=question,
                    relevant_columns=columns,
                    rationale=rationale,
                    confidence=confidence,
                    limitations=limitations,
                )
            )

    if temporal and numeric:
        add(
            QuestionKind.TREND,
            f"How does {numeric[0]} vary over {temporal[0]}?",
            (temporal[0], numeric[0]),
            "The schema contains a possible time field and numeric measure.",
            0.8,
            ("The time-like column still requires deterministic parsing validation.",),
        )
    if categorical and numeric:
        add(
            QuestionKind.CATEGORY_COMPARISON,
            f"How does {numeric[0]} differ across {categorical[0]} groups?",
            (categorical[0], numeric[0]),
            "A non-PII category and numeric measure are available.",
            0.75,
            ("Column names alone do not establish business meaning.",),
        )
    null_columns = tuple(
        issue.affected_columns[0]
        for issue in report.issues
        if issue.kind is DiagnosticIssueKind.NULL_VALUES and issue.affected_columns
    )
    if null_columns:
        add(
            QuestionKind.MISSINGNESS,
            "Which columns have the largest missing-data rates?",
            null_columns,
            "The deterministic profile observed null values.",
            0.95,
            ("Missingness patterns do not explain why values are absent.",),
        )
    duplicate_key_columns = next(
        (
            issue.affected_columns
            for issue in report.issues
            if issue.kind is DiagnosticIssueKind.DUPLICATE_KEYS
        ),
        (),
    )
    if duplicate_key_columns:
        add(
            QuestionKind.DUPLICATE_KEYS,
            "Which selected keys occur in more than one row?",
            duplicate_key_columns,
            "The deterministic diagnosis observed duplicate key values.",
            0.98,
            ("Duplicate keys may be valid unless the user confirms uniqueness.",),
        )
    if numeric:
        add(
            QuestionKind.DISTRIBUTION,
            f"What is the distribution of {numeric[0]}?",
            (numeric[0],),
            "The column has a numeric dtype suitable for descriptive statistics.",
            0.9,
            ("A distribution describes values but does not explain their cause.",),
        )
    if len(numeric) >= 2:
        add(
            QuestionKind.RELATIONSHIP,
            f"What relationship, if any, appears between {numeric[0]} and {numeric[1]}?",
            (numeric[0], numeric[1]),
            "Two numeric measures can be compared descriptively.",
            0.65,
            ("Association must not be described as causation.",),
        )
    return tuple(suggestions)
