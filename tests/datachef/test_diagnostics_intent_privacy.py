from __future__ import annotations

from pandas.testing import assert_frame_equal

from datachef.contracts import DiagnosticIssueKind, QuestionKind, UserIntent
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.intent import discover_questions
from datachef.privacy import build_planning_context


def test_raw_diagnosis_does_not_mutate_source(raw_dataframe) -> None:
    before = raw_dataframe.copy(deep=True)

    report = diagnose_raw_dataframe(
        raw_dataframe,
        selected_key_columns=("customer_id",),
    )

    assert_frame_equal(raw_dataframe, before)
    assert report.dataset_identity.row_count == 4
    assert report.dataset_identity.fingerprint


def test_duplicate_key_values_are_diagnosed(raw_dataframe) -> None:
    report = diagnose_raw_dataframe(
        raw_dataframe,
        selected_key_columns=("customer_id",),
    )

    issue = next(
        item for item in report.issues if item.kind is DiagnosticIssueKind.DUPLICATE_KEYS
    )
    assert issue.affected_columns == ("customer_id",)
    assert issue.evidence[0].value == 1
    assert "no row has been removed" in issue.explanation


def test_missing_selected_key_is_diagnosed_immediately(raw_dataframe) -> None:
    report = diagnose_raw_dataframe(
        raw_dataframe,
        selected_key_columns=("missing_key",),
    )

    issue = next(
        item
        for item in report.issues
        if item.kind is DiagnosticIssueKind.MISSING_KEY_COLUMN
    )
    assert issue.affected_columns == ("missing_key",)
    assert issue.evidence[0].value == 1


def test_empty_goal_produces_bounded_grounded_questions(raw_dataframe) -> None:
    report = diagnose_raw_dataframe(
        raw_dataframe,
        selected_key_columns=("customer_id",),
    )

    suggestions = discover_questions(report)

    assert 1 <= len(suggestions) <= 5
    known_columns = set(raw_dataframe.columns)
    assert all(set(item.relevant_columns) <= known_columns for item in suggestions)
    assert any(item.kind is QuestionKind.DUPLICATE_KEYS for item in suggestions)
    assert all(item.limitations for item in suggestions)


def test_planning_context_contains_no_raw_pii_values(
    raw_dataframe,
    user_intent: UserIntent,
) -> None:
    report = diagnose_raw_dataframe(
        raw_dataframe,
        selected_key_columns=user_intent.selected_key_columns,
    )
    questions = discover_questions(report)
    intent_with_pii_text = UserIntent.model_validate(
        {
            **user_intent.model_dump(),
            "user_goal": (
                "Analyze fictional.one@example.test and +1 555 010 1000 "
                "for identifier ABCDEFGHIJKLMNOPQRSTUVWX"
            ),
        }
    )

    context = build_planning_context(report, intent_with_pii_text, questions)
    serialized = context.model_dump_json()

    for value in raw_dataframe["email"].tolist() + raw_dataframe["phone"].tolist():
        assert value not in serialized
    assert "ABCDEFGHIJKLMNOPQRSTUVWX" not in serialized
    assert context.privacy_manifest.raw_rows_included is False
    assert context.privacy_manifest.row_samples_included is False
    assert intent_with_pii_text.user_goal not in serialized
