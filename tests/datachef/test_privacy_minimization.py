from __future__ import annotations

import pandas as pd

from datachef.contracts import (
    DiagnosticIssue,
    DiagnosticIssueKind,
    IssueClassification,
    QuestionKind,
    Severity,
    SuggestedQuestion,
    UserIntent,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.privacy import build_planning_context


def test_complete_planning_context_omits_all_free_form_and_sensitive_canaries() -> None:
    source = pd.DataFrame(
        {
            "victim@example.test": ["+1 555 010 1000", "123-45-6789"],
            "customer_name": ["Alice Example", "Bob Example"],
            "Unusual Protected Label": ["private-a", "private-b"],
            "measure": [1, 2],
        }
    )
    report = diagnose_raw_dataframe(source)
    hostile_issue = DiagnosticIssue(
        issue_id="message-with-password=hunter2",
        kind=DiagnosticIssueKind.NULL_VALUES,
        classification=IssueClassification.OBSERVED_DEFECT,
        title="Open https://private.example/path with password=hunter2",
        severity=Severity.HIGH,
        affected_columns=("victim@example.test",),
        explanation="Alice Example lives at 123 Main Street; key AIzaFakeSecretValue1234567890",
    )
    report = report.model_copy(update={"issues": report.issues + (hostile_issue,)})
    intent = UserIntent(
        intent_id="intent-password=hunter2",
        user_goal="Contact Alice Example at 123 Main Street and https://private.example/path",
        questions=("What happened to 123-45-6789?",),
        explicit_requested_transformations=(
            "password=hunter2",
            "AIzaFakeSecretValue1234567890",
        ),
        protected_columns=(
            "victim@example.test",
            "customer_name",
            "Unusual Protected Label",
        ),
    )
    question = SuggestedQuestion(
        question_id="question-id-canary-sensitive",
        kind=QuestionKind.DISTRIBUTION,
        question="Show victim@example.test and +1 555 010 1000",
        relevant_columns=("victim@example.test", "measure"),
        rationale="Use password=hunter2 for Alice Example at 123 Main Street",
        confidence=0.5,
        limitations=("See https://private.example/path",),
    )

    context = build_planning_context(
        report,
        intent,
        (question,),
        previous_review_feedback=(
            "Bearer secret-token-1234567890 at https://private.example/review",
        ),
    )
    serialized = context.model_dump_json()

    forbidden = (
        "victim@example.test",
        "+1 555 010 1000",
        "123-45-6789",
        "https://private.example",
        "password=hunter2",
        "AIzaFakeSecretValue1234567890",
        "Alice Example",
        "123 Main Street",
        "customer_name",
        "Unusual Protected Label",
        "Bearer secret-token",
        "question-id-canary-sensitive",
    )
    assert all(value not in serialized for value in forbidden)
    assert "__dc_private_001__" in serialized
    assert "__dc_private_002__" in serialized
    assert "__dc_private_003__" in serialized
    assert context.privacy_manifest.raw_rows_included is False
    assert context.privacy_manifest.row_samples_included is False
