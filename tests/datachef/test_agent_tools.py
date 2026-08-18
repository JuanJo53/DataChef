from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from datachef.agents.tools import (
    ALIASED,
    MISSING_COLUMN,
    UNKNOWN_ISSUE,
    UNSUPPORTED_REQUEST_MAX_LENGTH,
    ReportUnsupportedRequestArgs,
    CastColumnArgs,
    DeduplicateByKeysArgs,
    DropDuplicateRowsArgs,
    NormalizeMissingTokensArgs,
    PlanDraft,
    RenameColumnArgs,
    TrimWhitespaceArgs,
    apply_operation_args,
    build_operation_specs,
    discard_last_operation,
    estimate_current_plan,
    finalize_plan,
    inspect_profile,
    report_unsupported_request,
)
from datachef.contracts import (
    CastTarget,
    DiagnosticIssueKind,
    OperationType,
    UserIntent,
)
from datachef.diagnostics import diagnose_raw_dataframe
from datachef.planning import validate_plan
from datachef.planning.plan import create_transformation_plan
from datachef.privacy import build_column_alias_map, build_planning_context
from datachef.transform.operations import OPERATION_CATALOGUE


def _context(row_loss: float = 0.0):
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 2],
            "imgUrl": ["u", "v", "u"],
            "amount": [1.0, 2.0, 2.0],
        }
    )
    intent = UserIntent(
        intent_id="intent-agent",
        user_goal="Prepare for analysis.",
        selected_key_columns=("order_id",),
        acceptable_row_loss_pct=row_loss,
    )
    report = diagnose_raw_dataframe(frame, selected_key_columns=("order_id",))
    alias_map = build_column_alias_map(report, intent)
    return build_planning_context(report, intent, (), column_alias_map=alias_map)


def _issue_id(context, kind: DiagnosticIssueKind) -> str:
    return next(
        issue.issue_id
        for issue in context.diagnostic_report.issues
        if issue.kind is kind
    )


def test_every_executable_operation_type_has_a_proposal_tool() -> None:
    specs = build_operation_specs()

    assert {operation_type for _, operation_type, _ in specs} == set(OperationType)
    assert len(specs) == len(OPERATION_CATALOGUE) == 6
    assert all(definition.handler for definition in OPERATION_CATALOGUE.values())


def test_an_aliased_target_is_refused_in_loop_without_touching_the_draft() -> None:
    context = _context()
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)
    assert context.privacy_manifest.aliased_columns == ("__dc_private_002__",)

    result = apply_operation_args(
        draft,
        "propose_trim_whitespace",
        TrimWhitespaceArgs(
            target_columns=["__dc_private_002__"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )

    assert result == {"accepted": False, "reason_code": ALIASED}
    assert draft.operations == []
    assert draft.invocations[-1].accepted is False
    assert draft.invocations[-1].reason_code == ALIASED


def test_an_ordinary_column_proposal_produces_a_plan_the_validator_accepts() -> None:
    context = _context(row_loss=50.0)
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)

    result = apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["order_id"],
            diagnostic_issue_ids=[issue],
            rationale="Deterministic duplicate evidence.",
            expected_effect="Keep the first row per key.",
        ),
    )

    assert result["accepted"] is True
    validation = validate_plan(context, draft.build_plan())
    assert validation.valid, [item.code for item in validation.findings]


def test_the_committed_validator_still_catches_an_aliased_column() -> None:
    """Defence in depth: the tool guard is not the only guard."""

    context = _context(row_loss=50.0)
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)
    apply_operation_args(
        draft,
        "propose_trim_whitespace",
        TrimWhitespaceArgs(
            target_columns=["amount"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )
    smuggled = draft.operations[0].model_copy(
        update={"target_columns": ("__dc_private_002__",)}
    )
    plan = create_transformation_plan(
        dataset_id=context.dataset_identity.dataset_id,
        dataset_fingerprint=context.dataset_identity.fingerprint,
        version=1,
        operations=(smuggled,),
        summary="smuggled",
    )

    validation = validate_plan(context, plan)

    assert validation.valid is False
    assert "ALIASED_COLUMN_NOT_EXECUTABLE" in [item.code for item in validation.findings]


def test_a_hallucinated_column_is_refused_in_loop() -> None:
    context = _context()
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)

    result = apply_operation_args(
        draft,
        "propose_trim_whitespace",
        TrimWhitespaceArgs(
            target_columns=["no_such_column"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )

    assert result == {"accepted": False, "reason_code": MISSING_COLUMN}
    assert draft.operations == []


def test_an_unknown_diagnostic_issue_is_refused_in_loop() -> None:
    context = _context()
    draft = PlanDraft(context=context)

    result = apply_operation_args(
        draft,
        "propose_trim_whitespace",
        TrimWhitespaceArgs(
            target_columns=["amount"],
            diagnostic_issue_ids=["issue-invented"],
            rationale="r",
            expected_effect="e",
        ),
    )

    assert result == {"accepted": False, "reason_code": UNKNOWN_ISSUE}
    assert draft.operations == []


@pytest.mark.parametrize(
    "payload",
    (
        {"operation_type": "DROP_TABLE", "target_columns": ["amount"]},
        {"target_columns": ["amount"], "sql": "DROP TABLE orders"},
        {"target_columns": [], "rationale": "r", "expected_effect": "e"},
    ),
)
def test_an_out_of_allow_list_call_is_unrepresentable(payload: dict) -> None:
    with pytest.raises(ValidationError):
        TrimWhitespaceArgs(**payload)


def test_a_forged_plan_id_is_caught_and_the_whole_plan_rejected() -> None:
    context = _context(row_loss=50.0)
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)
    apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["order_id"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )
    forged = draft.build_plan().model_copy(update={"plan_id": "plan-agent-authored"})

    validation = validate_plan(context, forged)

    assert validation.valid is False
    assert "PLAN_ID_MISMATCH" in [item.code for item in validation.findings]


def test_a_forged_dataset_binding_is_caught() -> None:
    context = _context(row_loss=50.0)
    draft = PlanDraft(context=context)
    forged = create_transformation_plan(
        dataset_id="dataset-not-ours",
        dataset_fingerprint="0" * 64,
        version=1,
        operations=(),
        summary="forged",
    )

    codes = [item.code for item in validate_plan(context, forged).findings]

    assert "DATASET_ID_MISMATCH" in codes
    assert "DATASET_FINGERPRINT_MISMATCH" in codes


def test_operation_ids_are_computed_by_us_not_supplied_by_the_agent() -> None:
    context = _context(row_loss=50.0)
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)

    apply_operation_args(
        draft,
        "propose_trim_whitespace",
        TrimWhitespaceArgs(
            target_columns=["amount"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )
    apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["order_id"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )

    assert [item.operation_id for item in draft.operations] == [
        "op-001-trim_whitespace",
        "op-002-deduplicate_by_keys",
    ]
    assert "operation_id" not in TrimWhitespaceArgs.model_fields
    assert "plan_id" not in TrimWhitespaceArgs.model_fields
    assert "dataset_fingerprint" not in TrimWhitespaceArgs.model_fields


def test_inspect_profile_returns_schema_and_codes_but_no_cell_values() -> None:
    context = _context()
    draft = PlanDraft(context=context)

    profile = inspect_profile(draft)

    assert [item["name"] for item in profile["columns"]] == [
        "order_id",
        "__dc_private_002__",
        "amount",
    ]
    assert profile["aliased_columns"] == ["__dc_private_002__"]
    rendered = repr(profile)
    for cell_value in ("'u'", "'v'", "1.0", "2.0"):
        assert f": {cell_value}" not in rendered


def test_estimate_current_plan_reports_findings_with_context_names_only() -> None:
    context = _context(row_loss=0.0)
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)
    apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["order_id"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )

    estimate = estimate_current_plan(draft)

    assert estimate["valid"] is False
    assert [item["code"] for item in estimate["findings"]] == [
        "ROW_LOSS_THRESHOLD",
        "CUMULATIVE_ROW_LOSS_THRESHOLD",
    ]
    assert estimate["cumulative_estimated_row_loss_pct"] == pytest.approx(33.3333, rel=1e-3)
    assert estimate["acceptable_row_loss_pct"] == 0.0
    assert "imgUrl" not in repr(estimate)


def test_the_critic_in_loop_lets_the_agent_revise_within_one_run() -> None:
    """Propose destructively, price it, retract, and finish valid."""

    context = _context(row_loss=0.0)
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)

    proposed = apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["order_id"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )
    priced = estimate_current_plan(draft)
    retracted = discard_last_operation(draft)
    repriced = estimate_current_plan(draft)
    finalized = finalize_plan(draft, "Leave the table unchanged.")

    assert proposed["accepted"] is True
    assert priced["valid"] is False
    assert "ROW_LOSS_THRESHOLD" in [item["code"] for item in priced["findings"]]
    assert retracted["accepted"] is True
    assert repriced["valid"] is True
    assert repriced["operation_count"] == 0
    assert finalized == {"accepted": True, "operation_count": 0}
    assert validate_plan(context, draft.build_plan()).valid is True


def test_finalize_refuses_an_invalid_draft_with_codes_only() -> None:
    context = _context(row_loss=0.0)
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)
    apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["order_id"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )

    refusal = finalize_plan(draft, "Deduplicate orders.")

    assert refusal["accepted"] is False
    assert refusal["reason_codes"] == [
        "ROW_LOSS_THRESHOLD",
        "CUMULATIVE_ROW_LOSS_THRESHOLD",
    ]


@pytest.mark.parametrize(
    ("tool_name", "args"),
    (
        ("propose_cast_column", CastColumnArgs(
            target_columns=["amount"], target_type=CastTarget.NUMERIC,
            diagnostic_issue_ids=["x"], rationale="r", expected_effect="e")),
        ("propose_rename_column", RenameColumnArgs(
            target_columns=["amount"], new_name="total",
            diagnostic_issue_ids=["x"], rationale="r", expected_effect="e")),
        ("propose_normalize_missing_tokens", NormalizeMissingTokensArgs(
            target_columns=["amount"], tokens=["NA"],
            diagnostic_issue_ids=["x"], rationale="r", expected_effect="e")),
        ("propose_drop_duplicate_rows", DropDuplicateRowsArgs(
            diagnostic_issue_ids=["x"], rationale="r", expected_effect="e")),
    ),
)
def test_each_remaining_tool_routes_through_the_same_guarded_path(tool_name, args) -> None:
    context = _context(row_loss=50.0)
    draft = PlanDraft(context=context)

    result = apply_operation_args(draft, tool_name, args)

    # The invented issue id is refused by the same guard for every tool.
    assert result == {"accepted": False, "reason_code": UNKNOWN_ISSUE}
    assert draft.operations == []


def test_a_dedup_on_a_column_with_no_duplicate_key_metric_is_refused() -> None:
    """The estimator can only price nominated key sets (finding: row-loss blind spot)."""

    context = _context(row_loss=50.0)
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)
    assert [item.key_columns for item in context.diagnostic_report.key_duplicate_metrics] == [
        ("order_id",)
    ]

    result = apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["amount"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )

    assert result == {"accepted": False, "reason_code": "INVALID_KEY"}
    assert draft.operations == []
    assert draft.invocations[-1].reason_code == "INVALID_KEY"


def test_a_dedup_on_a_nominated_key_set_still_succeeds() -> None:
    context = _context(row_loss=50.0)
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)

    result = apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["order_id"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )

    assert result["accepted"] is True
    assert len(draft.operations) == 1


def test_the_trace_records_inspect_estimate_and_discard_in_order() -> None:
    context = _context(row_loss=0.0)
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)

    inspect_profile(draft)
    apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["order_id"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )
    estimate_current_plan(draft)
    discard_last_operation(draft)
    finalize_plan(draft, "Leave the table unchanged.")

    assert [item.tool_name for item in draft.invocations] == [
        "inspect_profile",
        "propose_deduplicate_by_keys",
        "estimate_current_plan",
        "discard_last_operation",
        "finalize_plan",
    ]
    critic = draft.invocations[2]
    assert critic.accepted is False
    assert "ROW_LOSS_THRESHOLD" in critic.critic_finding_codes
    assert critic.estimated_row_loss_pct == pytest.approx(33.3333, rel=1e-3)


def test_the_trace_leaks_no_cell_value_key_path_or_exception_text() -> None:
    context = _context(row_loss=0.0)
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)
    inspect_profile(draft)
    apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["order_id"],
            diagnostic_issue_ids=[issue],
            rationale="r",
            expected_effect="e",
        ),
    )
    estimate_current_plan(draft)

    rendered = " ".join(item.model_dump_json() for item in draft.invocations)

    for leak in ("'u'", "'v'", "AIza", "C:\\", "C:/", "Traceback", "gemini-", "imgUrl"):
        assert leak not in rendered


def test_report_unsupported_request_records_without_touching_the_plan() -> None:
    context = _context()
    draft = PlanDraft(context=context)
    issue = _issue_id(context, DiagnosticIssueKind.DUPLICATE_KEYS)
    apply_operation_args(
        draft,
        "propose_deduplicate_by_keys",
        DeduplicateByKeysArgs(
            keys=["order_id"],
            diagnostic_issue_ids=[issue],
            rationale="Duplicate keys were reported.",
            expected_effect="Removes duplicate order rows.",
        ),
    )
    before_plan = draft.build_plan()
    before_validation = validate_plan(context, before_plan)

    result = report_unsupported_request(
        draft,
        "mean imputation for the amount column",
    )

    assert result == {"accepted": True, "recorded_count": 1}
    assert draft.unsupported_requests == ["mean imputation for the amount column"]
    # The plan, its identity, and its validation are all untouched.
    after_plan = draft.build_plan()
    assert after_plan == before_plan
    assert after_plan.plan_id == before_plan.plan_id
    assert validate_plan(context, after_plan) == before_validation
    assert len(draft.operations) == 1


def test_report_unsupported_request_records_the_call_in_the_trace() -> None:
    draft = PlanDraft(context=_context())

    report_unsupported_request(draft, "drop the imgUrl column")

    invocation = draft.invocations[-1]
    assert invocation.tool_name == "report_unsupported_request"
    assert invocation.accepted is True
    # The trace carries codes and tool names, never model prose.
    assert "drop the imgUrl column" not in invocation.model_dump_json()
    assert invocation.operation_type is None
    assert invocation.target_columns == ()


def test_unsupported_request_free_text_is_length_bounded_at_the_boundary() -> None:
    assert UNSUPPORTED_REQUEST_MAX_LENGTH == 240
    field = ReportUnsupportedRequestArgs.model_fields["description"]
    assert field.metadata

    with pytest.raises(ValidationError):
        ReportUnsupportedRequestArgs(description="x" * (UNSUPPORTED_REQUEST_MAX_LENGTH + 1))
    with pytest.raises(ValidationError):
        ReportUnsupportedRequestArgs(description="")

    accepted = ReportUnsupportedRequestArgs(
        description="x" * UNSUPPORTED_REQUEST_MAX_LENGTH
    )
    assert len(accepted.description) == UNSUPPORTED_REQUEST_MAX_LENGTH


def test_unsupported_request_text_passes_through_the_existing_sanitizer() -> None:
    draft = PlanDraft(context=_context())

    report_unsupported_request(
        draft,
        "email the result to analyst@example.com when done",
    )

    recorded = draft.unsupported_requests[0]
    assert "analyst@example.com" not in recorded
    assert "[REDACTED_EMAIL]" in recorded


def test_unsupported_request_truncates_and_refuses_empty_text() -> None:
    draft = PlanDraft(context=_context())

    refused = report_unsupported_request(draft, "   ")

    assert refused["accepted"] is False
    assert draft.unsupported_requests == []
    assert draft.invocations[-1].accepted is False


def test_repeated_unsupported_request_is_recorded_once() -> None:
    draft = PlanDraft(context=_context())

    report_unsupported_request(draft, "median imputation")
    report_unsupported_request(draft, "median imputation")

    assert draft.unsupported_requests == ["median imputation"]


def test_unsupported_requests_do_not_widen_the_executable_tool_surface() -> None:
    """The allow-list is unchanged: reporting is not proposing."""

    specs = build_operation_specs()

    assert {operation_type for _, operation_type, _ in specs} == set(OperationType)
    assert len(specs) == 6
    assert "report_unsupported_request" not in {name for name, _, _ in specs}
